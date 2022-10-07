# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Model subclass Pipelining API interface."""
from collections import Counter
import itertools
from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig

from keras.ipu import extensions
from keras import layers
from keras.engine.functional import Functional
from keras.engine.sequential import Sequential
from keras.engine.training import Model


class ModelSubclass(Model):  # pylint: disable=abstract-method
  def __init__(self, layers_list, name=None):
    super().__init__(name=name)
    self.layers_list = layers_list

  def call(self, inputs):  # pylint: disable=arguments-differ
    x = inputs
    for layer in self.layers_list:
      x = layer(x)
    return x


class LayerDef():
  def __init__(self, layer_class, args=None, stage=None, name=None):
    self.layer_class = layer_class
    self.args = [] if args is None else args
    self.stage = stage
    self.name = name

  def flatten(self):
    return [self]

  def build(self):
    if self.stage is not None:
      with extensions.pipeline_stage_assignment.PipelineStage(self.stage):
        return self.layer_class(*(self.args), name=self.name)
    return self.layer_class(*(self.args), name=self.name)


class ModelDef():
  def __init__(self, model_classes, children, input_shape=(32,), name=None):
    self.model_classes = model_classes
    self.children = children
    self.input_shape = input_shape
    self.name = name

  def flatten(self):
    # Generate every flat (every ModelDef in the tree has one class) permutation
    # of this model def, preserving repeated (same instance) children.

    # Handle when this model has multiple classes flatten each separately and
    # concatenate the results.
    if len(self.model_classes) > 1:
      return sum(
          (ModelDef([x], self.children, self.input_shape, self.name).flatten()
           for x in self.model_classes), [])

    # Map of each child to a list of its flat permutations.
    flattened_children = {x: x.flatten() for x in set(self.children)}

    # List of all permutations of 1:1 maps from def to flat def.
    map_perms = (dict(zip(flattened_children, x))
                 for x in itertools.product(*flattened_children.values()))

    # List of every flat permutation of the children in this model def.
    children_perms = ([m[l] for l in self.children] for m in map_perms)

    # Generate a model def for every flat permutation of this model.
    return [
        ModelDef(self.model_classes, children, self.input_shape, self.name)
        for children in children_perms
    ]

  def build(self):
    if len(self.model_classes) > 1:
      raise RuntimeError("Attempting to build a ModelDef which has not been "
                         "flattened.")
    model_class = self.model_classes[0]

    # Make sure repeated children use the same instance.
    built_children_map = {x: x.build() for x in set(self.children)}
    built_children = [built_children_map[x] for x in self.children]

    if model_class == Functional:
      inputs = layers.Input(self.input_shape)
      outputs = inputs
      for child in built_children:
        outputs = child(outputs)
      model = Functional(inputs, outputs, name=self.name)

    elif model_class == Sequential:
      inputs = [layers.Input(self.input_shape)]
      model = Sequential(inputs + built_children, name=self.name)

    else:
      model = model_class(built_children, name=self.name)

    # Build for batch size 1 (doesn't matter what value is used).
    model.build((1,) + self.input_shape)
    return model

  def get_df_model_classes(self):
    # Returns a depth-first sequence of the model classes in this model def.
    if len(self.model_classes) > 1:
      raise RuntimeError(
          "Attempting to get model classes from a model which has "
          "not been flattened.")
    return sum((x.get_df_model_classes()
                for x in Counter(self.children) if isinstance(x, ModelDef)),
               [self.model_classes[0]])


model_defs = {}

model_defs["simple"] = ModelDef(
    name="simple_outer_model",
    model_classes=[Functional, Sequential, ModelSubclass],
    children=[
        LayerDef(layers.Flatten),
        ModelDef(name="simple_inner_model",
                 model_classes=[Functional, Sequential, ModelSubclass],
                 children=[
                     LayerDef(layers.Dense, args=[32]),
                     LayerDef(layers.Dense, args=[16]),
                 ]),
        LayerDef(layers.Dense, args=[8]),
    ]).flatten()

model_defs["double"] = ModelDef(
    name="double_outer_model",
    model_classes=[Functional, Sequential, ModelSubclass],
    children=[
        LayerDef(layers.Flatten),
        ModelDef(name="double_inner_model_1",
                 model_classes=[Functional, Sequential, ModelSubclass],
                 children=[
                     LayerDef(layers.Dense, args=[32]),
                 ]),
        ModelDef(name="double_inner_model_2",
                 model_classes=[Functional, Sequential, ModelSubclass],
                 children=[
                     LayerDef(layers.Dense, args=[16]),
                 ]),
        LayerDef(layers.Dense, args=[8]),
    ]).flatten()

model_defs["three_level"] = ModelDef(
    name="three_level_outer_model",
    model_classes=[Functional, Sequential, ModelSubclass],
    children=[
        LayerDef(layers.Flatten),
        ModelDef(name="three_level_middle_model",
                 model_classes=[Functional, Sequential, ModelSubclass],
                 children=[
                     LayerDef(layers.Dense, args=[32]),
                     ModelDef(
                         name="triple_nested_inner_model",
                         model_classes=[Functional, Sequential, ModelSubclass],
                         children=[
                             LayerDef(layers.Dense, args=[16]),
                         ]),
                 ]),
        LayerDef(layers.Dense, args=[8])
    ]).flatten()

# Not used directly, used as part of "repeated".
model_defs["repeated_inner"] = ModelDef(
    name="repeated_inner_model",
    model_classes=[Functional, Sequential, ModelSubclass],
    input_shape=(16,),
    children=[
        LayerDef(layers.Dense, args=[16]),
    ])
# Sequential models don't support repeated layers.
model_defs["repeated"] = ModelDef(
    name="repeated_outer_model",
    model_classes=[Functional, ModelSubclass],
    children=[
        LayerDef(layers.Flatten),
        LayerDef(layers.Dense, args=[16]),
        model_defs["repeated_inner"],
        model_defs["repeated_inner"],
        model_defs["repeated_inner"],
        model_defs["repeated_inner"],
        LayerDef(layers.Dense, args=[8]),
    ],
).flatten()

model_defs["scoped_assignments"] = ModelDef(
    name="scoped_assignments outer model",
    model_classes=[Functional, ModelSubclass],
    children=[
        LayerDef(layers.Flatten, stage=0),
        ModelDef(
            name="scoped_assignments_inner_model",
            model_classes=[Functional, ModelSubclass],
            children=[
                LayerDef(layers.Dense, args=[32], stage=1),
                LayerDef(layers.Dense, args=[16], stage=2),
            ],
        ),
        LayerDef(layers.Dense, args=[8], stage=3),
    ]).flatten()


def generate_test_data(*model_def_names):
  result = []
  for name in model_def_names:
    for model_def in model_defs[name]:
      model_class_strings = [
          x.__name__ for x in model_def.get_df_model_classes()
      ]
      test_name = "_" + "_".join([name] + model_class_strings)
      result.append((test_name, model_def))
  return result


test_data = {
    "general": generate_test_data("simple", "double", "three_level",
                                  "repeated"),
    "scoped": generate_test_data("scoped_assignments"),
}


def set_assignments(model):
  assignments = model.get_pipeline_stage_assignment()

  def setup_recursive(assignments, n):
    for assignment in assignments:
      if assignment.is_nested_model:
        n = setup_recursive(assignment.pipeline_stage_assignments, n)
      else:
        assignment.pipeline_stage = n
        n += 1
    return n

  setup_recursive(assignments, 0)
  model.set_pipeline_stage_assignment(assignments)


def check_assignments(test, model):
  assignments = model.get_pipeline_stage_assignment()

  def check_recursive(assignments, n):
    for assignment in assignments:
      if isinstance(assignment,
                    (extensions.NestedModelPipelineStageAssignment,
                     extensions.FunctionalNestedModelPipelineStageAssignment,
                     extensions.SequentialNestedModelPipelineStageAssignment)):
        n = check_recursive(assignment.pipeline_stage_assignments, n)
      else:
        test.assertEqual(assignment.pipeline_stage, n)
        n += 1
    return n

  check_recursive(assignments, 0)


def check_default_assignments(test, model):
  assignments = model.get_pipeline_stage_assignment()

  def check_recursive(assignments):
    for assignment in assignments:
      if assignment.is_nested_model:
        check_recursive(assignment.pipeline_stage_assignments)
      else:
        test.assertEqual(assignment.pipeline_stage, None)

  check_recursive(assignments)


class NestedPipelineApiTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(*test_data["general"])
  def testGetPipelineStageAssignmentDefault(self, model_def):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = model_def.build()
      check_default_assignments(self, m)

  @parameterized.named_parameters(*test_data["general"])
  def testSetPipelineStageAssignment(self, model_def):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = model_def.build()
      set_assignments(m)
      check_assignments(self, m)

  @parameterized.named_parameters(*test_data["scoped"])
  def testPipelineStageAssignmentWithScopes(self, model_def):
    # Sequential does not support pipeline stage assignment with scopes.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategy()
    with strategy.scope():
      m = model_def.build()
      check_assignments(self, m)


if __name__ == '__main__':
  tf.test.main()
