# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Functional Pipelining API interface."""
import tempfile
import os
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import gradient_accumulation as ga

from keras.ipu import extensions
from keras.engine import training as training_module
from keras import layers
from keras import models
from keras import testing_utils


def get_simple_model():
  d1 = layers.Input(32)
  d2 = layers.Input(32)
  f1 = layers.Flatten()(d1)
  f2 = layers.Flatten()(d2)
  x1 = layers.Dense(4)(f1)
  x2 = layers.Dense(4)(f2)
  l = layers.Dense(8)
  o1 = l(x1)
  o2 = l(x2)
  return training_module.Model((d1, d2), (o1, o2))


def get_model_with_assignments():
  d1 = layers.Input(32)
  d2 = layers.Input(32)

  with extensions.functional_extensions.PipelineStage(0):
    f1 = layers.Flatten()(d1)
  with extensions.functional_extensions.PipelineStage(1):
    f2 = layers.Flatten()(d2)
  with extensions.functional_extensions.PipelineStage(2):
    x1 = layers.Dense(4)(f1)
  with extensions.functional_extensions.PipelineStage(3):
    x2 = layers.Dense(4)(f2)

  with extensions.functional_extensions.PipelineStage(4):
    # Apply stage to layer, this can be overridden by stages assigned to
    # specific nodes.
    l = layers.Dense(8)

  # Already has stage 2 assigned to the layer.
  o1 = l(x1)

  with extensions.functional_extensions.PipelineStage(5):
    # Overrides layer assignment (stage 4).
    o2 = l(x2)

  return training_module.Model((d1, d2), (o1, o2))


def get_model_with_partial_assignments():
  d1 = layers.Input(32)
  d2 = layers.Input(32)

  with extensions.functional_extensions.PipelineStage(0):
    f1 = layers.Flatten()(d1)
  f2 = layers.Flatten()(d2)
  x1 = layers.Dense(4)(f1)
  x2 = layers.Dense(4)(f2)
  l = layers.Dense(8)
  o1 = l(x1)
  o2 = l(x2)
  return training_module.Model((d1, d2), (o1, o2))


def check_assignments(instance, assignments):
  instance.assertTrue(
      all(
          isinstance(
              assignment, extensions.functional_extensions.
              FunctionalLayerPipelineStageAssignment)
          for assignment in assignments))


class FunctionalPipelineApiTest(tf.test.TestCase):
  @testing_utils.run_v2_only
  def testGetPipelineStageAssignmentDefault(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test default assignment.
      assignments = m.get_pipeline_stage_assignment()

      # 3 layers, but each layer is called twice.
      self.assertEqual(len(assignments), 6)
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Set assignments.
      nodes_to_stage = {}
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
        nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))] = i  # pylint: disable=protected-access
      m.set_pipeline_stage_assignment(assignments)

      # Get assignments, and verify they are the same as the ones we set.
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      for assignment in assignments:
        self.assertEqual(assignment.pipeline_stage, nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))])  # pylint: disable=protected-access

  @testing_utils.run_v2_only
  def testResetPipelineStageAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Set assignments.
      nodes_to_stage = {}
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i
        nodes_to_stage[str(
            id(assignment.layer._inbound_nodes[assignment.node_index]))] = i  # pylint: disable=protected-access
      m.set_pipeline_stage_assignment(assignments)
      self.assertTrue(m._is_pipelined())  # pylint: disable=protected-access

      # Reset assignments, and check all nodes have no assignment.
      m.reset_pipeline_stage_assignment()
      assignments = m.get_pipeline_stage_assignment()
      check_assignments(self, assignments)
      self.assertTrue(
          all(assignment.pipeline_stage is None for assignment in assignments))

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignmentWithInvalidNumberOfAssignments(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test number of assignments.
      assignments = m.get_pipeline_stage_assignment()
      assignments.pop()
      with self.assertRaisesRegex(
          ValueError,
          r"The size of the provided `pipeline_stage_assignment` \(5\) does "
          r"not match the total number of invocations of layers in the model "
          r"\(currently 6\)."):
        m.set_pipeline_stage_assignment(assignments)

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignmentWithInvalidAssignmentClass(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test type of elements.
      assignments = m.get_pipeline_stage_assignment()
      assignments[0] = None
      with self.assertRaisesRegex(
          ValueError,
          r"All elements of `pipeline_stage_assignment` need to be instances "
          r"of `FunctionalLayerPipelineStageAssignment`."):
        m.set_pipeline_stage_assignment(assignments)

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignmentWithMissingAssignment(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test all nodes assigned a pipeline stage.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments[:-1]):
        assignment.pipeline_stage = i
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* with node index 0 has not been assigned a pipeline "
          r"stage."):
        m.set_pipeline_stage_assignment(assignments)

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignmentWithEmptyStages(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test not all stages have been assigned to.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i * 2
      with self.assertRaisesRegex(
          ValueError,
          r"Pipeline stages in the graph need to be strictly increasing, found "
          r"pipeline stages 0, 2, 4, 6, 8, 10, however the following pipeline "
          r"stages are missing 1, 3, 5, 7, 9."):
        m.set_pipeline_stage_assignment(assignments)

  @testing_utils.run_v2_only
  def testSetPipelineStageAssignmentWithDependencyOnLaterStage(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      # Test can make pipeline stage post order.
      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i - 1
      assignments[0].pipeline_stage = assignments[-1].pipeline_stage
      with self.assertRaisesRegex(
          ValueError,
          r"Layer dense.* in pipeline stage 1 has a dependency from a pipeline "
          r"stage"):
        m.set_pipeline_stage_assignment(assignments)

  @testing_utils.run_v2_only
  def testPipelineStageAssignmentWithScopes(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_model_with_assignments()
      # Should not need to build the model. That should be done automatically,
      # since we are passing in data.
      inputs = [
          np.ones(shape=(6, 32), dtype=np.int32),
          np.ones(shape=(6, 32), dtype=np.int32)
      ]
      m.compile(steps_per_execution=6)
      m.set_pipelining_options(device_mapping=[0] * 6)
      m.predict(inputs, batch_size=1)

      self.assertTrue(m._is_pipelined())  # pylint: disable=protected-access

      # Check assignments from scopes are applied.
      for assignment in m.get_pipeline_stage_assignment():
        if assignment.layer == m.layers[2]:
          self.assertEqual(assignment.pipeline_stage, 0)
        if assignment.layer == m.layers[3]:
          self.assertEqual(assignment.pipeline_stage, 1)
        if assignment.layer == m.layers[4]:
          self.assertEqual(assignment.pipeline_stage, 2)
        if assignment.layer == m.layers[5]:
          self.assertEqual(assignment.pipeline_stage, 3)
        if assignment.layer == m.layers[6]:
          if assignment.node_index == 0:
            self.assertEqual(assignment.pipeline_stage, 4)
          else:
            self.assertEqual(assignment.pipeline_stage, 5)

  @testing_utils.run_v2_only
  def testRunModelWithPartialPipelineStageAssignments(self):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      with self.assertRaisesRegex(
          ValueError,
          r"All layers of a pipelined model must have an associated pipeline "
          r"stage. However, .* has not been assigned to one."):
        get_model_with_partial_assignments()

  @testing_utils.run_v2_only
  def testSaveRestore(self):
    cfg = config.IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      def check_assignments_are(model, assignments):
        for assignment in assignments:
          if assignment.layer is model.layers[2]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 0)
          if assignment.layer is model.layers[3]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 1)
          if assignment.layer is model.layers[4]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 2)
          if assignment.layer is model.layers[5]:
            self.assertEqual(assignment.node_index, 0)
            self.assertEqual(assignment.pipeline_stage, 3)
          if assignment.layer is model.layers[6]:
            if assignment.node_index == 0:
              self.assertEqual(assignment.pipeline_stage, 4)
            else:
              self.assertEqual(assignment.node_index, 1)
              self.assertEqual(assignment.pipeline_stage, 5)

      # Test default assignment.
      assignments = m.get_pipeline_stage_assignment()
      for assignment in assignments:
        if assignment.layer is m.layers[2]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 0
        if assignment.layer is m.layers[3]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 1
        if assignment.layer is m.layers[4]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 2
        if assignment.layer is m.layers[5]:
          self.assertEqual(assignment.node_index, 0)
          assignment.pipeline_stage = 3
        if assignment.layer is m.layers[6]:
          if assignment.node_index == 0:
            assignment.pipeline_stage = 4
          else:
            self.assertEqual(assignment.node_index, 1)
            assignment.pipeline_stage = 5

      m.set_pipeline_stage_assignment(assignments)
      check_assignments_are(m, assignments)

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(save_path)
        self.assertTrue(m._is_pipelined)  # pylint: disable=protected-access
        assignments = m.get_pipeline_stage_assignment()
        check_assignments_are(m, assignments)

  @testing_utils.run_v2_only
  def testSetPipeliningOptionsWithNonIntegerTypeDeviceMapping(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      with self.assertRaisesRegex(
          ValueError, "Expected `device_mapping` to be a list of integers"):
        m.set_pipelining_options(device_mapping=[0.0] * 10)

  @testing_utils.run_v2_only
  def testSetPipeliningOptionsWithInvalidKeys(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()

      with self.assertRaisesRegex(
          ValueError,
          "Found `gradient_accumulation_count` key in `pipelining_kwargs`. Set "
          "the `gradient_accumulation_steps_per_replica` argument to "
          "`set_pipelining_options` instead."):
        m.set_pipelining_options(gradient_accumulation_count=10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `repeat_count` key in `pipelining_kwargs`. This argument is "
          "automatically set by Keras"):
        m.set_pipelining_options(repeat_count=10)

      with self.assertRaisesRegex(
          ValueError,
          "Found `batch_serialization_iterations` key in `pipelining_kwargs`. "
          "This argument is not compatible with Keras"):
        m.set_pipelining_options(batch_serialization_iterations=10)

  @testing_utils.run_v2_only
  def testSaveAndRestorePipeliningOptions(self):
    cfg = IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = get_simple_model()
      m.build([(1, 1), (1, 1)])

      assignments = m.get_pipeline_stage_assignment()
      for i, assignment in enumerate(assignments):
        assignment.pipeline_stage = i

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=10,
                               device_mapping=[4, 3, 2, 1, 0],
                               accumulate_outfeed=True,
                               gradient_accumulation_reduction_method='mean')

      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = models.load_model(save_path)
        self.assertEqual(
            m._pipelining_gradient_accumulation_steps_per_replica,  # pylint: disable=protected-access
            10)
        self.assertEqual(m._pipelining_device_mapping, [4, 3, 2, 1, 0])  # pylint: disable=protected-access
        self.assertEqual(m._pipelining_accumulate_outfeed, True)  # pylint: disable=protected-access
        self.assertEqual(
            m._gradient_accumulation_reduction_method,  # pylint: disable=protected-access
            ga.GradientAccumulationReductionMethod.MEAN)

  def testPrintPipelineStageSummary(self):
    cfg = config.IPUConfig()

    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      d1 = layers.Input(32, name="input_a")
      d2 = layers.Input(32, name="input_b")
      f1 = layers.Flatten()(d1)
      f2 = layers.Flatten()(d2)
      c1 = layers.Concatenate()([f1, f2])
      x1 = layers.Dense(4)(c1)
      y1 = tf.multiply(1.0, x1)
      m = training_module.Model((d1, d2), y1)

      strings = []

      def print_fn(x):
        strings.append(x)

      m.print_pipeline_stage_assignment_summary(line_length=85,
                                                print_fn=print_fn)

      # pylint: disable=line-too-long
      self.assertEqual(strings[0], 'Model: "model"')
      self.assertEqual(strings[1], '_' * 85)
      self.assertEqual(
          strings[2],
          'Layer (type) (node index)         Input Layers                      Pipeline Stage   '
      )
      self.assertEqual(strings[3], '=' * 85)
      self.assertEqual(
          strings[4],
          'flatten (Flatten) (0)             input_a                           None             '
      )
      self.assertEqual(strings[5], '_' * 85)
      self.assertEqual(
          strings[6],
          'flatten_1 (Flatten) (0)           input_b                           None             '
      )
      self.assertEqual(strings[7], '_' * 85)
      self.assertEqual(
          strings[8],
          'concatenate (Concatenate) (0)     flatten                           None             '
      )
      self.assertEqual(
          strings[9],
          '                                  flatten_1                                          '
      )
      self.assertEqual(strings[10], '_' * 85)
      self.assertEqual(
          strings[11],
          'dense (Dense) (0)                 concatenate                       None             '
      )
      self.assertEqual(strings[12], '_' * 85)
      self.assertEqual(
          strings[13],
          'tf.math.multiply (TFOpLambda) (0) dense                             None             '
      )
      self.assertEqual(strings[14], '=' * 85)
      # pylint: enable=line-too-long


if __name__ == '__main__':
  tf.test.main()
