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
# =============================================================================
"""
Pipeline stage assignment types for IPU specific Keras Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.util.tf_export import keras_export

from keras.utils import generic_utils

# pylint: disable=g-inconsistent-quotes
ipu_strategy = generic_utils.LazyLoader("ipu_strategy", globals(),
                                        "tensorflow.python.ipu.ipu_strategy")
# pylint: enable=g-inconsistent-quotes


@keras_export('keras.ipu.PipelineStage')
class PipelineStage:
  """A scope within which Keras layers and/or calls to Keras layers can be
  assigned to pipeline stages.

  Pipeline stages can be assigned to all calls of `Layer` by constructing the
  `Layer` within a :class:`~keras.ipu.PipelineStage` scope as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategy()
    input_layer = Input(2)
    with strategy.scope():
      with PipelineStage(0):
        x = Dense(4)(input_layer)

      with PipelineStage(1):
        x = Dense(4)(x)

  Pipeline stages can also be assigned to individual `Layer` calls, as follows:

  .. code-block:: python

    strategy = ipu.ipu_strategy.IPUStrategy()
    input_layer = Input(2)
    l = Dense(4)
    with strategy.scope():
      with PipelineStage(0):
        x = l(input_layer)

      with PipelineStage(1):
        x = l(x)

  Pipeline stages assigned to `Layer` calls take precedence over those assigned
  when constructing the `Layer`.
  """
  def __init__(self, stage):
    """Creates a scope within which Keras layers and/or calls to Keras layers
    are assigned to pipeline stages.

    Arguments:
      stage: The pipeline stage that any Keras layers created and/or called
        will be assigned to within this scope.
    """
    self._stage = stage

  def __enter__(self):
    if self._stage < 0:
      raise ValueError("%d is not a valid pipeline stage.")

    strategy = ds_context.get_strategy()
    if not isinstance(strategy, ipu_strategy.IPUStrategyV1):
      raise RuntimeError("PipelineStage may only be used from "
                         "within an IPUStrategy context.")

    if hasattr(strategy, "_pipeline_stage"):
      raise RuntimeError("Pipeline stages must not be nested.")

    strategy._pipeline_stage = self._stage  # pylint: disable=protected-access

    return self

  def __exit__(self, _exception_type, _value, _traceback):
    strategy = ds_context.get_strategy()
    assert strategy and hasattr(strategy, "_pipeline_stage")
    delattr(strategy, "_pipeline_stage")


@keras_export('keras.ipu.ModelLayerPipelineStageAssignment')
class ModelLayerPipelineStageAssignment:
  """A class to indicate at which pipeline stage a layer in a `Model` subclass
  should be executed.

  Keras layers can be called multiple times in order to share weights between
  layers. A new :class:`~keras.ipu.ModelLayerPipelineStageAssignment` is
  required for every call. For example, `node_index=0` will correspond to the
  first time the layer was called. As weights are shared, all stages a given
  layer is assigned to must be mapped to the same device.
  """
  def __init__(self, layer, node_index, pipeline_stage=None):
    """Create a new :class:`~keras.ipu.ModelLayerPipelineStageAssignment`.

    Args:
      layer (keras.layers.Layer): The Keras layer which this assignment is for.
      node_index (int): The specific call to the layer.
      pipeline_stage (int): If provided, indicates which pipeline stage this
        layer should be assigned to. If not provided this layer will be
        unassigned.
    """
    self._layer = layer
    self._node_index = node_index
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """The Keras layer associated with this assignment."""
    return self._layer

  @property
  def node_index(self):
    """The specific call to the layer."""
    return self._node_index

  @property
  def inbound_layers(self):
    """The input layers for the layer in this assignment. This can be
    useful for identifying which specific `node_index` this is."""
    node = self._layer._inbound_nodes[self.node_index]  # pylint: disable=protected-access
    return [n.layer for n in node.parent_nodes]

  @property
  def pipeline_stage(self):
    """The pipeline stage this layer has been assigned to. If `None`,
    then this layer has not been assigned to a pipeline stage."""
    return self._pipeline_stage

  @pipeline_stage.setter
  def pipeline_stage(self, value):
    """Setter of
    :py:meth:`~keras.ipu.ModelLayerPipelineStageAssignment.pipeline_stage`

    Args:
      value (int): The pipeline stage to assign this layer to."""
    self._pipeline_stage = value

  def __str__(self):
    return ("Layer: {} (node index {}) is assigned to pipeline "
            "stage: {}".format(self.layer.name, self.node_index,
                               self.pipeline_stage))


@keras_export('keras.ipu.FunctionalLayerPipelineStageAssignment')
class FunctionalLayerPipelineStageAssignment(ModelLayerPipelineStageAssignment
                                             ):
  """A class to indicate at which pipeline stage a layer in a `Functional` model
  should be executed.


  Keras layers can be called multiple times in order to share weights between
  layers. A new :class:`~keras.ipu.FunctionalLayerPipelineStageAssignment` is
  required for every call. For example, `node_index=0` will correspond to the
  first time the layer was called. As weights are shared, all stages a given
  layer is assigned to must be mapped to the same device.
  """
  def __init__(self, layer, node_index, pipeline_stage=None):  # pylint: disable=useless-super-delegation
    """Create a new :class:`~keras.ipu.FunctionalLayerPipelineStageAssignment`.

    Args:
      layer (keras.layers.Layer): The Keras which this assignment is for.
      node_index (int): The specific call to the layer.
      pipeline_stage (int): If provided, indicates which pipeline stage this
        layer should be assigned to. If not provided this layer will be
        unassigned.
    """
    super().__init__(layer, node_index, pipeline_stage)


@keras_export('keras.ipu.SequentialLayerPipelineStageAssignment')
class SequentialLayerPipelineStageAssignment:
  """A class to indicate at which pipeline stage a layer in a `Sequential` model
  should be executed.
  """
  def __init__(self, layer, pipeline_stage=None):
    """Create a new :class:`~keras.ipu.SequentialLayerPipelineStageAssignment`.

    Args:
      layer (keras.layers.Layer): The Keras layer which this assignment is for.
      pipeline_stage (int): If provided, indicates which pipeline stage this
        layer should be assigned to. If not provided this layer will be
        unassigned.
    """
    self._layer = layer
    self.pipeline_stage = pipeline_stage

  @property
  def layer(self):
    """The Keras layer associated with this assignment."""
    return self._layer

  @property
  def pipeline_stage(self):
    """The pipeline stage this layer has been assigned to. If `None`,
    then this layer has not been assigned to a pipeline stage."""
    return self._pipeline_stage

  @pipeline_stage.setter
  def pipeline_stage(self, value):
    """Setter of
    :py:meth:`~keras.ipu.SequentialLayerPipelineStageAssignment.pipeline_stage`

    Args:
      value (int): The pipeline stage to assign this layer to."""
    self._pipeline_stage = value

  def __str__(self):
    return ("Layer: {} is assigned to pipeline stage: {}".format(
        self.layer.name, self.pipeline_stage))


@keras_export('keras.ipu.NestedModelPipelineStageAssignment')
class NestedModelPipelineStageAssignment:
  """A class containing the pipeline stage assignments for a nested model in a
  `Model` subclass. These are separate from assignments set directly on the
  nested model, though any such existing assignments are used as defaults.

  Nested models can be called multiple times. A new
  :class:`~keras.ipu.NestedModelPipelineStageAssignment` is required for every
  call. For example, `node_index=0` will correspond to the first time the nested
  model was called. All stages a given layer is assigned to must be mapped to
  the same device.
  """
  def __init__(self, nested_model, node_index, pipeline_stage_assignments):
    """Create a new :class:`~keras.ipu.NestedModelPipelineStageAssignment`.

    Args:
      nested_model (keras.Model): The nested Keras model which this assignment
        is for.
      node_index (int): The specific call to the nested model.
      pipeline_stage_assignments (list): The pipeline stage assignments for this
        call to a nested model.
    """
    self._nested_model = nested_model
    self._node_index = node_index
    self._pipeline_stage_assignments = pipeline_stage_assignments

  @property
  def nested_model(self):
    """The nested model associated with this assignment."""
    return self._nested_model

  @property
  def node_index(self):
    """The index of the specific call to the nested model."""
    return self._node_index

  @property
  def inbound_layers(self):
    """The input layers for the nested model in this assignment. This can be
    useful for identifying which specific `node_index` this is."""
    node = self._layer._inbound_nodes[self.node_index]  # pylint: disable=protected-access
    return [n.layer for n in node.parent_nodes]

  @property
  def pipeline_stage_assignments(self):
    """The pipeline stage assignments for this nested model."""
    return list(self._pipeline_stage_assignments)

  def __str__(self):
    assignments_string = "".join(f"\n  {x}"
                                 for x in self.pipeline_stage_assignments)
    return (f"Model: {self.nested_model.name} (node index {self.node_index}) "
            f"has the following pipeline stage assignments:"
            f"{assignments_string}")


@keras_export('keras.ipu.FunctionalNestedModelPipelineStageAssignment')
class FunctionalNestedModelPipelineStageAssignment(
    NestedModelPipelineStageAssignment):
  """A class containing the pipeline stage assignments for a nested model in a
  `Functional` model. These are separate from assignments set directly on the
  nested model, though any such existing assignments are used as defaults.

  Nested models can be called multiple times. A new
  :class:`~keras.ipu.FunctionalNestedModelPipelineStageAssignment` is required
  for every call. For example, `node_index=0` will correspond to the first time
  the nested model was called. All stages a given layer is assigned to must be
  mapped to the same device.
  """
  def __init__(self, nested_model, node_index, pipeline_stage_assignments):  # pylint: disable=useless-super-delegation
    """Create a new :class:`~keras.ipu.FunctionalNestedModelPipelineStageAssignment`.

    Args:
      nested_model (keras.Model): The nested Keras model which this assignment
        is for.
      node_index (int): The specific call to the nested model.
      pipeline_stage_assignments (list): The pipeline stage assignments for this
        call to a nested model.
    """
    super().__init__(nested_model, node_index, pipeline_stage_assignments)


@keras_export('keras.ipu.SequentialNestedModelPipelineStageAssignment')
class SequentialNestedModelPipelineStageAssignment:
  """A class containing the pipeline stage assignments for a nested model in a
  `Sequential` model. These are separate from assignments set directly on the
  nested model, though any such existing assignments are used as defaults.

  Nested models can be called multiple times. A new
  :class:`~keras.ipu.SequentialNestedModelPipelineStageAssignment` is required
  for every call. For example, `node_index=0` will correspond to the first time
  the nested model was called. All stages a given layer is assigned to must be
  mapped to the same device.
  """
  def __init__(self, nested_model, pipeline_stage_assignments):
    """Create a new :class:`~keras.ipu.SequentialNestedModelPipelineStageAssignment`.

    Args:
      nested_model (keras.Model): The nested Keras model which this assignment
        is for.
      pipeline_stage_assignments (list): The pipeline stage assignments for this
        call to a nested model.
    """
    self._nested_model = nested_model
    self._pipeline_stage_assignments = pipeline_stage_assignments

  @property
  def nested_model(self):
    """The nested Keras model associated with this assignment."""
    return self._nested_model

  @property
  def pipeline_stage_assignments(self):
    """The pipeline stage assignments for this nested model."""
    return list(self._pipeline_stage_assignments)

  def __str__(self):
    assignments_string = "".join(f"\n  {x}"
                                 for x in self.pipeline_stage_assignments)
    return (
        f"Model: {self.nested_model.name} has the following pipeline stage "
        f"assignments:{assignments_string}")
