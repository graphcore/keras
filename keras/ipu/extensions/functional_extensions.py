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
# =============================================================================
"""
IPU specific Keras Functional Model extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import copy

import tensorflow.compat.v2 as tf

from keras.ipu.extensions import extensions_base
from keras.ipu.extensions.pipeline_stage_assignment import FunctionalLayerPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import FunctionalNestedModelPipelineStageAssignment
from keras.engine import functional
from keras.engine import node as node_module


class FunctionalExtension(extensions_base.KerasExtensionBase):  # pylint: disable=abstract-method
  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
    extensions_base.KerasExtensionBase.__init__(self)
    self._pipeline_stage_assignment = []

  def _is_pipelined(self):
    return bool(self._pipeline_stage_assignment)

  def _get_config_supported(self):
    return True

  def _get_config_delegate(self):
    # Get the Keras config.
    config = self.get_config(__extension_delegate=False)
    # Get the KerasExtensionBase config and merge it in.
    extension_config = self._get_base_config()
    config.update(extension_config)
    # Add pipelining options.
    # Get index for each layer.
    layer_to_index = {}
    for i, layer in enumerate(self.layers):
      layer_to_index[str(id(layer))] = i
    config["pipeline_stage_assignment"] = [
        (layer_to_index[str(id(assignment.layer))], assignment.node_index,
         assignment.pipeline_stage)
        for assignment in self._pipeline_stage_assignment
    ]
    return config

  def _deserialize_from_config_supported(self, config):
    del config
    return True

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _deserialize_from_config_delegate(self, config):
    self._from_base_config(config)
    # Extract pipelining options.
    self._pipeline_stage_assignment = [
        FunctionalLayerPipelineStageAssignment(self.layers[layer_idx],
                                               node_index, stage)
        for layer_idx, node_index, stage in config.get(
            "pipeline_stage_assignment", [])
    ]

  def set_asynchronous_callbacks(self, asynchronous=False):
    # pylint:disable=line-too-long
    """Sets the asynchronous callback options when calling `fit()`, `evaluate()`
    and `predict()`.

    When running `fit()`, `evaluate()` and `predict()` the callbacks the model
    is configured with are executed after `steps_per_execution` steps have
    executed. Enabling asynchronous callbacks means that the callbacks are
    invoked after every step, even when `steps_per_execution > 1`. This can
    reduce the latency of receiving per step results and metrics at a cost of
    an extra thread running in the background of the application.
    Note that this option is ignored for `fit()` and `evaluate()` when
    running a pipelined model and `accumulate_outfeed=True` (configured via
    :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipelining_options`).

    Args:
      asynchronous: If `True`, enables asynchronous callbacks. Defalts to
        `False`.
    """
    # pylint:enable=line-too-long
    self._set_asynchronous_callbacks_impl(asynchronous)

  def set_replication_options(self, replicated_metric_reduction_method='NONE'):
    """Configure behaviour when using this model with replication.

    Args:
      replicated_metric_reduction_method: Cross-replica reduction method to use
        when returning metrics which exist across multiple replicas.
        Defaults to `ReplicatedMetricReductionMethod.NONE`
        (see :class:`~keras.ipu.ReplicatedMetricReductionMethod`).
    """
    self._set_replication_options_impl(replicated_metric_reduction_method)

  def set_gradient_accumulation_options(
      self,
      gradient_accumulation_steps_per_replica=None,
      gradient_accumulation_reduction_method='sum',
      use_v2_gradient_accumulation_optimizer=False,
      **gradient_accumulation_optimizer_kwargs):
    # pylint:disable=line-too-long
    """Sets the gradient accumulation options for non-pipelined models which are
    to be used when training a model.

    When set, and `gradient_accumulation_steps_per_replica > 1`, the optimizer
    which the current model has been compiled with is wrapped in
    :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
    This means that each replica will accumulate the gradients for
    `gradient_accumulation_steps_per_replica` steps, these accumulated gradients
    are then all-reduced across the replicas and the weight update is performed.

    Gradient accumulation allows us to simulate bigger batch sizes. For example
    if we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there is single replica in
    the system, this simulates an input batch of size 64.
    If we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there are 4 replicas in
    the system, this simulates an input batch of size 256.

    See the :ref:`gradient-accumulation` section for more details.

    The value of `gradient_accumulation_steps_per_replica` has no effect when
    using `evaluate()` or `predict()`.

    Note that the `minimize` API of the provided optimizer will not be called
    when gradient accumulation is enabled. As such, overriding `minimize` in
    a custom optimizer will cause a `ValueError` to be raised.

    Args:
      gradient_accumulation_steps_per_replica: An integer which indicates the
        number of steps the gradients will be accumulated for in each replica.
        The `steps_per_execution` value used when compiling the model must be
        divisible by `gradient_accumulation_steps_per_replica`. This value is
        saved/loaded when the model is saved/loaded.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients
        can either be directly summed up or scaled such that we compute a mean
        of all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using
        float16, but gives smaller gradients and might require adjusting
        the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`).
      use_v2_gradient_accumulation_optimizer: When enabled, the `OptimizerV2`
        based IPU Keras `GradientAccumulationOptimizer`
        (see :class:`~keras.ipu.optimizers.gradient_accumulation_optimizer.GradientAccumulationOptimizer`)
        is used in place of the default IPU TensorFlow `GradientAccumulationOptimizerV2`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationOptimizerV2`).
        Default is False.
      gradient_accumulation_optimizer_kwargs: All remaining keyword arguments
        are forwarded to
        :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationOptimizerV2`.
        See the optimizer for all the available arguments. Must not contain
        `opt` or `num_mini_batches` as keys. Note that this dictionary is not
        serializable, which means that when the model is being saved, these
        values are not saved. When restoring/loading a model, please call
        :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_gradient_accumulation_options`
        again.
    """
    # pylint:enable=line-too-long
    self._set_gradient_accumulation_options_impl(
        gradient_accumulation_steps_per_replica,
        gradient_accumulation_reduction_method,
        use_v2_gradient_accumulation_optimizer,
        gradient_accumulation_optimizer_kwargs)

  def set_pipelining_options(self,
                             gradient_accumulation_steps_per_replica=None,
                             device_mapping=None,
                             accumulate_outfeed=None,
                             gradient_accumulation_reduction_method='sum',
                             **pipelining_kwargs):
    # pylint:disable=line-too-long
    """Sets the pipelining options, including gradient accumulation options,
    for pipelined models.

    Before training a pipelined model, the `gradient_accumulation_steps_per_replica`
    argument needs to be set as pipelined models always perform gradient
    accumulation when training. Setting
    `gradient_accumulation_steps_per_replica > 1` means that each replica will
    accumulate the gradients for `gradient_accumulation_steps_per_replica`
    steps. These accumulated gradients are then all-reduced across the replicas
    and the weight update is performed.

    Gradient accumulation allows us to simulate bigger batch sizes. For example
    if we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there is single replica in
    the system, this simulates an input batch of size 64.
    If we have a model where each step is of batch size 16 and we set
    `gradient_accumulation_steps_per_replica=4` and there are 4 replicas in
    the system, this simulates an input batch of size 256.

    When training a data-parallel model, enabling gradient accumulation also
    reduces the communication overhead as the all-reduce of gradients is now
    performed after each replica has performed
    `gradient_accumulation_steps_per_replica` steps instead of after each step.

    See the :ref:`gradient-accumulation` section for more details.

    The value of `gradient_accumulation_steps_per_replica` has no effect when
    using `evaluate()` or `predict()`.

    Note that the `minimize` API of the provided optimizer will not be called
    when pipelining is enabled. As such, overriding `minimize` in a custom
    optimizer will cause a `ValueError` to be raised.

    Args:
      gradient_accumulation_steps_per_replica: An integer which indicates the
        number of steps the gradients will be accumulated for in each replica.
        The `steps_per_execution` value used when compiling the model must be
        divisible by `gradient_accumulation_steps_per_replica`. This value is
        saved/loaded when the model is saved/loaded.
      device_mapping: If provided, a list of length equal to the number of
        pipeline stages assigned in this model. An element at index `i` in the
        list represents which IPU the `i`'th pipeline stage should reside on.
        This can be used to make sure computational stages which share Keras
        layers/`tf.Variable` objects are resident on the same IPU. This value is
        saved/loaded when the model is saved/loaded.
      accumulate_outfeed: The metrics from the model are normally enqueued as
        soon as they're available. If this option is True, the data will
        instead be accumulated when they're available and enqueued at the end of
        pipeline execution, reducing the amount of host <-> device
        communication. When used with training, the accumulated metrics are
        normalised by `gradient_accumulation_steps_per_replica`. When used with
        evaluation, the accumulated metrics are normalised by `steps_per_epoch`.
        This option is ignored when doing prediction. When using
        `accumulate_outfeed`, model callbacks will be called with the same data
        for the batches which the data was accumulated for. This value is
        saved/loaded when the model is saved/loaded.
      gradient_accumulation_reduction_method:  (Experimental)  Reduction method
        to use when accumulating gradients. During the iterations in each
        optimizer step, the computed gradients can either be directly summed up
        or scaled such that we compute a mean of all gradients for each
        variable. Computing a mean avoids potential issues with overflow during
        accumulation especially when using float16, but gives smaller gradients
        and might require adjusting the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`).
      pipelining_kwargs: All remaining keyword arguments are forwarded to
        :func:`~tensorflow.python.ipu.pipelining_ops.pipeline`. Note that this
        dictionary is not serializable, which means that when the model is
        being saved, these values are not saved. When restoring/loading a model,
        please call
        :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipelining_options`
        again.
    """
    # pylint:enable=line-too-long
    self._set_pipelining_options_impl(gradient_accumulation_steps_per_replica,
                                      device_mapping, accumulate_outfeed,
                                      gradient_accumulation_reduction_method,
                                      pipelining_kwargs)

  def set_infeed_queue_options(self, **kwargs):
    """Sets the options for all instances of `IPUInfeedQueue` generated
    when executing the model.

    When using `fit()`, `evalute()` and `predict()`, an instance of
    :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` is created
    to efficiently feed data from the dataset to the device. Instances of
    `IPUInfeedQueue` can be created with optional arguments, such as
    `prefetch_depth`, which can increase the throughput of the model.

    Args:
      **kwargs: All keyword arguments are forwarded to
        :class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue`.
    """
    self._set_infeed_queue_options_impl(**kwargs)

  def set_outfeed_queue_options(self, **kwargs):
    """Sets the options for all instances of
    :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`
    generated when executing the model.

    When using `fit()`, `evalute()` and `predict()`, an instance of
    :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue` is created
    to efficiently feed data from the device to the host. Instances of
    :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`
    can be created with optional arguments, such as
    `buffer_depth`, which can increase the throughput of the model.

    Args:
      **kwargs: All keyword arguments are forwarded to
        :class:`~tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue`.
    """
    self._set_outfeed_queue_options_impl(**kwargs)

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _get_pipelined_post_order(self, pipeline_stage_assignment, inputs=None):
    # Create a post order per pipeline stage as post order does not take
    # pipeline stages into account, for example multiple pipeline stages might
    # have output layers. Try reordering the nodes to preserve post order
    # and to make sure pipeline stages can still be executed in order.
    # `inputs` can be specified to create a post order starting with the given
    # inputs. Otherwise, the inputs specified during tracing are used.

    post_order_per_stage = {}
    post_order = self._create_post_order()

    if not isinstance(pipeline_stage_assignment, list):
      raise ValueError("`pipeline_stage_assignment` needs to be a list.")

    if len(pipeline_stage_assignment) != len(post_order):
      raise ValueError(
          f"The length of the provided `pipeline_stage_assignment` "
          f"({len(pipeline_stage_assignment)}) does not match the number of "
          f"layers in the graph ({len(post_order)}). "
          f"Each layer needs to be assigned a pipeline stage "
          f"(excluding input layers).")

    tensor_dict = {}

    def visited(tensors, new_tensors=None):
      tensors = tf.nest.flatten(tensors)
      ids = [str(id(t)) for t in tensors]
      if new_tensors is not None:
        tensors = tf.nest.flatten(new_tensors)

      for t_id, tensor in zip(ids, tensors):
        tensor_dict[t_id] = [tensor] * self._tensor_usage_count[t_id]

    visited(self.inputs, inputs)

    if inputs is None:
      # Assign inputs to stage 0. If using explicit inputs, skip this as the
      # inputs come from outside this model.
      for tensor in self.inputs:
        layer = tensor._keras_history.layer  # pylint: disable=protected-access
        assert len(layer.inbound_nodes) == 1
        post_order_per_stage.setdefault(0, []).append(layer.inbound_nodes[0])

    stages = set()
    for assignment, node in zip(pipeline_stage_assignment, post_order):
      if isinstance(node.layer, extensions_base.KerasExtensionBase):
        # If node is a nested model, calculate its post-order-per-stage and
        # append it to the post-order-per-stage we are building for this model.

        if not isinstance(assignment,
                          FunctionalNestedModelPipelineStageAssignment):
          raise ValueError(
              f"The pipeline stage assignment for nested model "
              f"{node.layer.name} in {self.name} must be an instance of "
              f"`FunctionalNestedModelPipelineStageAssignment`. Instead the "
              f"assignment was of type {type(assignment).__name__}.")

        # Check that the assignment is for this node.
        if id(assignment.nested_model) != id(node.layer):
          raise ValueError(
              f"The order of `pipeline_stage_assignment` does not match the "
              f"post-order generated from the graph ({assignment.layer.name} "
              f"!= {node.layer.name}).")

        # Get a post order for the nested model using the tensors from this
        # post order as inputs.
        new_args, _ = node.map_arguments(tensor_dict)

        nested_post_order_per_stage, output_tensors = (
            node.layer._get_pipelined_post_order(  # pylint: disable=protected-access
                assignment.pipeline_stage_assignments,
                inputs=new_args))

        # Map the outputs of the nested model node to the output tensors from
        # its post order.
        visited(node.outputs, output_tensors)

        # Update the current post order with nodes from the nested model.
        stages.update(nested_post_order_per_stage)
        for stage, nested_post_order in nested_post_order_per_stage.items():
          for nested_node in nested_post_order:
            post_order_per_stage.setdefault(stage, []).append(nested_node)

        continue

      if not isinstance(assignment, FunctionalLayerPipelineStageAssignment):
        raise ValueError(
            f"The pipeline stage assignment for layer {node.layer.name} in "
            f"{self.name} must be an instance of "
            f"`FunctionalLayerPipelineStageAssignment`. Instead the assignment "
            f"was of type {type(assignment).__name__}.")

      # Check that the assignment is for this node.
      if id(assignment.layer) != id(node.layer):
        raise ValueError(
            f"The order of `pipeline_stage_assignment` does not match the "
            f"post-order generated from the graph ({assignment.layer.name} "
            f"!= {node.layer.name}).")

      if assignment.pipeline_stage is None:
        raise ValueError(
            f"Layer {assignment.layer.name} with node_index "
            f"{assignment.node_index} has not been assigned a pipeline stage "
            f"in `pipeline_stage_assignment`.")

      # Create a new node using the tensors from the current post order as
      # inputs (the current order tracks tensors through nested models).
      new_args, new_kwargs = node.map_arguments(tensor_dict)
      new_outputs = node.layer(*new_args, **new_kwargs)
      visited(node.outputs, new_outputs)
      new_node = node_module.Node(node.layer, new_args, new_kwargs,
                                  new_outputs)

      # Add node to the post order for its pipeline stage.
      post_order_per_stage.setdefault(assignment.pipeline_stage,
                                      []).append(new_node)
      stages.add(assignment.pipeline_stage)

    computed_set = set()

    if inputs is not None:
      # If using explicit inputs, mark them as already computed.
      computed_set.update(str(id(x)) for x in inputs)

    # New post order executes all the layers within a pipeline stage and it
    # makes sure that all the layer inputs have already executed.
    for stage_id in range(max(stages) + 1):
      for node in post_order_per_stage.get(stage_id, []):
        all_inputs_executed = all(x in computed_set
                                  for x in node.flat_input_ids)
        if not all_inputs_executed:
          raise ValueError(
              f"Layer {node.outbound_layer.name} in pipeline stage {stage_id} "
              f"has a dependency from a pipeline stage which has not yet "
              f"been executed. Layers can only use outputs from current or "
              f"previous pipeline stages.")

        # Update computed_set.
        computed_set.update(node.flat_output_ids)

    output_tensors = [
        tensor_dict[str(id(x))].pop() for x in tf.nest.flatten(self.outputs)
    ]
    return post_order_per_stage, output_tensors

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _build_with_dtypes(self, input_shape, input_dtype):
    # Just call through to basic build. Functional models always have explicit
    # dtypes anyway as they always have Input layers.
    del input_dtype
    if not self.built:
      self.build(input_shape)

  def get_pipeline_stage_assignment(self):
    """Returns the pipeline stage assignment of the layers in the model.

    If :meth:`~keras.ipu.FunctionalExtension.set_pipeline_stage_assignment()`
    has been called before, then it returns a copy of the current assignment,
    otherwise returns a list of
    :class:`~keras.ipu.FunctionalLayerPipelineStageAssignment` and
    :class:`~keras.ipu.FunctionalNestedModelPipelineStageAssignment` for each
    layer invocation (excluding input layers) and nested model in the model. The
    list is in post order (execution order)."""
    if self._pipeline_stage_assignment:
      return copy.copy(self._pipeline_stage_assignment)

    post_order = self._create_post_order()

    output = []
    for node in post_order:
      layer = node.layer
      node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
      if isinstance(layer, extensions_base.KerasExtensionBase):
        output.append(
            FunctionalNestedModelPipelineStageAssignment(
                layer, node_index, layer.get_pipeline_stage_assignment()))
      else:
        output.append(FunctionalLayerPipelineStageAssignment(
            layer, node_index))

    return output

  def _validate_pipeline_stage_assignment(self, pipeline_stage_assignment):
    # A pipeline stage assignment is valid if the graph can be scheduled.
    self._get_pipelined_post_order(pipeline_stage_assignment)

  def _get_pipelining_from_nodes_supported(self):
    return True

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def _get_pipelining_from_nodes_delegate(self):
    """Populates pipelining information obtained from users annotating their
    model with `PipelineStage`"""
    post_order = self._create_post_order()

    def node_has_pipeline_stage(node):
      return (hasattr(node, "_pipeline_stage")
              or hasattr(node.outbound_layer, "_pipeline_stage"))

    any_node_has_pipeline_stage = any(
        node_has_pipeline_stage(node) for node in post_order)

    if not any_node_has_pipeline_stage:
      return

    # If any node has pipelining attached to it, then they all need it.
    # Create pipeline stage assignments.
    pipeline_stage_assignment = []
    for node in post_order:
      layer = node.layer
      node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access

      if isinstance(layer, extensions_base.KerasExtensionBase):
        # If layer is a nested model.
        pipeline_stage_assignment.append(
            FunctionalNestedModelPipelineStageAssignment(
                layer, node_index, layer.get_pipeline_stage_assignment()))
        continue

      if not hasattr(node, "_pipeline_stage"):
        if not hasattr(node.outbound_layer, "_pipeline_stage"):
          raise ValueError(
              f"All layers of a pipelined model must have an associated "
              f"pipeline stage. However, {node.outbound_layer.name} has not "
              f"been assigned to one. Pipeline stages can be assigned when a "
              f"layer is constructed, or each time a layer is called. "
              f"Different pipeline stages can assigned to each call.")
        node._pipeline_stage = node.outbound_layer._pipeline_stage  # pylint: disable=protected-access
      pipeline_stage_assignment.append(
          FunctionalLayerPipelineStageAssignment(
              layer, node_index, pipeline_stage=node._pipeline_stage))  # pylint: disable=protected-access

    self._validate_pipeline_stage_assignment(pipeline_stage_assignment)
    self._pipeline_stage_assignment = pipeline_stage_assignment

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def set_pipeline_stage_assignment(self, pipeline_stage_assignment):
    """Sets the pipeline stage assignment of all the invocations of all the
    layers in the model.

    Sets the pipeline stage assignment of all the invocations of all the
    layers (excluding input layers) in the model which is used to create a
    model-parallel execution of this model when calling `fit()`, `evaluate()`
    and `predict()`. Note that this pipelining stage assignment is ignored when
    using the `call()` function on this model.

    Args:
      pipeline_stage_assignment: A list of the same length as the total number
        of invocations of all the layers in this model (excluding input layers).
        All elements have to be instances of
        :class:`~keras.ipu.FunctionalLayerPipelineStageAssignment` which are
        used to indicate which pipeline stage a particular layer invocation
        should be assigned to.

    Raises:
      ValueError: `pipeline_stage_assignment` is not a valid assignment.
    """

    self._validate_pipeline_stage_assignment(pipeline_stage_assignment)
    self._pipeline_stage_assignment = pipeline_stage_assignment

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()
    self._pipeline_maximum_stage = None

  @tf.__internal__.tracking.no_automatic_dependency_tracking
  def reset_pipeline_stage_assignment(self):
    """Resets the pipeline stage assignment so that the model is no longer
    pipelined."""
    self._pipeline_stage_assignment = []
    self._pipeline_maximum_stage = None

    # Pipelining has changed therefore functions need to be recompiled.
    self._reset_ipu_extension()

  def print_pipeline_stage_assignment_summary(self,
                                              line_length=None,
                                              print_fn=None):
    """Prints a summary of the pipeline stage assignment of the model.

    Arguments:
        line_length: Total length of printed lines (for example, set this to
          adapt the display to different terminal window sizes).
        print_fn: Print function to use. It will be called on each line of the
          summary. You can set it to a custom function in order to capture the
          string summary. It defaults to `print` (prints to stdout).
    """
    line_length = line_length or 89

    def print_assignment_fn(assignment, print_row):
      layer = assignment.layer
      node_index = str(assignment.node_index)
      inbound_layers = tf.nest.flatten(assignment.inbound_layers)
      pipeline_stage = str(assignment.pipeline_stage)

      name = layer.name
      input_layer_names = [l.name for l in inbound_layers]

      cls_name = layer.__class__.__name__
      if not input_layer_names:
        first_input = ''
      else:
        first_input = input_layer_names[0]

      fields = [
          name + ' (' + cls_name + ') (' + node_index + ')', first_input,
          pipeline_stage
      ]
      print_row(fields)

      # Print other inputs on the new line.
      if len(input_layer_names) > 1:
        for i in range(1, len(input_layer_names)):
          fields = ['', input_layer_names[i], '']
          print_row(fields)

    headers = ['Layer (type) (node index)', 'Input Layers', 'Pipeline Stage']
    column_widths = [.4, .8, 1.]
    self._print_pipeline_stage_assignment_summary_impl(print_assignment_fn,
                                                       headers, column_widths,
                                                       line_length, print_fn)

  def _validate_call_function(self):
    call_function_overridden = not (
        hasattr(self.call, "__func__")
        and self.call.__func__ == functional.Functional.call)
    if call_function_overridden and self._is_pipelined():
      raise RuntimeError(
          f"The function `call` for the model {self.name} has been overridden. "
          f"This is not supported for pipelined Keras Functional models.")
