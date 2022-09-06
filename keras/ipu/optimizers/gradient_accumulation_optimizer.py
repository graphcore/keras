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
Optimizer wrapper to accumulate gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.compiler.plugin.poplar.driver import threestate_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.ipu.ops import op_util
from tensorflow.python.ipu.optimizers import \
gradient_accumulation_optimizer as ga

from keras.ipu.optimizers.optimizer_v2_wrapper import _OptimizerV2Wrapper


class GradientAccumulationOptimizer(_OptimizerV2Wrapper):
  """An optimizer which performs the weight update after multiple batches
  have been accumulated.
  """
  @staticmethod
  def bool_to_three_state(value, default):
    if value is None:
      return default

    if value:
      return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_ON)

    return threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)

  def __new__(cls, opt, num_mini_batches, *nargs, **kwargs):  #pylint: disable=unused-argument
    if num_mini_batches == 1:
      return opt
    return super(GradientAccumulationOptimizer, cls).__new__(cls)

  def __init__(self,
               opt,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               reduction_method=ga.GradientAccumulationReductionMethod.SUM,
               name="GradientAccumulationOptimizer"):
    """
    Construct a GradientAccumulationOptimizer. Note this doesn't divide
    by the number of mini-batches.

    Args:
      opt: An existing optimizer to encapsulate.
      num_mini_batches: The number of mini-batches the gradients
                        will be accumulated for.
      offload_weight_update_variables: When enabled, any `tf.Variable` which is
        only used by the weight update of the pipeline (for example the
        accumulator variable when using the `tf.MomentumOptimizer`), will be
        stored in the remote memory. During the weight update this variable will
        be streamed onto the device and then streamed back to the remote memory
        after it has been updated. Requires the machine to be configured with
        support for `Poplar remote buffers`. Offloading variables into remote
        memory can reduce maximum memory liveness, but can also increase the
        computation time of the weight update.
        When set to `None` the variables will be placed in either in-processor
        or remote memory automatically based on the current best placement
        strategy.
      replicated_optimizer_state_sharding: If True, any `tf.Variable` which is
        offloaded (for example the accumulator variable when using the
        `tf.MomentumOptimizer`), will be partitioned across the replicas.
        This can exploit the additional bandwidth of the IPU-Links to improve
        overall throughput, however it might increase the code size and hence
        the model might need adjusting (for example the PopLibs option
        `availableMemoryProportion` might need to be changed).
      dtype: The data type used for the gradient accumulation buffer.
        One of:
          - `None`: Use an accumulator of the same type as the variable type.
          - A `DType`: Use this type for all the accumulators.
          - A callable that takes the variable and returns a `DType`: Allows
            specifying the accumulator type on a per-variable basis.

        The gradients passed to `Optimizer.apply_gradients` will have the dtype
        requested here. If that dtype is different from the variable dtype
        a cast is needed at some point to make them compatible. If you want
        to cast the gradients immediately, you can wrap your optimizer in the
        `MapGradientOptimizer` with a `tf.cast`.
      reduction_method: Reduction method to use when accumulating gradients.
        During the iterations in each optimizer step, the computed gradients
        can either be directly summed up or scaled such that we compute a mean
        of all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using
        float16, but gives smaller gradients and might require adjusting
        the learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.optimizers.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizer".
    """
    if num_mini_batches < 1:
      raise ValueError("num_mini_batches must be >= 1")

    super().__init__(opt, name=name)
    self._num_mini_batches = num_mini_batches
    self._offload_weight_update_variables = self.bool_to_three_state(
        offload_weight_update_variables,
        threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_UNDEFINED))
    self._replicated_optimizer_state_sharding = self.bool_to_three_state(
        replicated_optimizer_state_sharding,
        self._offload_weight_update_variables)
    self._dtype = dtype

    self._reduction_method = reduction_method

  def _resource_apply_dense(self, grad, handle, apply_state):  # pylint: disable=missing-type-doc,missing-return-type-doc,missing-raises-doc
    """Apply gradient to variable referenced by `handle`.

    Args:
      grad: The gradient to be applied.
      handle: A handle to the variable to apply the gradient to.
      apply_state: State passed down to wrapped optimizer's apply functions.
    Returns:
      The updated variable.
    """
    accum_scale, grad_scale = ga._compute_scales(  # pylint: disable=protected-access
        self._reduction_method, self._num_mini_batches)
    acc_gv = op_util.accumulate_gradients([(grad, handle)], self._dtype,
                                          accum_scale, grad_scale)

    if len(acc_gv) != 1:
      raise RuntimeError(
          "Single (grad, var) pair expected after gradient accumulation.")
    acc_grad, acc_var = acc_gv[0]

    # Create an explicit function call for the apply gradients - note that we
    # allow external captures here.
    apply_grad_ops = []

    def resource_update_(gradient_accumulation_count):
      gen_poputil_ops.gradient_accumulation_count(gradient_accumulation_count)
      updated_var = \
        super(GradientAccumulationOptimizer, self)._resource_apply_dense(  # pylint: disable=protected-access
            acc_grad, acc_var, apply_state)

      if updated_var is not None:
        apply_grad_ops.append(updated_var)

    return op_util.create_resource_update(
        resource_update_,
        self._name,  # pylint: disable=protected-access
        apply_grad_ops,
        self._offload_weight_update_variables,
        self._replicated_optimizer_state_sharding,
        self._num_mini_batches)

  def get_config(self):
    """
    Returns the config of the `GradientAccumulationOptimizer` instance.
    """
    config = super().get_config()
    config.update({
        'num_mini_batches': self._num_mini_batches,
        'offload_weight_update_variables':
        self._offload_weight_update_variables,
        'replicated_optimizer_state_sharding':
        self._replicated_optimizer_state_sharding,
        'dtype': self._dtype,
        'reduction_method': self._reduction_method
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):  # pylint: disable=missing-type-doc,missing-return-type-doc
    """Creates a `GradientAccumulationOptimizer` from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        A `GradientAccumulationOptimizer` instance.
    """
    if cls != GradientAccumulationOptimizer:
      raise ValueError(
          "GradientAccumulationOptimizer.from_config can only be used to "
          "create a GradientAccumulationOptimizer instance. If subclassing, "
          "the child class must implement its own from_config method.")

    config = config.copy()
    _OptimizerV2Wrapper._verify_config(config)  # pylint: disable=protected-access
    inner_config = config.pop('inner_optimizer_config')
    inner_type = config.pop('inner_optimizer_type')
    inner_opt = inner_type(**inner_config)

    return GradientAccumulationOptimizer(inner_opt, **config)
