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
Optimizer wrapper for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ipu.ops import statistics_ops
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer as tf_optimizer

from keras import backend as K
from keras.ipu.optimizers.optimizer_v2_wrapper import _OptimizerV2Wrapper


def _is_power_of_two(x):
  return (x & (x - 1) == 0) and x != 0


class ALSOptimizer(_OptimizerV2Wrapper):
  """An optimizer that automatically computes and applies
  a loss scaling factor (LSF) prior to gradient computation.

  The LSF is computed such that the magnitude of the loss is increased
  to reduce numerical underflow. If the magnitude of the loss becomes too
  great and overflow occurs, then the LSF is automatically decreased.

  The automatic increase and decrease of the LSF is governed by sample
  statistics collected over computed gradients of type `float16`.

  Gradient statistics are collected on each backward pass, irresepective
  of `update_frequency`. Every `update_frequency` passes, the LSF is
  scaled by either `increase_factor` or `decrease_factor` depending on
  the state of the gradient statistics collected up to that point. If
  there is minimal overflow, then the LSF is scaled by `increase_factor`,
  otherwise it is scaled by `decrease_factor`. At LSF update time, the
  gradient statistics are reset for the following update period.

  Example using Keras Functional API:

  .. code-block:: python
    strategy = IPUStrategy()
    with strategy.scope():
      opt = SGD(0.01)
      opt_wrapper = ALSOptimizer(
        opt,
        initial_loss_scaling_factor=10.0,
        update_frequency=3,
        increase_factor=2.0)

      x, t = some_dataset_fn()
      input_l = Input(x.shape[1])

      dense = Dense(t.shape[1], activation='relu', dtype=np.float16)(input_l)

      m = Model(inputs=input_l,
                outputs=dense,
                gradient_accumulation_count=2)
      m.compile(optimizer=opt_wrapper, loss='mse')

      m.fit(x, t)

  Example using `tf.function`:

  .. code-block:: python
    strategy = IPUStrategy()
      opt = SGD(0.01)
      opt_wrapper = ALSOptimizer(
        opt,
        initial_loss_scaling_factor=10.0,
        update_frequency=3,
        increase_factor=2.0)

      x, t = some_dataset_fn()

      dense = Dense(t.shape[1], activation='relu', dtype=np.float16)

      @tf.function(jit_compile=True)
      def f(x, t):
        with GradientTape() as tape:
          y = dense(x)
          l = mean_squared_error(labels=t, predictions=y)

        opt_wrapper.minimize(l, dense.variables, tape=tape)
        return l

      loss = strategy.run(f, args=[x, t])
  """
  def __init__(self,
               opt,
               initial_loss_scaling_factor=1,
               update_frequency=8,
               increase_factor=2,
               max_loss_scaling_factor=32768,
               accumulate_statistics_over_update_period=True,
               ratio_threshold=10e-6,
               captured_grads_only=False,
               lpf_alpha=0.0,
               name="ALSOptimizer"):
    """Construct a new automatic loss scaling optimizer.

    Args:
      opt: An existing `Optimizer` to encapsulate.
      initial_loss_scaling_factor: The initial Loss Scaling Factor (LSF).
        Defaults to 1.
      update_frequency: The number of steps that should be taken before
        updating the LSF.
        Defaults to 8.
      increase_factor: The factor to scale the LSF by when increasing the LSF.
        Defaults to 2.
      max_loss_scaling_factor: The maximum value to which the LSF can increase.
        Defaults to 32768.
      accumulate_statistics_over_update_period: If true, statistics are
        accumulated over each `update_frequency` period, else they are
        collected once every `update_frequency` updates.
      ratio_threshold: The threshold over which the ratio of overflowed
        float16 gradients to all float16 gradients must exceed to cause a
        reduction in LSF. Ratios not meeting this threshold will cause an
        increase in LSF.
        Defaults to 10e-6.
      captured_grads_only: Whether to only use explicitly captured gradients (
        layers wrapped with either
        `keras.ipu.layers.capture_upstream_gradients.CaptureUpstreamGradients`
        or
        `keras.ipu.layers.capture_upstream_gradients.CaptureActivationGradients`
      ).
        Defaults to False.
      lpf_alpha: Low Pass Filtering (exponential type) coefficient, used for
        the collected gradient distributions when updating statistics. Setting
        this value to 1.0 will result in no statistical update of the
        ALSOptimizer state. Setting this value to 0.0 will result in no
        retention of the previous ALSOptimizer statistical state following an
        update period. Setting a lower value between these extrema should
        present a "smoother" LSF update pattern over time, such that
        `h(t) = alpha * h(t-1) + (1.0 - alpha) * h'(t)`, where h'(t) is the
        updated distribution at time `t`.
        Default is 0.0.
      name: Optional name prefix for the operation created when applying
        gradients.
        Defaults to "ALSOptimizer".
    """
    super().__init__(opt, name)

    self.update_frequency = update_frequency
    self.increase_factor = increase_factor
    self.max_loss_scaling_factor = max_loss_scaling_factor
    self.ratio_threshold = ratio_threshold
    self.initial_loss_scaling_factor = initial_loss_scaling_factor
    self.accumulate_statistics_over_update_period = \
      accumulate_statistics_over_update_period
    self.captured_grads_only = captured_grads_only
    self.lpf_alpha = lpf_alpha

    # Start with no collected stats.
    self._hist = variables.Variable(initial_value=array_ops.zeros(
        2, dtype=dtypes.float32),
                                    trainable=False,
                                    dtype=dtypes.float32,
                                    name="gradient_histogram")

    # Counter for LSF update.
    self._n = variables.Variable(initial_value=1,
                                 trainable=False,
                                 dtype=dtypes.int32,
                                 name="lsf_update_counter")

    # We have two histogram bins, each corresponding to a numerical state;
    # ok and overflow. As such, the binning of gradients is based
    # on the numerical extrema of the float16 representable range.
    self._hist_levels = constant_op.constant(
        [dtypes.float16.max - 2 * K.epsilon()], dtype=dtypes.float32)

    self._lsf = variables.Variable(initial_value=initial_loss_scaling_factor,
                                   trainable=False,
                                   dtype=dtypes.float32,
                                   name="loss_scaling_factor")

  def _assign_var(self, variable, value):
    @def_function.function(jit_compile=True)
    def f(var, val):
      return var.assign(val)

    return f(variable, value)

  def _update_histogram(self, h, skip_lpf=False):
    new_h = control_flow_ops.cond(
        ops.convert_to_tensor(skip_lpf), lambda: h,
        lambda: self.lpf_alpha * self.histogram + (1.0 - self.lpf_alpha) * h)

    self._assign_var(self._hist, new_h)

  def _lsf_update_due(self):
    return math_ops.equal(
        math_ops.floormod(self.update_counter, self.update_frequency), 0)

  def _should_update_hist(self):
    update_lsf = self._lsf_update_due()
    return math_ops.logical_or(
        ops.convert_to_tensor(self.accumulate_statistics_over_update_period),
        update_lsf)

  def _add_captured_grads(self, hist, captured_grads=None):
    if not captured_grads:
      return hist

    for g in captured_grads.values():
      if isinstance(g, (list, tuple)):
        for gg in g:
          hist = self._get_updated_hist(gg, hist)
      else:
        hist = self._get_updated_hist(g, hist)

    return hist

  def get_scaled_loss(self, loss):
    """Applies the current loss scaling factor to a given loss.

    Args:
      loss: The loss to be scaled.

    Returns:
      The scaled loss.
    """
    # Get as tensors, these may be variables.
    lsf = math_ops.cast(self.loss_scaling_factor, loss.dtype)
    return lsf * loss

  def _get_updated_hist(self, g, h):
    if g.dtype != dtypes.float16:
      return h

    g32 = array_ops.reshape(math_ops.cast(g, dtypes.float32), [-1])
    return statistics_ops.histogram_update(h,
                                           g32,
                                           self.clip_levels,
                                           absolute_of_input=True)

  def get_unscaled_gradients(self, grads):
    """Collects statistics from LSF scaled gradients and returns the
    same gradients unscaled.

    Args:
      grads: The gradients to be unscaled. These gradients should be
      computed from an LSF scaled loss.

    Returns:
      The unscaled gradients.
    """
    update_hist = self._should_update_hist()
    grads_rescaled = []

    def do_update_and_rescale(g, h, rescaled):
      # Add grads to histogram. If we are using only explicitly captured
      # gradients (passed via apply_gradients 'captured_grads' kwarg), then
      # skip the histogram update as these will be handled in apply_gradients.
      if not self.captured_grads_only:
        h = control_flow_ops.cond(update_hist,
                                  lambda: self._get_updated_hist(g, h),
                                  lambda: h)

      # Rescale grads.
      g_rescaled = g / math_ops.cast(self.loss_scaling_factor, g.dtype)

      rescaled.append(g_rescaled)
      return h

    hist = self.histogram
    is_list = isinstance(grads, list)
    if is_list:
      for g in grads:
        hist = do_update_and_rescale(g, hist, grads_rescaled)
    else:
      hist = do_update_and_rescale(grads, hist, grads_rescaled)

    grads_unscaled = grads_rescaled if is_list else grads_rescaled[0]
    self._update_histogram(hist)

    return grads_unscaled

  def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
    """Compute gradients of a scaled loss w.r.t. a given list of variables.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    def scaled_loss_fn():
      l = loss() if callable(loss) else loss
      return self.get_scaled_loss(l)

    grads_and_vars = super()._compute_gradients(  # pylint: disable=protected-access
        scaled_loss_fn,
        var_list,
        grad_loss=grad_loss,
        tape=tape)

    grads_and_vars_rescaled = []
    for g, v in grads_and_vars:
      gv = (self.get_unscaled_gradients(g), v)
      grads_and_vars_rescaled.append(gv)

    return grads_and_vars_rescaled

  def get_gradients(self, loss, params):
    """Compute gradients of a scaled loss w.r.t. a given list of params.

    Args:
      loss: A loss tensor.
      var_list: A list of variables to optimize.

    Returns:
      A list of LSF scaled gradients.
    """
    scaled_loss = self.get_scaled_loss(loss)
    scaled_grads = super().get_gradients(scaled_loss, params)

    return [self.get_unscaled_gradients(g) for g in scaled_grads]

  def _do_lsf_update(self, captured_grads=None):
    def _get_updated_lsf(histogram):
      ratio = histogram[1] / math_ops.reduce_sum(histogram)
      lsf = control_flow_ops.cond(
          math_ops.greater(ratio, self.ratio_threshold),
          lambda: self.loss_scaling_factor * self.decrease_factor,
          lambda: self.loss_scaling_factor * self.increase_factor)

      # Check the lsf hasn't over or under flowed.
      lsf = control_flow_ops.cond(math_ops.is_finite(lsf), lambda: lsf,
                                  lambda: self.loss_scaling_factor)

      # Check the lsf hasn't exceeded the maximum value.
      lsf = control_flow_ops.cond(
          math_ops.less(lsf, self.max_loss_scaling_factor), lambda: lsf,
          lambda: self.loss_scaling_factor)

      # Check that lsf >= 1
      return control_flow_ops.cond(math_ops.greater_equal(lsf,
                                                          1.0), lambda: lsf,
                                   lambda: self.loss_scaling_factor)

    # Are we due an LSF update?
    do_lsf_update = self._lsf_update_due()
    update_hist = self._should_update_hist()

    hist = control_flow_ops.cond(
        update_hist, lambda: self._add_captured_grads(
            self.histogram, captured_grads=captured_grads),
        lambda: self.histogram)

    # Get the latest LSF.
    lsf = control_flow_ops.cond(do_lsf_update, lambda: _get_updated_lsf(hist),
                                lambda: self.loss_scaling_factor)

    # Reset the gradient histogram if we have performed an LSF update.
    hist = control_flow_ops.cond(do_lsf_update,
                                 lambda: array_ops.zeros_like(hist),
                                 lambda: hist)

    # Update counter.
    n = control_flow_ops.cond(do_lsf_update, lambda: 1,
                              lambda: self.update_counter + 1)

    self._assign_var(self._lsf, lsf)
    self._update_histogram(hist, skip_lpf=do_lsf_update)
    self._assign_var(self._n, n)

  def apply_gradients(  #pylint: disable=arguments-differ
      self,
      grads_and_vars,
      captured_grads=None,
      global_step=None,
      name=None):
    """Apply gradients to variables and update the loss scale factor.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      captured_grads: A dictionary of captured gradients to be used for
        statistics collection when updating the ALS Loss Scale Factor.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    # Update ALSOptimizer internal state.
    self._do_lsf_update(captured_grads=captured_grads)

    # Apply grads.
    return super().apply_gradients(grads_and_vars, global_step, name)

  def reset(self):
    """Reset loss scaling."""
    self._assign_var(self._hist, array_ops.zeros_like(self.histogram))
    self._assign_var(self._n, 0)
    self._assign_var(self._lsf, self.initial_loss_scaling_factor)

  def get_config(self):
    """Returns the config of the `ALSOptimizer` instance.
    """
    config = super().get_config()
    config.update({
        'initial_loss_scaling_factor': self.initial_loss_scaling_factor,
        'update_frequency': self.update_frequency,
        'increase_factor': self.increase_factor,
        'max_loss_scaling_factor': self.max_loss_scaling_factor,
        'accumulate_statistics_over_update_period':
        self.accumulate_statistics_over_update_period,
        'ratio_threshold': self.ratio_threshold,
        'captured_grads_only': self.captured_grads_only
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates an `ALSOptimizer` from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same optimizer from the config
    dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An `ALSOptimizer` instance.
    """
    config = config.copy()
    _OptimizerV2Wrapper._verify_config(config)
    inner_config = config.pop('inner_optimizer_config')
    inner_type = config.pop('inner_optimizer_type')
    inner_opt = inner_type(**inner_config)

    return ALSOptimizer(inner_opt, **config)

  def get_name(self):
    return self._name

  @property
  def histogram(self):
    return ops.convert_to_tensor(self._hist)

  @histogram.setter
  def histogram(self, _):
    raise ValueError("histogram is a read only property.")

  @property
  def normalized_histogram(self):
    return statistics_ops.histogram_normalize(self.histogram)

  @normalized_histogram.setter
  def normalized_histogram(self, _):
    raise ValueError("normalized_histogram is a read only property.")

  @property
  def loss_scaling_factor(self):
    return ops.convert_to_tensor(self._lsf)

  @loss_scaling_factor.setter
  def loss_scaling_factor(self, _):
    raise ValueError("loss_scaling_factor is a read only property.")

  @property
  def update_counter(self):
    return ops.convert_to_tensor(self._n)

  @update_counter.setter
  def update_counter(self, _):
    raise ValueError("update_counter is a read only property.")

  @property
  def update_frequency(self):
    return self._update_frequency

  @update_frequency.setter
  def update_frequency(self, val):
    if val <= 0:
      raise ValueError("update_frequency must be nonzero and positive")
    self._update_frequency = val

  @property
  def increase_factor(self):
    return self._increase_factor

  @increase_factor.setter
  def increase_factor(self, val):
    if not _is_power_of_two(val):
      raise ValueError("increase_factor must be a power of two")

    self._increase_factor = val

  @property
  def decrease_factor(self):
    return 1.0 / self.increase_factor

  @decrease_factor.setter
  def decrease_factor(self, _):
    raise ValueError("decrease_factor is a read only property.")

  @property
  def clip_levels(self):
    return self._hist_levels

  @clip_levels.setter
  def clip_levels(self, _):
    raise ValueError("clip_levels is a read only property.")

  @property
  def initial_loss_scaling_factor(self):
    return self._initial_lsf

  @initial_loss_scaling_factor.setter
  def initial_loss_scaling_factor(self, val):
    if not _is_power_of_two(val):
      raise ValueError("initial_loss_scaling_factor must be a power of two")

    if hasattr(self, '_max_loss_scaling_factor') and \
      val >= self.max_loss_scaling_factor:
      raise ValueError("initial_loss_scaling_factor must be less "
                       "than max_loss_scaling_factor")

    if hasattr(self, '_max_loss_scaling_factor') and \
      val * self.increase_factor >= self.max_loss_scaling_factor:
      raise ValueError(
          "initial_loss_scaling_factor x increase_factor must be less "
          "than max_loss_scaling_factor")

    self._initial_lsf = val

  @property
  def max_loss_scaling_factor(self):
    return self._max_loss_scaling_factor

  @max_loss_scaling_factor.setter
  def max_loss_scaling_factor(self, val):
    if not _is_power_of_two(val):
      raise ValueError("max_loss_scaling_factor must be a power of two")

    if hasattr(self, '_max_loss_scaling_factor') and \
      val >= self.max_loss_scaling_factor:
      raise ValueError("initial_loss_scaling_factor must be less "
                       "than max_loss_scaling_factor")

    if hasattr(self, '_max_loss_scaling_factor') and \
      val * self.increase_factor >= self.max_loss_scaling_factor:
      if not _is_power_of_two(val):
        raise ValueError(
            "initial_loss_scaling_factor x increase_factor must be less "
            "than max_loss_scaling_factor")

    self._max_loss_scaling_factor = val

  @property
  def accumulate_statistics_over_update_period(self):
    return self._accumulate_stats

  @accumulate_statistics_over_update_period.setter
  def accumulate_statistics_over_update_period(self, val):
    self._accumulate_stats = val

  @property
  def ratio_threshold(self):
    return self._ratio_threshold

  @ratio_threshold.setter
  def ratio_threshold(self, val):
    if val >= 1.0 or val <= 0.0:
      raise ValueError(
          "ratio_threshold must be greater than zero and less than one")
    self._ratio_threshold = val

  @property
  def supports_captured_grads(self):
    return True

  @property
  def captured_grads_only(self):
    return self._captured_grads_only

  @captured_grads_only.setter
  def captured_grads_only(self, val):
    if not hasattr(self, '_captured_grads_only'):
      self._captured_grads_only = val
      return

    if val != self.captured_grads_only:
      self.reset()
      self._captured_grads_only = val

  @property
  def lpf_alpha(self):
    return self._lpf_alpha

  @lpf_alpha.setter
  def lpf_alpha(self, val):
    if not hasattr(self, '_lpf_alpha'):
      self._lpf_alpha = val
      return

    if val != self.lpf_alpha:
      self.reset()
      self._lpf_alpha = val


class ALSOptimizerGradientAccumulationWrapper(
    ga.GradientAccumulationOptimizerV2):
  def __init__(self,
               als_optimizer,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               reduction_method=ga.GradientAccumulationReductionMethod.SUM,
               name="ALSOptimizerGradientAccumulationWrapper"):
    """Construct a Gradient Accumulation Optimizer V2 for use with instances of
    `ALSOptimizer`.

    It should be noted that this Optimizer wraps the Keras `OptimizerV2`
    derived instance of `ALSOptimizer` in a non-Keras `Optimizer` interface.
    When performing Gradient Accumulation with Keras, it is not necessary to
    explicitly wrap `ALSOptimizer`, rather
    `keras.Model.set_gradient_accumulation_options` should be used.

    Args:
      opt: An existing `ALSOptimizer` to encapsulate.
      num_mini_batches: Number of mini-batches the gradients will be accumulated
        for.
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
        During the iterations in each optimizer step, the computed gradients can
        either be directly summed up or scaled such that we compute a mean of
        all gradients for each variable. Computing a mean avoids potential
        issues with overflow during accumulation especially when using float16,
        but gives smaller gradients and might require adjusting the
        learning-rate accordingly.
        Defaults to `GradientAccumulationReductionMethod.SUM`
        (see :class:`~tensorflow.python.ipu.gradient_accumulation.GradientAccumulationReductionMethod`)  # pylint: disable=line-too-long
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientAccumulationOptimizerV2".
    """

    if not isinstance(als_optimizer, ALSOptimizer):
      raise ValueError(
          "keras.ipu.ALSOptimizerGradientAccumulationWrapper can only be used "
          "with instances of keras.ipu.ALSOptimizer.")

    super().__init__(
        als_optimizer,
        num_mini_batches,
        offload_weight_update_variables=offload_weight_update_variables,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
        dtype=dtype,
        reduction_method=reduction_method,
        name=name)

  def compute_gradients(  #pylint: disable=arguments-differ
      self,
      loss,
      var_list=None,
      gate_gradients=tf_optimizer.Optimizer.GATE_OP,  #pylint: disable=unused-argument
      aggregation_method=None,  #pylint: disable=unused-argument
      colocate_gradients_with_ops=False,  #pylint: disable=unused-argument
      grad_loss=None):  #pylint: disable=unused-argument
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the `get_gradients` method of the wrapped `ALSOptimizer`.
    The gradients will be aggregated in this wrappers `apply_gradients` method
    so that the gradients may be modified with options such as clipping with
    per replica global norm if needed.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    grads = self._opt.get_gradients(loss, var_list)
    return list(zip(grads, var_list))

  def apply_gradients(  #pylint: disable=arguments-differ
      self,
      grads_and_vars,
      global_step=None,
      captured_grads=None,
      name=None):
    """Apply gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      captured_grads: An optional dictionary (indexed by tags) of captured
      grads to be forwarded onto the the wrapped `keras.ipu.ALSOptimizer`
      instance.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """

    # GradientAccumulationOptimizerV2 has no support for the passing of
    # captured grads (nor should it; it's a concept we introduced for Keras).
    # So, when GradientAccumulationOptimizerV2.apply_gradients calls into
    # ALSOptimizer.apply_gradients it won't pass captured_grads, meaning
    # that ALSOptimizer._do_lsf_update won't get them either. So we add them
    # to the histogram here.
    def handle_captured():
      return self.als_optimizer._add_captured_grads(  # pylint: disable=protected-access
          self.als_optimizer.histogram,
          captured_grads=captured_grads)

    if captured_grads:
      should_update = self.als_optimizer._should_update_hist()  # pylint: disable=protected-access
      hist = control_flow_ops.cond(should_update, handle_captured,
                                   lambda: self.als_optimizer.histogram)

      self.als_optimizer._update_histogram(  # pylint: disable=protected-access
          hist,
          skip_lpf=math_ops.logical_not(should_update))

    return super().apply_gradients(  # pylint: disable=protected-access
        grads_and_vars,
        global_step=global_step,
        name=name)

  @property
  def als_optimizer(self):
    return self._opt

  @property
  def histogram(self):
    return self.als_optimizer.histogram

  @property
  def normalized_histogram(self):
    return self.als_optimizer.normalized_histogram

  @property
  def loss_scaling_factor(self):
    return self.als_optimizer.loss_scaling_factor

  @property
  def update_counter(self):
    return self.als_optimizer.update_counter

  @property
  def update_frequency(self):
    return self.als_optimizer.update_frequency

  @property
  def increase_factor(self):
    return self.als_optimizer.increase_factor

  @property
  def decrease_factor(self):
    return self.als_optimizer.decrease_factor

  @property
  def clip_levels(self):
    return self.als_optimizer.clip_levels

  @property
  def initial_loss_scaling_factor(self):
    return self.als_optimizer.initial_loss_scaling_factor

  @property
  def max_loss_scaling_factor(self):
    return self.als_optimizer.max_loss_scaling_factor

  @property
  def accumulate_statistics_over_update_period(self):
    return self.als_optimizer.accumulate_statistics_over_update_period

  @property
  def ratio_threshold(self):
    return self.als_optimizer.ratio_threshold

  @property
  def supports_captured_grads(self):
    return self.als_optimizer.supports_captured_grads

  @property
  def captured_grads_only(self):
    return self.als_optimizer.captured_grads_only

  @property
  def lpf_alpha(self):
    return self.als_optimizer.lpf_alpha
