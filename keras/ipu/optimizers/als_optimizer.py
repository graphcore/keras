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

from dataclasses import dataclass

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

from tensorflow.python.util import lazy_loader
puo = lazy_loader.LazyLoader(
    "parameter_unscaling_optimizer", globals(),
    "keras.ipu.optimizers.parameter_unscaling_optimizer")

import keras
from keras import backend as K
from keras.optimizer_v2.optimizer_v2 import OptimizerV2
from keras.ipu.optimizers.optimizer_v2_wrapper import _OptimizerV2Wrapper
from keras.ipu.optimizers.gradient_accumulation_optimizer import GradientAccumulationOptimizer


def _is_power_of_two(x):
  return (x & (x - 1) == 0) and x != 0


@dataclass
class ALSDefaults:
  initial_loss_scaling_factor = 1
  update_frequency = 8
  increase_factor = 2
  max_loss_scaling_factor = 32768
  accumulate_statistics_over_update_period = True
  ratio_threshold = 10e-6
  captured_grads_only = False
  lpf_alpha = 0.0


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
  def __init__(
      self,
      opt,
      initial_loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor,
      update_frequency=ALSDefaults.update_frequency,
      increase_factor=ALSDefaults.increase_factor,
      max_loss_scaling_factor=ALSDefaults.max_loss_scaling_factor,
      accumulate_statistics_over_update_period=ALSDefaults.
      accumulate_statistics_over_update_period,
      ratio_threshold=ALSDefaults.ratio_threshold,
      captured_grads_only=ALSDefaults.captured_grads_only,
      lpf_alpha=ALSDefaults.lpf_alpha,
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
    self._hist = OptimizerV2.add_weight(self,
                                        'gradient_histogram', (2),
                                        dtype=dtypes.float32,
                                        trainable=False)

    # Counter for LSF update.
    self._n = OptimizerV2.add_weight(
        self,
        'lsf_update_counter', (),
        dtype=dtypes.int32,
        initializer=keras.initializers.Constant(0),
        trainable=False)

    self._lsf = OptimizerV2.add_weight(
        self,
        'loss_scaling_factor', (),
        dtype=dtypes.float32,
        initializer=keras.initializers.Constant(initial_loss_scaling_factor),
        trainable=False)

    self._prev_lsf = OptimizerV2.add_weight(
        self,
        'prev_loss_scaling_factor', (),
        dtype=dtypes.float32,
        initializer=keras.initializers.Constant(initial_loss_scaling_factor),
        trainable=False)

  @staticmethod
  def _is_als_hyper(name):
    return name in ('update_frequency', 'increase_factor',
                    'max_loss_scaling_factor', 'ratio_threshold',
                    'initial_loss_scale_factor',
                    'accumulate_statistics_over_update_period',
                    'captured_grads_only', 'lpf_alpha')

  def __setattr__(self, name, value):
    if ALSOptimizer._is_als_hyper(name):
      OptimizerV2.__setattr__(self, name, value)
    else:
      super().__setattr__(name, value)

  def __getattribute__(self, name):
    if ALSOptimizer._is_als_hyper(name):
      return OptimizerV2.__getattribute__(self, name)
    return super().__getattribute__(name)

  def _set_hyper(self, name, value):
    if ALSOptimizer._is_als_hyper(name):
      OptimizerV2._set_hyper(self, name, value)  # pylint: disable=protected-access
    else:
      super()._set_hyper(name, value)  # pylint: disable=protected-access

  def _get_hyper(self, name, dtype=None):
    if ALSOptimizer._is_als_hyper(name):
      return OptimizerV2._get_hyper(self, name, dtype=dtype)  # pylint: disable=protected-access
    return super()._get_hyper(name, dtype=dtype)  # pylint: disable=protected-access

  def _create_hypers(self):
    super()._create_hypers()  # pylint: disable=protected-access
    OptimizerV2._create_hypers(self)  # pylint: disable=protected-access

  def _assign_var(self, variable, value):
    @def_function.function(jit_compile=True)
    def f(var, val):
      return var.assign(val)

    return f(variable, value)

  def _update_histogram(self, h, skip_lpf=False):
    alpha = self._get_hyper('lpf_alpha')
    new_h = control_flow_ops.cond(
        ops.convert_to_tensor(skip_lpf), lambda: h,
        lambda: alpha * self._hist + (1.0 - alpha) * h)

    self._assign_var(self._hist, new_h)

  def _lsf_update_due(self):
    update_freq = self._get_hyper('update_frequency')

    def test_if_due():
      return math_ops.logical_and(
          math_ops.greater(self._n, 0),
          math_ops.equal(math_ops.floormod(self._n, update_freq), 0))

    return control_flow_ops.cond(math_ops.equal(update_freq, 1),
                                 lambda: ops.convert_to_tensor(True),
                                 test_if_due)

  def _should_update_hist(self):
    accum = self._get_hyper('accumulate_statistics_over_update_period')
    return math_ops.logical_or(accum, self._lsf_update_due())

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
    lsf = math_ops.cast(self._lsf, loss.dtype)
    return lsf * loss

  def _get_updated_hist(self, g, h):
    if g.dtype != dtypes.float16:
      return h

    # We have two histogram bins, each corresponding to a numerical state;
    # ok and overflow. As such, the binning of gradients is based
    # on the numerical extrema of the float16 representable range.
    clip_levels = constant_op.constant([dtypes.float16.max - 2 * K.epsilon()],
                                       dtype=dtypes.float32)

    g32 = array_ops.reshape(math_ops.cast(g, dtypes.float32), [-1])
    return statistics_ops.histogram_update(h,
                                           g32,
                                           clip_levels,
                                           absolute_of_input=True)

  def _add_grads(self, grads):
    """Collects statistics from LSF scaled gradients.

    Args:
      grads: The gradients to be unscaled. These gradients should be
      computed from an LSF scaled loss.
    """
    update_hist = self._should_update_hist()

    def do_update(g, h):
      # Add grads to histogram. If we are using only explicitly captured
      # gradients (passed via apply_gradients 'captured_grads' kwarg), then
      # skip the histogram update as these will be handled in apply_gradients.
      captured_only = self._get_hyper('captured_grads_only')
      return control_flow_ops.cond(
          math_ops.logical_and(update_hist,
                               math_ops.logical_not(captured_only)),
          lambda: self._get_updated_hist(g, h), lambda: h)

    hist = self._hist
    is_list = isinstance(grads, list)
    if is_list:
      for g in grads:
        hist = do_update(g, hist)
    else:
      hist = do_update(grads, hist)

    self._update_histogram(hist)

  def _get_unscaled_gradients(self, grads):
    """Unscales the LSF scaled gradients.

    Args:
      grads: The gradients to be unscaled. These gradients should be
      computed from an LSF scaled loss. If the wrapped optimizer is
      an instance of a specialization in
      `keras.ipu.optimizers.als_optimizer_specializations`, then a
      no-op is performed (i.e. identity) as the rescaling will instead
      be performed in the optimizer specialization.

    Returns:
      The unscaled gradients.
    """
    if isinstance(self._opt, puo._ParameterUnscalingOptimizer):  # pylint: disable=protected-access
      # Do a sanity check; we should only have one of these optimizer types
      # if this ALSOptimizer is a _ParameterUnscalingALSOptimizer.
      if not isinstance(self, puo._ParameterUnscalingALSOptimizer):  # pylint: disable=protected-access
        raise ValueError(
            "ALSOptimizer has been used to wrap an instance of "
            "_ParameterUnscalingOptimizer, but is not itself an instance "
            "of _ParameterUnscalingALSOptimizer.")

      return grads

    grads_rescaled = []

    def do_rescale(g, rescaled):
      rescaled.append(g / math_ops.cast(self.loss_scaling_factor, g.dtype))

    is_list = isinstance(grads, list)
    if is_list:
      for g in grads:
        do_rescale(g, grads_rescaled)
    else:
      do_rescale(grads, grads_rescaled)

    return grads_rescaled if is_list else grads_rescaled[0]

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

    return super()._compute_gradients(  # pylint: disable=protected-access
        scaled_loss_fn,
        var_list,
        grad_loss=grad_loss,
        tape=tape)

  def get_gradients(self, loss, params):
    """Compute gradients of a scaled loss w.r.t. a given list of params.

    Args:
      loss: A loss tensor.
      var_list: A list of variables to optimize.

    Returns:
      A list of LSF scaled gradients.
    """
    scaled_loss = self.get_scaled_loss(loss)
    return super().get_gradients(scaled_loss, params)

  def _do_lsf_update(self, captured_grads=None):
    def _get_updated_lsf(histogram):
      inc_factor = self._get_hyper('increase_factor')

      ratio = histogram[1] / math_ops.reduce_sum(histogram)
      threshold = self._get_hyper('ratio_threshold')
      lsf = control_flow_ops.cond(math_ops.greater(ratio, threshold),
                                  lambda: self._lsf / inc_factor,
                                  lambda: self._lsf * inc_factor)

      # Check the lsf hasn't over or under flowed.
      lsf = control_flow_ops.cond(math_ops.is_finite(lsf), lambda: lsf,
                                  lambda: self._lsf)

      # Check the lsf hasn't exceeded the maximum value.
      max_lsf = self._get_hyper('max_loss_scaling_factor')
      lsf = control_flow_ops.cond(math_ops.less(lsf, max_lsf), lambda: lsf,
                                  lambda: self._lsf)

      # Check that lsf >= 1
      return control_flow_ops.cond(math_ops.greater_equal(lsf, 1.0),
                                   lambda: lsf, lambda: self._lsf)

    # Are we due an LSF update?
    do_lsf_update = self._lsf_update_due()
    update_hist = self._should_update_hist()

    hist = control_flow_ops.cond(
        update_hist, lambda: self._add_captured_grads(
            self._hist, captured_grads=captured_grads), lambda: self._hist)

    # Get the latest LSF.
    lsf = control_flow_ops.cond(do_lsf_update, lambda: _get_updated_lsf(hist),
                                lambda: self._lsf)

    prev_lsf = control_flow_ops.cond(math_ops.equal(lsf, self._lsf),
                                     lambda: self._prev_lsf, lambda: self._lsf)

    # Reset the gradient histogram if we have performed an LSF update.
    hist = control_flow_ops.cond(do_lsf_update,
                                 lambda: array_ops.zeros_like(hist),
                                 lambda: hist)

    # Update counter.
    n = control_flow_ops.cond(do_lsf_update, lambda: 0, lambda: self._n + 1)

    self._assign_var(self._lsf, lsf)
    self._assign_var(self._prev_lsf, prev_lsf)
    self._update_histogram(hist, skip_lpf=do_lsf_update)
    self._assign_var(self._n, n)

    if isinstance(self._opt, puo._ParameterUnscalingOptimizer):  # pylint: disable=protected-access
      self._opt._lsf = ops.convert_to_tensor(self._lsf)  # pylint: disable=protected-access
      self._opt._prev_lsf = ops.convert_to_tensor(self._prev_lsf)  # pylint: disable=protected-access

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
    # Add the scaled grads to the histogram and get the unscaled grads for
    # the update.
    grads_and_vars_rescaled = []
    for g, v in grads_and_vars:
      self._add_grads(g)
      gv = (self._get_unscaled_gradients(g), v)
      grads_and_vars_rescaled.append(gv)

    # Update ALSOptimizer internal state.
    self._do_lsf_update(captured_grads=captured_grads)

    # Apply grads.
    return super().apply_gradients(grads_and_vars_rescaled, global_step, name)

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
  def previous_loss_scaling_factor(self):
    return ops.convert_to_tensor(self._prev_lsf)

  @previous_loss_scaling_factor.setter
  def previous_loss_scaling_factor(self, _):
    raise ValueError("previous_loss_scaling_factor is a read only property.")

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
    self._set_hyper('update_frequency', self._update_frequency)

  @property
  def increase_factor(self):
    return self._increase_factor

  @increase_factor.setter
  def increase_factor(self, val):
    if not _is_power_of_two(val):
      raise ValueError("increase_factor must be a power of two")

    self._increase_factor = val
    self._set_hyper('increase_factor', float(self._increase_factor))

  @property
  def decrease_factor(self):
    return 1.0 / self.increase_factor

  @decrease_factor.setter
  def decrease_factor(self, _):
    raise ValueError("decrease_factor is a read only property.")

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
    self._set_hyper('initial_loss_scale_factor', float(self._initial_lsf))

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
    self._set_hyper('max_loss_scaling_factor',
                    float(self._max_loss_scaling_factor))

  @property
  def accumulate_statistics_over_update_period(self):
    return self._accumulate_stats

  @accumulate_statistics_over_update_period.setter
  def accumulate_statistics_over_update_period(self, val):
    self._accumulate_stats = val
    self._set_hyper('accumulate_statistics_over_update_period', val)

  @property
  def ratio_threshold(self):
    return self._ratio_threshold

  @ratio_threshold.setter
  def ratio_threshold(self, val):
    if val >= 1.0 or val <= 0.0:
      raise ValueError(
          "ratio_threshold must be greater than zero and less than one")
    self._ratio_threshold = val
    self._set_hyper('ratio_threshold', self._ratio_threshold)

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
      self._set_hyper('captured_grads_only', self._captured_grads_only)
      return

    if val != self.captured_grads_only:
      self.reset()
      self._captured_grads_only = val
      self._set_hyper('captured_grads_only', self._captured_grads_only)

  @property
  def lpf_alpha(self):
    return self._lpf_alpha

  @lpf_alpha.setter
  def lpf_alpha(self, val):
    if not hasattr(self, '_lpf_alpha'):
      self._lpf_alpha = val
      self._set_hyper('lpf_alpha', self._lpf_alpha)
      return

    if val != self.lpf_alpha:
      self.reset()
      self._lpf_alpha = val
      self._set_hyper('lpf_alpha', self._lpf_alpha)


class ALSGradientAccumulationOptimizer(GradientAccumulationOptimizer):
  def __init__(self,
               als_optimizer,
               num_mini_batches,
               offload_weight_update_variables=None,
               replicated_optimizer_state_sharding=False,
               dtype=None,
               reduction_method=ga.GradientAccumulationReductionMethod.SUM,
               name="ALSGradientAccumulationOptimizer"):
    if not isinstance(als_optimizer, ALSOptimizer):
      raise ValueError(
          "ALSGradientAccumulationOptimizer can only be used with instances "
          "of ALSOptimizer.")

    super().__init__(
        als_optimizer,
        num_mini_batches,
        offload_weight_update_variables=offload_weight_update_variables,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
        dtype=dtype,
        reduction_method=reduction_method,
        name=name)

  def get_gradients(self, loss, params):
    return self.als_optimizer.get_gradients(loss, params)

  def apply_gradients(  #pylint: disable=arguments-differ
      self,
      grads_and_vars,
      captured_grads=None,
      name=None,
      experimental_aggregate_gradients=True):
    # Add the scaled grads to the histogram and get the unscaled grads for
    # the update.
    grads_and_vars_rescaled = []
    for g, v in grads_and_vars:
      self.als_optimizer._add_grads(g)  # pylint: disable=protected-access
      gv = (self.als_optimizer._get_unscaled_gradients(g), v)  # pylint: disable=protected-access
      grads_and_vars_rescaled.append(gv)

    # Update ALSOptimizer internal state.
    self.als_optimizer._do_lsf_update(captured_grads=captured_grads)  # pylint: disable=protected-access

    return super().apply_gradients(
        grads_and_vars_rescaled,
        name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

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
  def previous_loss_scaling_factor(self):
    return self.als_optimizer.previous_loss_scaling_factor

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

  @classmethod
  def from_config(cls, config, custom_objects=None):  # pylint: disable=missing-type-doc,missing-return-type-doc
    """Creates an `ALSGradientAccumulationOptimizer` from its config.

    This method is the reverse of `get_config` (inherited from
    `GradientAccumulationOptimizer`), capable of instantiating the same
    optimizer from the config dictionary.

    Arguments:
        config: A Python dictionary, typically the output of get_config.
        custom_objects: A Python dictionary mapping names to additional Python
          objects used to create this optimizer, such as a function used for a
          hyperparameter.

    Returns:
        An `ALSGradientAccumulationOptimizer` instance.
    """
    config = config.copy()
    _OptimizerV2Wrapper._verify_config(config)  # pylint: disable=protected-access
    inner_config = config.pop('inner_optimizer_config')

    inner_type = config.pop('inner_optimizer_type')
    if not inner_type == ALSOptimizer:
      raise ValueError(
          "ALSGradientAccumulationOptimizer.from_config can only be used with "
          "configurations that have ALSOptimizer as the specified inner_type.")

    inner_opt = ALSOptimizer.from_config(inner_config)

    return ALSGradientAccumulationOptimizer(inner_opt, **config)
