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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

from keras import backend as K
from keras.ipu.optimizers.optimizer_v2_wrapper import _OptimizerV2Wrapper


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
        increase_factor=2.0,
        decrease_factor=0.5)

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
        increase_factor=2.0,
        decrease_factor=0.5)

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
               initial_loss_scaling_factor=1.0,
               update_frequency=8,
               increase_factor=2.0,
               decrease_factor=0.5,
               max_loss_scaling_factor=32768,
               accumulate_statistics_over_update_period=True,
               ratio_threshold=10e-6,
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
      decrease_factor: The factor to scale the LSF by when decreasing the LSF.
        Defaults to 0.5.
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
      name: Optional name prefix for the operation created when applying
        gradients.
        Defaults to "ALSOptimizer".
    """
    super().__init__(opt, name)

    self.update_frequency = update_frequency
    self.increase_factor = increase_factor
    self.decrease_factor = decrease_factor
    self.max_loss_scaling_factor = max_loss_scaling_factor
    self.ratio_threshold = ratio_threshold
    self.initial_loss_scaling_factor = initial_loss_scaling_factor
    self.accumulate_statistics_over_update_period = \
      accumulate_statistics_over_update_period

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

  def _lsf_update_due(self):
    return math_ops.equal(
        math_ops.floormod(self.update_counter, self.update_frequency), 0)

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

  def get_unscaled_gradients(self, grads):
    """Collects statistics from LSF scaled gradients and returns the
    same gradients unscaled.

    Args:
      grads: The gradients to be unscaled. These gradients should be
      computed from an LSF scaled loss.

    Returns:
      The unscaled gradients.
    """
    update_hist = self._lsf_update_due()
    update_hist = math_ops.logical_or(
        ops.convert_to_tensor(self.accumulate_statistics_over_update_period),
        update_hist)

    hist = self.histogram

    grads_rescaled = []

    def get_updated_hist(g, h):
      if g.dtype != dtypes.float16:
        return h

      g32 = array_ops.reshape(math_ops.cast(g, dtypes.float32), [-1])
      return statistics_ops.histogram_update(h,
                                             g32,
                                             self.clip_levels,
                                             absolute_of_input=True)

    def do_update_and_rescale(g, h, rescaled):
      # Add grads to histogram.
      h = control_flow_ops.cond(update_hist, lambda: get_updated_hist(g, h),
                                lambda: h)

      # Rescale grads.
      g_rescaled = g / math_ops.cast(self.loss_scaling_factor, g.dtype)

      rescaled.append(g_rescaled)
      return h

    is_list = isinstance(grads, list)
    if is_list:
      for g in grads:
        hist = do_update_and_rescale(g, hist, grads_rescaled)
    else:
      hist = do_update_and_rescale(grads, hist, grads_rescaled)

    grads_unscaled = grads_rescaled if is_list else grads_rescaled[0]
    self._assign_var(self._hist, hist)
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

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):  #pylint: disable=arguments-differ
    """Apply gradients to variables and update the loss scale factor.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
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

    # Get the latest LSF.
    lsf = control_flow_ops.cond(do_lsf_update,
                                lambda: _get_updated_lsf(self.histogram),
                                lambda: self.loss_scaling_factor)

    # Reset the gradient histogram if we have performed an LSF update.
    hist = control_flow_ops.cond(do_lsf_update,
                                 lambda: array_ops.zeros_like(self.histogram),
                                 lambda: self.histogram)

    # Update counter.
    n = control_flow_ops.cond(do_lsf_update, lambda: 1,
                              lambda: self.update_counter + 1)

    self._assign_var(self._lsf, lsf)
    self._assign_var(self._hist, hist)
    self._assign_var(self._n, n)

    # Apply grads.
    return super().apply_gradients(grads_and_vars, global_step, name)

  def reset(self):
    """Reset loss scaling."""
    self._assign_var(self._hist, array_ops.zeros_like(self.histogram))
    self._assign_var(self._n, 0)
    self._assign_var(self._lsf, self.initial_loss_scaling_factor)

  def get_config(self):
    """
    Returns the config of the `ALSOptimizer` instance.
    """
    config = super().get_config()
    config.update({
        'initial_loss_scaling_factor': self.initial_loss_scaling_factor,
        'update_frequency': self.update_frequency,
        'increase_factor': self.increase_factor,
        'decrease_factor': self.decrease_factor,
        'max_loss_scaling_factor': self.max_loss_scaling_factor,
        'accumulate_statistics_over_update_period':
        self.accumulate_statistics_over_update_period,
        'ratio_threshold': self.ratio_threshold
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
    dec = self.decrease_factor if hasattr(self, '_decrease_factor') else None

    if val <= 0:
      raise ValueError("increase_factor must be nonzero and positive")

    if dec and val <= dec:
      raise ValueError("increase_factor must be greater than decrease_factor")

    self._increase_factor = val

  @property
  def decrease_factor(self):
    return self._decrease_factor

  @decrease_factor.setter
  def decrease_factor(self, val):
    inc = self.increase_factor if hasattr(self, '_increase_factor') else None

    if val <= 0:
      raise ValueError("decrease_factor must be nonzero and positive")

    if inc and val >= inc:
      raise ValueError("decrease_factor must be less than increase_factor")

    self._decrease_factor = val

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
    if val <= 0:
      raise ValueError(
          "initial_loss_scaling_factor must be nonzero and positive")

    if val >= self.max_loss_scaling_factor:
      raise ValueError("initial_loss_scaling_factor must be less "
                       "than max_loss_scaling_factor")

    if val * self.increase_factor >= self.max_loss_scaling_factor:
      raise ValueError(
          "initial_loss_scaling_factor x increase_factor must be less "
          "than max_loss_scaling_factor")

    self._initial_lsf = val

  @property
  def max_loss_scaling_factor(self):
    return self._max_loss_scaling_factor

  @max_loss_scaling_factor.setter
  def max_loss_scaling_factor(self, val):
    if val <= 1:
      raise ValueError("max_loss_scaling_factor must be greater than one")

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
