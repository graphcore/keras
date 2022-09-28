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
Adam specializations for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from keras.ipu.optimizers.als_optimizer import ALSDefaults
from keras.ipu.optimizers import parameter_unscaling_optimizer as puo
from keras.optimizer_v2 import adam


class _ParameterUnscalingAdam(puo._ParameterUnscalingOptimizer, adam.Adam):  # pylint: disable=protected-access
  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor):
    puo._ParameterUnscalingOptimizer.__init__(self, loss_scaling_factor)

    adam.Adam.__init__(self,
                       learning_rate=learning_rate,
                       beta_1=beta_1,
                       beta_2=beta_2,
                       epsilon=epsilon,
                       amsgrad=False,
                       name="Adam_HypLSFScale")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(_ParameterUnscalingAdam,
          self)._prepare_local(var_device, var_dtype, apply_state)

    self._prepare_local_lsf(var_device, var_dtype, apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    grad_fp32 = math_ops.cast(grad, tf.float32)
    sigma = coefficients["lsf"]

    # Compute v_new.
    v = self.get_slot(var, "v")
    v_fp32 = math_ops.cast(v, tf.float32)
    beta_2 = math_ops.cast(coefficients["beta_2_t"], tf.float32)
    beta_2_scaled = beta_2 * sigma * sigma
    v_new = v_fp32 * beta_2_scaled + (1 - beta_2) * grad_fp32 * grad_fp32

    # Compute m_new.
    m = self.get_slot(var, "m")
    m_fp32 = math_ops.cast(m, tf.float32)
    beta_1 = math_ops.cast(coefficients["beta_1_t"], tf.float32)
    beta_1_scaled = beta_1 * sigma
    m_new = m_fp32 * beta_1_scaled + (1 - beta_1) * grad_fp32

    # Compute var_new.
    beta_1_pwr = math_ops.cast(coefficients["beta_1_power"], tf.float32)
    beta_2_pwr = math_ops.cast(coefficients["beta_2_power"], tf.float32)
    eta = math_ops.cast(coefficients["lr_t"], tf.float32)
    eta_new = (eta * math_ops.sqrt(1 - beta_2_pwr) / (1 - beta_1_pwr))
    epsilon = math_ops.cast(coefficients["epsilon"], tf.float32) * sigma
    var_fp32 = math_ops.cast(var, tf.float32)
    var_new = var_fp32 - eta_new * m_new / (math_ops.sqrt(v_new) + epsilon)

    return control_flow_ops.group(
        state_ops.assign(var,
                         math_ops.cast(var_new, var.dtype),
                         use_locking=self._use_locking).op,
        state_ops.assign(m,
                         math_ops.cast(m_new, m.dtype),
                         use_locking=self._use_locking).op,
        state_ops.assign(v,
                         math_ops.cast(v_new, v.dtype),
                         use_locking=self._use_locking).op)


class ALSOptimizerAdam(puo._ParameterUnscalingALSOptimizer):  # pylint: disable=protected-access
  """An Adam optimizer that performs Automatic Loss Scaling,
  specifically handling moment updates.
  """
  def __init__(
      self,
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-7,
      initial_loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor,
      update_frequency=ALSDefaults.update_frequency,
      increase_factor=ALSDefaults.increase_factor,
      max_loss_scaling_factor=ALSDefaults.max_loss_scaling_factor,
      accumulate_statistics_over_update_period=ALSDefaults.
      accumulate_statistics_over_update_period,
      ratio_threshold=ALSDefaults.ratio_threshold,
      captured_grads_only=ALSDefaults.captured_grads_only,
      lpf_alpha=ALSDefaults.lpf_alpha,
      histogram_bin_edge=ALSDefaults.histogram_bin_edge,
      name="ALSOptimizerAdam"):
    """Construct a new automatic loss scaling Adam optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use, The
        learning rate.
        Defaults to 0.001.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates.
        Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use, The
        exponential decay rate for the 2nd moment estimates.
        Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
        Defaults to 1e-7.
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
      histogram_bin_edge: The magnitude at which gradients are considered
        to have overflowed.
        Defaults to 2^13.
      name: Optional name prefix for the operation created when applying
        gradients.
        Defaults to "ALSOptimizerAdam".
    """
    opt = _ParameterUnscalingAdam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        loss_scaling_factor=initial_loss_scaling_factor)

    super().__init__(opt,
                     initial_loss_scaling_factor=initial_loss_scaling_factor,
                     update_frequency=update_frequency,
                     increase_factor=increase_factor,
                     max_loss_scaling_factor=max_loss_scaling_factor,
                     accumulate_statistics_over_update_period=\
                      accumulate_statistics_over_update_period,
                     ratio_threshold=ratio_threshold,
                     captured_grads_only=captured_grads_only,
                     lpf_alpha=lpf_alpha,
                     histogram_bin_edge=histogram_bin_edge,
                     name=name)

  def get_config(self):
    """Returns the config of the `ALSOptimizerAdam` instance.
    """
    als_cfg = super().get_config()

    adam_type = als_cfg.pop("inner_optimizer_type")
    assert adam_type == _ParameterUnscalingAdam

    adam_cfg = als_cfg.pop("inner_optimizer_config")
    _ = adam_cfg.pop("amsgrad")
    _ = adam_cfg.pop("decay")
    _ = adam_cfg.pop("name")

    als_cfg.update(adam_cfg)
    return als_cfg

  @classmethod
  def from_config(cls, config, custom_objects=None):
    assert cls == ALSOptimizerAdam
    return ALSOptimizerAdam(**config)
