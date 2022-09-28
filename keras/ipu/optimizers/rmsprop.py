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
RMSProp specialization for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from keras.ipu.optimizers.als_optimizer import ALSDefaults
from keras.ipu.optimizers import parameter_unscaling_optimizer as puo
from keras.optimizer_v2 import rmsprop


class _ParameterUnscalingRMSProp(
    puo._ParameterUnscalingOptimizer,  # pylint: disable=protected-access
    rmsprop.RMSProp):
  """An Adam optimizer that performs Automatic Loss Scaling,
  specifically handling moment updates.
  """
  def __init__(self,
               learning_rate=0.001,
               rho=0.9,
               momentum=0.0,
               epsilon=1e-7,
               centered=False,
               loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor):
    puo._ParameterUnscalingOptimizer.__init__(self, loss_scaling_factor)

    rmsprop.RMSProp.__init__(self,
                             learning_rate=learning_rate,
                             rho=rho,
                             momentum=momentum,
                             epsilon=epsilon,
                             centered=centered,
                             name="RMSProp_HypLSFScale")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(_ParameterUnscalingRMSProp,
          self)._prepare_local(var_device, var_dtype, apply_state)

    self._prepare_local_lsf(var_device, var_dtype, apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    var_fp32 = math_ops.cast(var, tf.float32)
    grad_fp32 = math_ops.cast(grad, tf.float32)

    beta = self._get_hyper("decay", tf.float32)
    eta = math_ops.cast(coefficients["lr_t"], tf.float32)

    sigma = coefficients["lsf"]
    sigma_prev = coefficients["prev_lsf"]

    epsilon = math_ops.cast(coefficients["epsilon"],
                            tf.float32) * sigma * sigma

    m = self.get_slot(var, "momentum") if self._momentum else 0
    m_fp32 = math_ops.cast(m, tf.float32)
    mu = math_ops.cast(coefficients["momentum"], tf.float32)

    v = self.get_slot(var, "rms")
    v_fp32 = math_ops.cast(v, tf.float32)
    beta_sq_scaled = ((sigma * sigma) / (sigma_prev * sigma_prev)) * beta
    v_new = beta_sq_scaled * v_fp32 + (1.0 - beta) * grad_fp32 * grad_fp32

    if self._momentum and self.centered:
      h = self.get_slot(var, "mg")
      h_fp32 = math_ops.cast(h, tf.float32)
      beta_scaled = beta * sigma / sigma_prev
      h_new = beta_scaled * h_fp32 + (1.0 - beta) * grad_fp32

      denom = math_ops.sqrt(v_new + epsilon + h_new * h_new)
      m_new = mu * m_fp32 + grad_fp32 / denom

      var_new = var_fp32 - eta * m_new

      return control_flow_ops.group(
          state_ops.assign(h,
                           math_ops.cast(h_new, h.dtype),
                           use_locking=self._use_locking).op,
          state_ops.assign(v,
                           math_ops.cast(v_new, v.dtype),
                           use_locking=self._use_locking).op,
          state_ops.assign(var,
                           math_ops.cast(var_new, var.dtype),
                           use_locking=self._use_locking).op,
          state_ops.assign(m,
                           math_ops.cast(m_new, m.dtype),
                           use_locking=self._use_locking).op)

    m_new = m_fp32 * mu + grad_fp32 / math_ops.sqrt(v_new + epsilon)
    var_new = var_fp32 - eta * m_new

    updates = [
        state_ops.assign(var,
                         math_ops.cast(var_new, var.dtype),
                         use_locking=self._use_locking).op
    ]

    if self._momentum:
      updates.append(
          state_ops.assign(m,
                           math_ops.cast(m_new, m.dtype),
                           use_locking=self._use_locking).op)

    return control_flow_ops.group(*updates)


class ALSOptimizerRMSProp(puo._ParameterUnscalingALSOptimizer):  # pylint: disable=protected-access
  """An RMSProp optimizer that performs Automatic Loss Scaling,
  specifically handling moment updates.
  """
  def __init__(
      self,
      learning_rate=0.001,
      rho=0.9,
      momentum=0.0,
      epsilon=1e-7,
      centered=False,
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
      name="ALSOptimizerRMSProp"):
    """Construct a new automatic loss scaling RMSProp optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate.
        Defaults to 0.001.
      rho: Discounting factor for the history/coming gradient.
        Defaults to 0.9.
      momentum: A scalar or a scalar `Tensor`.
        Defaults to 0.0.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
          Defaults to 1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory.
        Defaults to `False`.
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
        Defaults to "ALSOptimizerRMSProp".
    """
    opt = _ParameterUnscalingRMSProp(
        learning_rate=learning_rate,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered,
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

    rmsprop_type = als_cfg.pop("inner_optimizer_type")
    assert rmsprop_type == _ParameterUnscalingRMSProp

    rmsprop_cfg = als_cfg.pop("inner_optimizer_config")
    _ = rmsprop_cfg.pop("decay")
    _ = rmsprop_cfg.pop("name")

    als_cfg.update(rmsprop_cfg)
    return als_cfg

  @classmethod
  def from_config(cls, config, custom_objects=None):
    assert cls == ALSOptimizerRMSProp
    return ALSOptimizerRMSProp(**config)
