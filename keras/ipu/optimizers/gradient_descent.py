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
SGD specialization for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from keras.ipu.optimizers.als_optimizer import ALSDefaults
from keras.ipu.optimizers import parameter_unscaling_optimizer as puo
from keras.optimizer_v2 import gradient_descent


class _ParameterUnscalingSGD(
    puo._ParameterUnscalingOptimizer,  # pylint: disable=protected-access
    gradient_descent.SGD):
  """A custom SGD implementation that handles scaling of
  fp16 hyperparameters.
  """
  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor):
    puo._ParameterUnscalingOptimizer.__init__(self, loss_scaling_factor)

    gradient_descent.SGD.__init__(self,
                                  learning_rate=learning_rate,
                                  momentum=momentum,
                                  nesterov=False,
                                  name="SGD_HypLSFScale")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(_ParameterUnscalingSGD, self)._prepare_local(var_device, var_dtype,
                                                       apply_state)

    self._prepare_local_lsf(var_device, var_dtype, apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    grad_fp32 = math_ops.cast(grad, tf.float32)
    var_fp32 = math_ops.cast(var, tf.float32)

    eta = math_ops.cast(coefficients["lr_t"], tf.float32)
    sigma = coefficients["lsf"]

    if self._momentum:
      alpha = math_ops.cast(coefficients["momentum"], tf.float32)
      sigma_prev = coefficients["prev_lsf"]

      # Compute updated momentum.
      m = self.get_slot(var, "momentum")
      m_fp32 = math_ops.cast(m, tf.float32)
      c = alpha * sigma / sigma_prev
      m_new = c * m_fp32 + grad_fp32

      # Compute updated variable.
      var_new = var_fp32 - (eta / sigma) * m_new

      return control_flow_ops.group(
          state_ops.assign(m,
                           math_ops.cast(m_new, m.dtype),
                           use_locking=self._use_locking).op,
          state_ops.assign(var,
                           math_ops.cast(var_new, var.dtype),
                           use_locking=self._use_locking).op)

    var_new = var_fp32 - (eta / sigma) * grad_fp32
    return state_ops.assign(var, math_ops.cast(var_new, var.dtype))


class ALSOptimizerSGD(puo._ParameterUnscalingALSOptimizer):  # pylint: disable=protected-access
  """An SGD optimizer that performs ALS.
  """
  def __init__(
      self,
      learning_rate=0.01,
      momentum=0.0,
      initial_loss_scaling_factor=ALSDefaults.initial_loss_scaling_factor,
      update_frequency=ALSDefaults.update_frequency,
      increase_factor=ALSDefaults.increase_factor,
      max_loss_scaling_factor=ALSDefaults.max_loss_scaling_factor,
      accumulate_statistics_over_update_period=ALSDefaults.
      accumulate_statistics_over_update_period,
      ratio_threshold=ALSDefaults.ratio_threshold,
      captured_grads_only=ALSDefaults.captured_grads_only,
      lpf_alpha=ALSDefaults.lpf_alpha,
      name="ALSOptimizerSGD"):
    opt = _ParameterUnscalingSGD(
        learning_rate=learning_rate,
        momentum=momentum,
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
                     name=name)

  def get_config(self):
    """Returns the config of the `ALSOptimizerAdam` instance.
    """
    als_cfg = super().get_config()

    sgd_type = als_cfg.pop("inner_optimizer_type")
    assert sgd_type == _ParameterUnscalingSGD

    sgd_cfg = als_cfg.pop("inner_optimizer_config")
    _ = sgd_cfg.pop("nesterov")
    _ = sgd_cfg.pop("decay")
    _ = sgd_cfg.pop("name")

    als_cfg.update(sgd_cfg)
    return als_cfg

  @classmethod
  def from_config(cls, config, custom_objects=None):
    assert cls == ALSOptimizerSGD
    return ALSOptimizerSGD(**config)
