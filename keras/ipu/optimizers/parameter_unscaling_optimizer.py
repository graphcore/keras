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
Optimizer specializations for automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from keras.ipu.optimizers.als_optimizer import ALSOptimizer
from keras.ipu.optimizers.als_optimizer import ALSDefaults
from keras.ipu.optimizers.als_optimizer import _is_power_of_two


class _ParameterUnscalingOptimizer:
  def __init__(self, loss_scale_factor):
    if not loss_scale_factor >= 1:
      raise ValueError(
          "loss_scale_factor must be greater than or equal to one.")

    if not _is_power_of_two(loss_scale_factor):
      raise ValueError("loss_scale_factor must be a power of two.")

    self._lsf = math_ops.cast(loss_scale_factor, tf.float32)
    self._prev_lsf = math_ops.cast(loss_scale_factor, tf.float32)

  def _prepare_local_lsf(self, var_device, var_dtype, apply_state):
    apply_state[(var_device, var_dtype)].update({
        "prev_lsf":
        ops.convert_to_tensor(self._prev_lsf),
        "lsf":
        ops.convert_to_tensor(self._lsf)
    })


class _ParameterUnscalingALSOptimizer(ALSOptimizer):
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
      histogram_bin_edge=ALSDefaults.histogram_bin_edge,
      replication_factor=ALSDefaults.replication_factor,
      name="_ParameterUnscalingALSOptimizer"):
    if not isinstance(opt, _ParameterUnscalingOptimizer):
      raise ValueError(
          "opt must be an OptimizerV2 specialization derived from "
          "_ParameterUnscalingOptimizer")

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
                     replication_factor=replication_factor,
                     name=name)
