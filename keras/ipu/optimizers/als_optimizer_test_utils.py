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

import copy
import itertools

import numpy as np

from tensorflow.python.ops import init_ops

from keras import layers
from keras.ipu.optimizers import ALSOptimizer
from keras.ipu.optimizers import ALSGradientAccumulationOptimizer
from keras.ipu.layers import CaptureUpstreamGradients
from keras.ipu.layers import CaptureActivationGradients
from keras.optimizer_v2 import adam as adam_v2
from keras.optimizer_v2 import gradient_descent as gradient_descent_v2

NUM_BATCHES = 4
BATCH_SIZE = 64
INPUT_SHAPE = (NUM_BATCHES * BATCH_SIZE, 4)
OUTPUT_SHAPE = (NUM_BATCHES * BATCH_SIZE, 128)

DATA = np.ones(shape=INPUT_SHAPE, dtype=np.float32)
TARGETS = 2 * np.ones(shape=OUTPUT_SHAPE, dtype=np.float32)
TARGETS_HUGE = TARGETS * np.finfo(np.float16).max * 10


def dense_fn(dtype, wrapper_type=None, n=OUTPUT_SHAPE[1], init=1.0):
  wrapper = lambda x: x
  if wrapper_type:
    assert wrapper_type in (CaptureUpstreamGradients,
                            CaptureActivationGradients)

    wrapper = wrapper_type

  return wrapper(
      layers.Dense(n,
                   activation='relu',
                   dtype=dtype,
                   kernel_initializer=init_ops.constant_initializer(init)))


def generate_test_cases(no_ga=False):
  OPTIMIZER_CASES = [{
      'testcase_name': 'Adam',
      'optimizer_type': adam_v2.Adam,
      'optimizer_args': [0.01],
      'optimizer_kwargs': {}
  }, {
      'testcase_name': 'GradientDescent',
      'optimizer_type': gradient_descent_v2.SGD,
      'optimizer_args': [0.01],
      'optimizer_kwargs': {},
  }]

  ALS_OPTIMIZER_KWARG_CASES = [{
      'initial_loss_scaling_factor': 8,
      'update_frequency': 2,
      'increase_factor': 2,
  }, {
      'initial_loss_scaling_factor': 16,
      'update_frequency': 2,
      'increase_factor': 4
  }, {
      'initial_loss_scaling_factor': 32,
      'update_frequency': 2,
      'increase_factor': 8
  }, {
      'initial_loss_scaling_factor': 32,
      'update_frequency': 2,
      'increase_factor': 16
  }]

  WRAPPER_CASES = [(None, 'no_wrapper'),
                   (CaptureUpstreamGradients, 'CaptureUpstreamGradients'),
                   (CaptureActivationGradients, 'CaptureActivationGradients')]

  GRADIENT_ACCUMULATION_CASES = [1, 2] if not no_ga else [1]

  case_cartesian_product = itertools.product(OPTIMIZER_CASES,
                                             ALS_OPTIMIZER_KWARG_CASES,
                                             WRAPPER_CASES,
                                             GRADIENT_ACCUMULATION_CASES)

  cases = []
  n = 0
  for opt_case, als_case, wrapper_case, ga_case in case_cartesian_product:
    c = copy.deepcopy(opt_case)
    c['testcase_name'] += f"TestCase{n}_{wrapper_case[1]}_GA{ga_case}"
    c['als_kwargs'] = als_case
    c['wrapper_type'] = wrapper_case[0]
    c['ga_steps_per_replica'] = ga_case
    cases.append(c)
    n += 1
  return cases


def get_grads_and_vars(ga_steps_per_replica, variables, opt_wrapper, loss):
  if ga_steps_per_replica == 1:
    assert isinstance(opt_wrapper, ALSOptimizer)
  else:
    assert isinstance(opt_wrapper, ALSGradientAccumulationOptimizer)

  grads = opt_wrapper.get_gradients(loss, variables)
  return list(zip(grads, variables))
