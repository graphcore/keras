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

from absl.testing import parameterized

import tensorflow.compat.v2 as tf
from tensorflow.compiler.plugin.poplar.driver import threestate_pb2

from keras.ipu.optimizers import ALSOptimizer
from keras.ipu.optimizers import ALSGradientAccumulationOptimizer
from keras.ipu.optimizers import GradientAccumulationOptimizer
from keras.optimizer_v2 import gradient_descent


def sgd_fn():
  return gradient_descent.SGD(0.01)


def als_opt_fn():
  return ALSOptimizer(sgd_fn())


def als_grad_accum_opt_fn():
  v = threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)
  return ALSGradientAccumulationOptimizer(
      als_opt_fn(),
      num_mini_batches=2,
      offload_weight_update_variables=v,
      replicated_optimizer_state_sharding=v)


def grad_accum_opt_fn():
  v = threestate_pb2.ThreeState.Name(threestate_pb2.THREESTATE_OFF)
  return GradientAccumulationOptimizer(sgd_fn(),
                                       num_mini_batches=2,
                                       offload_weight_update_variables=v,
                                       replicated_optimizer_state_sharding=v)


TEST_CASES = [{
    'testcase_name': 'ALSOptimizer',
    'opt_fn': als_opt_fn,
}, {
    'testcase_name': 'ALSGradientAccumulationOptimizer',
    'opt_fn': als_grad_accum_opt_fn,
}, {
    'testcase_name': 'GradientAccumulationOptimizer',
    'opt_fn': grad_accum_opt_fn,
}]


class ALSOptimizerPipelineTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(*TEST_CASES)
  def testCreateFromConfig(self, opt_fn):
    opt_1 = opt_fn()
    opt_1_config = opt_1.get_config()

    opt_2_config = opt_1_config.copy()
    opt_2_config['name'] += "_copy"

    opt_2 = opt_1.__class__.from_config(opt_2_config)
    self.assertEqual(opt_2.get_config(), opt_2_config)

  @parameterized.named_parameters(*TEST_CASES)
  def testWeightsPropertyRead(self, opt_fn):
    opt = opt_fn()
    w = opt.weights
    opt.set_weights(2 * w)
    self.assertEqual(opt.weights, 2 * w)

  @parameterized.named_parameters(*TEST_CASES)
  def testWeightsPropertyWrite(self, opt_fn):
    opt = opt_fn()
    with self.assertRaisesRegex(AttributeError, "can't set attribute"):
      opt.weights = 1

  @parameterized.named_parameters(*TEST_CASES)
  def testVariablesMethod(self, opt_fn):
    opt = opt_fn()
    self.assertEqual(opt.get_weights(), opt.variables())

  @parameterized.named_parameters(*TEST_CASES)
  def testGetSetWeights(self, opt_fn):
    opt_1 = opt_fn()
    opt_2 = opt_fn()

    opt_2.set_weights([w * 2 for w in opt_1.get_weights()])

    for a, b in zip(opt_1.get_weights(), opt_2.get_weights()):
      self.assertEqual(b, 2 * a)


if __name__ == "__main__":
  tf.test.main()
