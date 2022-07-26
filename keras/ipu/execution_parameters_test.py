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
from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.ipu import ipu_strategy

import keras
from keras import testing_utils


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = tf.constant(x_val, shape=[32])
  constant_l = tf.constant(y_val, shape=[2])

  ds = tf.data.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def simple_sequential_model():
  return keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(4),
      keras.layers.Dense(2),
  ])


def simple_functional_model():
  d = keras.layers.Input(32)
  x = keras.layers.Flatten()(d)
  x = keras.layers.Dense(4)(x)
  x = keras.layers.Dense(2)(x)
  return keras.Model(d, x)


class KerasModelExecutionParametersTest(tf.test.TestCase,
                                        parameterized.TestCase):
  @parameterized.parameters([simple_sequential_model, simple_functional_model])
  @tu.test_uses_ipus(num_ipus=1)
  @testing_utils.run_v2_only
  def testGradientAccumulation(self, model_fn):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = model_fn()
      m.compile('sgd', loss='mse', steps_per_execution=8)

      with self.assertRaisesRegex(
          RuntimeError,
          r"The model has been configured to use gradient accumulation for "
          r"training, however the current `steps_per_execution` value \(set to "
          r"8\) is not divisible by `gradient_accumulation_steps_per_replica` "
          r"\(3\)"):
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=3)
        m.fit(test_dataset(length=8))

  @parameterized.parameters([simple_sequential_model, simple_functional_model])
  @tu.test_uses_ipus(num_ipus=8)
  @testing_utils.run_v2_only
  def testGradientAccumulationReplicated(self, model_fn):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 8
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = model_fn()
      m.compile('sgd', loss='mse', steps_per_execution=8)

      with self.assertRaisesRegex(
          RuntimeError,
          r"The model has been configured to use gradient accumulation for "
          r"training, however the current `steps_per_execution` value \(set to "
          r"8\) is not divisible by `gradient_accumulation_steps_per_replica` "
          r"\(3\)"):
        m.set_gradient_accumulation_options(
            gradient_accumulation_steps_per_replica=3)
        m.fit(test_dataset(length=64))


if __name__ == "__main__":
  tf.test.main()
