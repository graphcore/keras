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
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.distributed import popdist_strategy

import keras
from keras import testing_utils
from keras.mixed_precision import policy


class MixedPrecissionTest(tf.test.TestCase):
  def _test_run_model(self):

    mp_policy = policy.Policy('mixed_float16')
    policy.set_policy(mp_policy)

    input_layer = keras.layers.Input(shape=(32), dtype=np.single, batch_size=2)
    init = keras.initializers.Constant(0.1)
    x = keras.layers.Dense(4, name="layer0",
                           kernel_initializer=init)(input_layer)
    x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)

    m = keras.Model(input_layer, x)
    m.compile('sgd', loss='mse', steps_per_execution=4)

    input_x = np.full([60, 32], 1.0, dtype=np.single)
    input_y = np.full([60], 1.0, dtype=np.single)
    m.fit(input_x, input_y, batch_size=2, steps_per_epoch=1, epochs=1)

  @testing_utils.run_v2_only
  def testIPUStrategy(self):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      self._test_run_model()

  @testing_utils.run_v2_only
  def testPopDistStrategy(self):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    strategy = popdist_strategy.PopDistStrategy()
    with strategy.scope():
      self._test_run_model()


if __name__ == "__main__":
  tf.test.main()
