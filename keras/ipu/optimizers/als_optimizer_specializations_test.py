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

import numpy as np
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ipu_strategy import IPUStrategyV1

from keras.ipu.optimizers import ALSOptimizerAdam
from keras.ipu.optimizers import ALSOptimizerRMSProp
from keras.ipu.optimizers import ALSOptimizerSGD

import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import np_utils


def ds_fn(dtype=tf.float32):
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0
  x_train = np.expand_dims(x_train, -1)

  y_train = np_utils.to_categorical(y_train, 10)

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, dtype), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


def model_fn(opt):
  init = keras.initializers.Constant(0.5)

  input_layer = keras.Input(shape=(28, 28, 1))
  x = layers.Conv2D(32,
                    kernel_size=(3, 3),
                    activation="relu",
                    dtype=tf.float16,
                    kernel_initializer=init)(input_layer)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Conv2D(64,
                    kernel_size=(3, 3),
                    activation="relu",
                    dtype=tf.float16,
                    kernel_initializer=init)(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(10,
                   activation="softmax",
                   dtype=tf.float32,
                   kernel_initializer=init)(x)

  m = keras.Model(input_layer, x)
  m.compile(optimizer=opt, loss="categorical_crossentropy")
  return m


CASES = [
    lambda: ALSOptimizerAdam(update_frequency=64),  # pylint: disable=unnecessary-lambda
    lambda: ALSOptimizerRMSProp(update_frequency=64),  # pylint: disable=unnecessary-lambda
    lambda: ALSOptimizerRMSProp(
        update_frequency=64, centered=True, momentum=0.01),  # pylint: disable=unnecessary-lambda
    lambda: ALSOptimizerRMSProp(update_frequency=64, momentum=0.01),  # pylint: disable=unnecessary-lambda
    lambda: ALSOptimizerSGD(update_frequency=64),  # pylint: disable=unnecessary-lambda
    lambda: ALSOptimizerSGD(update_frequency=64, momentum=0.01)  # pylint: disable=unnecessary-lambda
]


class ALSOptimizerSpecializationsTest(tf.test.TestCase,
                                      parameterized.TestCase):
  @parameterized.parameters(CASES)
  def testConvergence(self, opt_fn):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    strategy = IPUStrategyV1()
    with strategy.scope():
      m = model_fn(opt_fn())
      history = m.fit(ds_fn(), epochs=3, verbose=False, steps_per_epoch=128)

    last_loss = float('inf')
    for l in history.history['loss']:
      self.assertTrue(np.isfinite(l))
      self.assertLess(l, last_loss)
      last_loss = l


if __name__ == "__main__":
  tf.test.main()
