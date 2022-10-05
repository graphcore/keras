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
# ==============================================================================
import tempfile
import os
from absl.testing import parameterized

import tensorflow.compat.v2 as tf
import numpy as np

from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga

import keras
from keras import testing_utils
from keras.datasets import mnist
from keras.mixed_precision import policy
from keras.optimizer_v2 import gradient_descent as gradient_descent_v2


def get_mnist_dataset(batch_size):
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  x_train = x_train.astype('float32')
  y_train = y_train.astype('int32')

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).batch(batch_size, drop_remainder=True).repeat()

  return train_ds.repeat()


def simple_sequential_model():
  return keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')
  ])


def simple_functional_model():
  d = keras.layers.Input((28, 28))
  x = keras.layers.Flatten()(d)
  x = keras.layers.Dense(10,
                         activation='softmax',
                         kernel_initializer=keras.initializers.Constant(0.5),
                         bias_initializer='zeros')(x)
  return keras.Model(d, x)


class SimpleSubclassedModel(keras.engine.training.Model):
  def __init__(self, hidden_units):
    super().__init__()
    self.hidden_units = hidden_units
    self.dense_layers = [
        keras.layers.Dense(u,
                           activation='softmax',
                           kernel_initializer=keras.initializers.Constant(0.5),
                           bias_initializer='zeros') for u in hidden_units
    ]

  def get_config(self):
    config = super().get_config()
    config["hidden_units"] = self.hidden_units
    return config

  def call(self, inputs):  # pylint: disable=arguments-differ
    x = keras.layers.Flatten()(inputs)
    for layer in self.dense_layers:
      x = layer(x)
    return x


class KerasGradientAccumulationTest(tf.test.TestCase, parameterized.TestCase):
  TESTCASES = testing_utils.generate_combinations_with_testcase_name(
      model_fn=[
          simple_sequential_model, simple_functional_model,
          lambda: SimpleSubclassedModel([10, 20, 10])
      ],
      replication_factor=[1, 2],
      reduction_method=[
          ga.GradientAccumulationReductionMethod.MEAN,
          ga.GradientAccumulationReductionMethod.RUNNING_MEAN
      ],
      use_v2_gao=[False, True])

  MINIMIZE_TESTCASES = [{
      'testcase_name': 'mixed_precision',
      'mixed': True
  }, {
      'testcase_name': 'non_mixed_precision',
      'mixed': False
  }]

  @parameterized.named_parameters(*TESTCASES)
  @testing_utils.run_v2_only
  def testModels(self, model_fn, replication_factor, reduction_method,
                 use_v2_gao):
    tu.skip_if_not_enough_ipus(self, replication_factor)

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = replication_factor
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    batch_size = 12
    gradient_accumulation_steps = 16
    gradient_accumulation_steps_per_replica = (gradient_accumulation_steps //
                                               replication_factor)
    steps_per_epoch = 64
    epochs = 2

    def opt_fn(lr=0.1):
      if not use_v2_gao:
        return gradient_descent.GradientDescentOptimizer(lr)
      return gradient_descent_v2.SGD(lr)

    # Run on CPU - simulate gradient accumulation by just using a bigger batch
    # size but less steps per epoch.
    m = model_fn()
    optimizer = opt_fn()
    m.compile(optimizer, loss=keras.losses.SparseCategoricalCrossentropy())
    m.fit(get_mnist_dataset(batch_size * gradient_accumulation_steps),
          steps_per_epoch=steps_per_epoch // gradient_accumulation_steps,
          epochs=epochs)
    cpu_weights = m.weights

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = model_fn()

      optimizer = opt_fn()
      if hasattr(optimizer, '_learning_rate'):  # v1
        optimizer._learning_rate /= replication_factor
      else:  # v2
        optimizer.lr.assign(optimizer.lr / replication_factor)

      m.compile(optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                steps_per_execution=gradient_accumulation_steps * 2)
      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=
          gradient_accumulation_steps_per_replica,
          gradient_accumulation_reduction_method=reduction_method,
          use_v2_gradient_accumulation_optimizer=use_v2_gao)
      m.fit(get_mnist_dataset(batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs)
      ipu_weights = m.weights

      self.assertAllClose(cpu_weights, ipu_weights)

      # Test that the extra properties are restored.
      with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "model")
        m.save(save_path)
        m = keras.models.load_model(save_path)
        self.assertEqual(
            m._gradient_accumulation_steps_per_replica,  # pylint: disable=protected-access
            gradient_accumulation_steps_per_replica)
        self.assertEqual(
            m._gradient_accumulation_reduction_method,  # pylint: disable=protected-access
            reduction_method)
        self.assertFalse(m._gradient_accumulation_optimizer_kwargs)  # pylint: disable=protected-access

  @parameterized.named_parameters(*MINIMIZE_TESTCASES)
  @testing_utils.run_v2_only
  def testNoOverrideMinimizeWithGradientAccumulation(self, mixed):
    class BadSGD(gradient_descent_v2.SGD):
      def __init__(self, lr):  # pylint: disable=useless-super-delegation
        super().__init__(lr)

      def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):  # pylint: disable=unused-argument
        return 0

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    if mixed:
      mp_policy = policy.Policy('mixed_float16')
      policy.set_policy(mp_policy)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      input_layer = keras.layers.Input(1)
      x = keras.layers.Dense(1)(input_layer)
      l = keras.losses.MeanSquaredError(reduction="sum")

      m = keras.Model(inputs=input_layer, outputs=x)
      m.compile(loss=l, optimizer=BadSGD(0.1), steps_per_execution=2)

      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=2)

      with self.assertRaisesRegex(ValueError,
                                  "must not override OptimizerV2.minimize"):
        data = [np.ones((64, 1), dtype=np.float16)] * 2
        _ = m.fit(*data, epochs=1, verbose=False)


if __name__ == '__main__':
  tf.test.main()
