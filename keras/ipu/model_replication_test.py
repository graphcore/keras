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
from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.ipu import ipu_strategy

import keras
from keras import keras_parameterized
from keras import testing_utils
from keras.ipu import ReplicatedMetricReductionMethod


def calculate_metrics_cpu(x, y, loss, metrics, num_replicas, reduction_mode):
  if not isinstance(metrics, list):
    metrics = [metrics]

  metrics = [loss] + metrics
  x = np.array(x)
  y = np.array(y)
  data_per_replica = x.shape[0] // num_replicas
  assert (x.shape[0] % num_replicas) == 0

  # Split the input data per replica [replica, ...].
  x = np.stack(np.split(x, data_per_replica), axis=1)
  y = np.stack(np.split(y, data_per_replica), axis=1)

  # [metric, replica, ...].
  results = [[
      metric(y[replica], x[replica]).numpy() for replica in range(num_replicas)
  ] for metric in metrics]

  if reduction_mode == ReplicatedMetricReductionMethod.NONE:
    return [result[-1] for result in results]
  if reduction_mode == ReplicatedMetricReductionMethod.LIST:
    return results
  if reduction_mode == ReplicatedMetricReductionMethod.SUM:
    return [np.sum(result) for result in results]
  if reduction_mode == ReplicatedMetricReductionMethod.MEAN:
    return [np.mean(result) for result in results]
  raise ValueError(f"Invalid reduction mode: {reduction_mode}")


class IPUModelReplicatedTest(keras_parameterized.TestCase):
  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = keras.Model(input_layer, x)
      m.compile('sgd', loss='mse', steps_per_execution=24)

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2)

      # The result is a Numpy array of predictions
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (96, 2))
      for i, r in enumerate(result):
        self.assertEqual(0, np.sum(r != result[i - 1]))

  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2Ga3DropsSamples(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = keras.Model(input_layer, x)

      m.compile('sgd', loss='mse', steps_per_execution=3)

      # Input data
      input_x = np.full([60, 32], 1.0, dtype=np.single)
      result = m.predict(input_x, batch_size=2)
      self.assertEqual(result.shape[0], 60)

  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testPredictWithNumpyDataBs2Replicas2Steps2Ga1(self):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32),
                                       dtype=np.single,
                                       batch_size=2)
      init = keras.initializers.Constant(0.1)
      x = keras.layers.Dense(4, name="layer0",
                             kernel_initializer=init)(input_layer)
      x = keras.layers.Dense(2, name="layer1", kernel_initializer=init)(x)
      m = keras.Model(input_layer, x)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse', steps_per_execution=2)

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=2, steps=2)

      # The result is a Numpy array of predictions
      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape, (4, 2))
      for i, r in enumerate(result):
        self.assertEqual(0, np.sum(r != result[i - 1]))

  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testPredictMultipleOutputReplicas2(self):
    def predict_input_fn():
      x1 = np.ones((64, 32), dtype=np.float32)
      x2 = np.ones((64, 32), dtype=np.float32)
      x3 = np.ones((64, 32), dtype=np.float32)

      return (x1, x2, x3)

    # Intentional skip from input to middle of model.
    def model_fn():
      input_1 = keras.Input(32)
      input_2 = keras.Input(32)
      input_3 = keras.Input(32)

      init = keras.initializers.Constant(1)

      dense_1 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_1)
      dense_2 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_2)

      cat = keras.layers.Concatenate()([dense_1, dense_2, input_3])

      dense_3 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)
      dense_4 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)

      return ((input_1, input_2, input_3), (dense_3, dense_4))

    # IPU Test.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'], steps_per_execution=2)

      predict_out = model.predict(predict_input_fn(), batch_size=4)

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # Comparison.
    self.assertEqual(np.shape(cpu_predict_out), np.shape(predict_out))

    for output in range(2):
      for cpu_predict, ipu_predict in zip(cpu_predict_out[output],
                                          predict_out[output]):
        np.testing.assert_almost_equal(cpu_predict, ipu_predict)

  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testPredictMultipleOutputDifferentShapesReplicas2(self):
    def predict_input_fn():
      x1 = np.ones((64, 32), dtype=np.float32)
      x2 = np.ones((64, 32), dtype=np.float32)
      x3 = np.ones((64, 32), dtype=np.float32)

      return (x1, x2, x3)

    # Intentional skip from input to middle of model.
    def model_fn():
      input_1 = keras.Input(32)
      input_2 = keras.Input(32)
      input_3 = keras.Input(32)

      init = keras.initializers.Constant(1)

      dense_1 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_1)
      dense_2 = keras.layers.Dense(16,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(input_2)

      cat = keras.layers.Concatenate()([dense_1, dense_2, input_3])

      dense_3 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)
      dense_4 = keras.layers.Dense(2,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)

      return ((input_1, input_2, input_3), (dense_3, dense_4))

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # IPU Test.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'], steps_per_execution=2)

      predict_out = model.predict(predict_input_fn(), batch_size=4)

    # Comparison.
    self.assertEqual(np.shape(cpu_predict_out[0]), np.shape(predict_out[0]))
    for cpu_predict, ipu_predict in zip(cpu_predict_out[0], predict_out[0]):
      np.testing.assert_almost_equal(cpu_predict, ipu_predict)

    self.assertEqual(np.shape(cpu_predict_out[1]), np.shape(predict_out[1]))
    for cpu_predict, ipu_predict in zip(cpu_predict_out[1], predict_out[1]):
      np.testing.assert_almost_equal(cpu_predict[0], ipu_predict[0])
      np.testing.assert_almost_equal(cpu_predict[1], ipu_predict[1])

  @parameterized.parameters(list(ReplicatedMetricReductionMethod))
  @tu.test_uses_ipus(num_ipus=2)
  @testing_utils.run_v2_only
  def testMetricReduction(self, reduction_method):
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()

    x = [0.0, 0.0, 0.0, 0.0]
    y = [1.0, 2.0, 3.0, 4.0]
    loss = tf.keras.metrics.MAE
    metrics = [tf.keras.metrics.MSE]
    expected_metric_values = calculate_metrics_cpu(x, y, loss, metrics, 2,
                                                   reduction_method)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(1, drop_remainder=True)

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      inputs = keras.Input(shape=1)
      outputs = tf.keras.layers.Dense(1,
                                      kernel_initializer='zeros',
                                      use_bias=False,
                                      trainable=False)(inputs)
      model = tf.keras.Model(inputs, outputs)
      model.build(input_shape=(1, 1))
      model.compile(loss=loss, metrics=metrics, steps_per_execution=4)
      model.set_replication_options(
          replicated_metric_reduction_method=reduction_method)
      actual_metric_values = model.evaluate(ds, steps=4)

    # Check the results of each metric.
    for i, _ in enumerate(metrics):
      self.assertAllEqual(actual_metric_values[i], expected_metric_values[i])


if __name__ == "__main__":
  tf.test.main()
