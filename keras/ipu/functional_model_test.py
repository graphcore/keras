# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for IPU Keras Model"""

import numpy as np
import pva

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python import ipu
from tensorflow.python.training import gradient_descent

import keras
from keras.engine import base_layer_utils
from keras import testing_utils


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = tf.constant(x_val, shape=[32])
  constant_l = tf.constant(y_val, shape=[2])

  ds = tf.data.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def test_inference_dataset(length=None, batch_size=1, x_val=1.0):
  constant_d = tf.constant(x_val, shape=[32])

  ds = tf.data.Dataset.from_tensors(constant_d)
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def test_dataset_two_input_output(length=None,
                                  batch_size=1,
                                  x_val=1.0,
                                  y_val=0.2,
                                  input_names=None,
                                  target_names=None):
  ds = tf.data.Dataset.from_tensors(({
      input_names[0]:
      tf.constant(x_val, shape=[32]),
      input_names[1]:
      tf.constant(x_val, shape=[16])
  }, {
      target_names[0]:
      tf.constant(y_val, shape=[2]),
      target_names[1]:
      tf.constant(y_val, shape=[1])
  }))
  ds = ds.repeat(length)
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


def test_dataset_two_input_output_np(length=96,
                                     x_val=1.0,
                                     y_val=0.2,
                                     input_names=None,
                                     target_names=None):
  inputs = {
      input_names[0]: np.ones((length, 32), dtype=np.float32) * x_val,
      input_names[1]: np.ones((length, 16), dtype=np.float32) * x_val
  }
  targets = {
      target_names[0]: np.ones((length, 2), dtype=np.float32) * y_val,
      target_names[1]: np.ones((length, 1), dtype=np.float32) * y_val
  }

  return (inputs, targets)


def simple_model(x, layer_sizes, w=None):
  assert layer_sizes

  init = 'glorot_uniform'
  if w:
    assert w > 0
    init = keras.initializers.Constant(w)

  y = keras.layers.Dense(layer_sizes[0],
                         activation=keras.activations.relu,
                         kernel_initializer=init)(x)

  for n in layer_sizes[1:]:
    y = keras.layers.Dense(n,
                           activation=keras.activations.relu,
                           kernel_initializer=init)(y)
  return y


class BatchCallbackCounter(keras.callbacks.Callback):
  def __init__(self):
    super().__init__()
    self._count = 0
    self._logs = []

  def on_batch_end(self, batch, logs=None):
    del batch
    self._logs.append(logs)
    self._count = self._count + 1

  def count(self):
    return self._count

  def logs(self):
    return self._logs


class IPUModelModelTest(tf.test.TestCase):
  @testing_utils.run_v2_only
  def testModelCreation(self):
    # Simple single input, single output model.
    input_layer = keras.layers.Input(shape=(2))
    x = simple_model(input_layer, [2, 4])
    y = keras.layers.Activation(keras.activations.relu)(x)
    m = keras.Model(inputs=input_layer, outputs=y)

    # Verify dims.
    self.assertEqual(
        m._input_layers[0].get_output_at(0).get_shape().as_list(),  # pylint: disable=protected-access
        [None, 2])
    self.assertEqual(
        m._output_layers[0].get_output_at(0).get_shape().as_list(), [None, 4])  # pylint: disable=protected-access

  @testing_utils.run_v2_only
  def testModelCreationMultipleInput(self):
    # Simple two input, one output model.
    input_layer = keras.layers.Input(shape=(2))
    input_layer_two = keras.layers.Input(shape=(2))

    x = simple_model(input_layer, [2, 4])
    xx = simple_model(input_layer_two, [2, 4])

    x_con = keras.layers.concatenate([x, xx])
    y = keras.layers.Activation(keras.activations.relu)(x_con)

    m = keras.Model(inputs=[input_layer, input_layer_two], outputs=y)

    # Verify dims.
    self.assertEqual(len(m._input_layers), 2)  # pylint: disable=protected-access
    for d in m._input_layers:  # pylint: disable=protected-access
      self.assertEqual(d.get_output_at(0).get_shape().as_list(), [None, 2])
    self.assertEqual(
        m._output_layers[0].get_output_at(0).get_shape().as_list(), [None, 8])  # pylint: disable=protected-access

  @testing_utils.run_v2_only
  def testModelCreationMultipleOutput(self):
    # Simple one input, two output model.
    input_layer = keras.layers.Input(shape=(2))
    x = simple_model(input_layer, [2, 4])

    y = keras.layers.Activation(keras.activations.tanh)(x)
    yy = keras.layers.Activation(keras.activations.sigmoid)(x)

    m = keras.Model(inputs=input_layer, outputs=[y, yy])

    self.assertEqual(
        m._input_layers[0].get_output_at(0).get_shape().as_list(),  # pylint: disable=protected-access
        [None, 2])
    self.assertEqual(len(m._output_layers), 2)  # pylint: disable=protected-access
    for d in m._output_layers:  # pylint: disable=protected-access
      self.assertEqual(d.get_output_at(0).get_shape().as_list(), [None, 4])

  @testing_utils.run_v2_only
  def testMustCallCompileFit(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8])
      m = keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.fit(test_dataset())

  @testing_utils.run_v2_only
  def testMustCallCompileEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8])
      m = keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(
          RuntimeError, "You must compile your model before training/testing"):
        m.evaluate(test_dataset())

  @testing_utils.run_v2_only
  def testNeedTupleDatasetFit(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8])
      m = keras.Model(inputs=input_layer, outputs=x)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(ValueError,
                                  r"When providing an infinite dataset"):
        m.fit(test_inference_dataset())

  @testing_utils.run_v2_only
  def testNeedTupleDatasetEvaluate(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8])
      m = keras.Model(inputs=input_layer, outputs=x)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(ValueError,
                                  r"When providing an infinite dataset"):
        m.evaluate(test_inference_dataset())

  # @testing_utils.run_v2_only
  def testNeedNonTupleDatasetPredict(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8])
      m = keras.Model(inputs=input_layer, outputs=x)

      with self.assertRaisesRegex(ValueError,
                                  r"When providing an infinite dataset"):
        m.predict(test_dataset())

  @testing_utils.run_v2_only
  def testUnlimitedDatasetHasNoStepsPerEpoch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)
      m.compile('sgd', loss='mse')

      with self.assertRaisesRegex(ValueError,
                                  r"When providing an infinite dataset"):
        m.fit(test_dataset(), epochs=2)

  @testing_utils.run_v2_only
  def testResultsOneEpochWithTfOptimizerNoAccumulation_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = gradient_descent.GradientDescentOptimizer(0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=96))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)

    # Run the CPU equivalent.
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [8, 8, 2], w=0.4)
    m_cpu = keras.Model(inputs=input_layer, outputs=x)
    opt_cpu = gradient_descent.GradientDescentOptimizer(0.001)
    m_cpu.compile(opt_cpu, loss='mse')

    # Fit the weights to the dataset
    cpu_loss = m_cpu.fit(test_dataset(length=96)).history['loss'][0]

    # history['loss'] is one loss value per epoch (of which there is 1)
    ipu_loss = history.history['loss'][0]

    self.assertAllClose(ipu_loss, cpu_loss)

  @testing_utils.run_v2_only
  def testFitWithTensorData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = tf.constant(1.0, shape=[72, 32])
      input_y = tf.constant(0.2, shape=[72, 2])

      # Fit the weights to the dataset
      history = m.fit(input_x, input_y, batch_size=1)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @testing_utils.run_v2_only
  def testFitWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      history = m.fit(input_x, input_y, batch_size=1)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @testing_utils.run_v2_only
  def testEvalWithNumpyData(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      input_x = np.full([72, 32], 1.0, dtype=np.single)
      input_y = np.full([72, 2], 0.2, dtype=np.single)

      # Fit the weights to the dataset
      result = m.evaluate(input_x, input_y, batch_size=1)
      self.assertEqual(type(result), float)

  @testing_utils.run_v2_only
  def testPredictWithNumpyDataBs1(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse', steps_per_execution=8)

      # Input data
      input_x = np.full([96, 32], 1.0, dtype=np.single)

      # Generate predictions
      result = m.predict(input_x, batch_size=1)

      self.assertEqual(type(result), np.ndarray)
      self.assertEqual(result.shape[0], 96)
      for i, r in enumerate(result):
        self.assertAllEqual(r, result[i - 1])

    # Compare with CPU
    m = keras.Model(inputs=input_layer, outputs=x)
    cpu_result = m.predict(input_x, batch_size=1)

    self.assertEqual(cpu_result.shape, result.shape)

  @testing_utils.run_v2_only
  def testFitHistoryWithKerasOptimizer(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72))

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 1)
      self.assertEqual(type(history.history['loss'][0]), float)

  @testing_utils.run_v2_only
  def testFitHistoryTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(length=72), epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)

  @testing_utils.run_v2_only
  def testFitHistoryStepsPerExecution(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse', steps_per_execution=2)

      # Check that the callback is called for every two steps due to
      # `steps_per_execution`.
      cb = BatchCallbackCounter()
      m.fit(test_dataset(length=96), callbacks=[cb])
      self.assertEqual(cb.count(), 48)

  @testing_utils.run_v2_only
  def testFitTwice(self):
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg, output_execution_profile=True)
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      ds = test_dataset()
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [16, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=1)
      l = history.history['loss'][0]

      # # Record weights
      w_1 = [w.numpy() for w in m.weights]

      # Fit the weights to the dataset
      history = m.fit(ds, steps_per_epoch=1)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_2 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w1, w2 in zip(w_1, w_2):
        self.assertFalse(np.all(w1 == w2))

      # Should have compiled the graph once, and executed twice.
      self.assert_num_reports(report_helper, 1)
      report = pva.openReport(report_helper.find_report())
      self.assert_number_of_executions(report, 2)
      report_helper.clear_reports()

      # Fit the weights with a new dataset
      history = m.fit(test_dataset(), steps_per_epoch=1)

      # Loss should be different after second training.
      self.assertTrue(l > history.history['loss'][0])

      w_3 = [w.numpy() for w in m.weights]

      # Weights should be different too.
      for w2, w3 in zip(w_2, w_3):
        self.assertFalse(np.all(w2 == w3))

      # Don't need to compile the graph again.
      self.assert_num_reports(report_helper, 0)

  @testing_utils.run_v2_only
  def testFitHistoryStepsPerEpochTwoEpochs(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=144, epochs=2)

      # Should be only a loss stored in the history, and it should contain
      # only the single epochs value
      self.assertEqual(list(history.history.keys()), ['loss'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(type(history.history['loss'][1]), float)

  @testing_utils.run_v2_only
  def testFitWithLearningRateDecay(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001,
                                                    decay=0.1)
      m.compile(opt, loss='mse', steps_per_execution=8)

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @testing_utils.run_v2_only
  def testFitWithExponentialDecayLearningRateSchedule(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      lrs = keras.optimizer_v2.learning_rate_schedule.ExponentialDecay(
          0.001, 4, 0.1, staircase=True)
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @testing_utils.run_v2_only
  def testFitWithPiecewiseConstantDecayLearningRateSchedule(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    tu.enable_ipu_events(cfg)
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    report_json = tu.ReportJSON(self, eager_mode=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Clear old reports
      report_json.reset()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      lrs = keras.optimizer_v2.learning_rate_schedule.PiecewiseConstantDecay(
          boundaries=[8, 16], values=[0.001, 0.0005, 0.0001])
      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=lrs)
      m.compile(opt, loss='mse')

      # Fit the weights to the dataset
      m.fit(test_dataset(length=72), epochs=4)

      # Ensure that we are only downloading the weights at the end of each
      # epoch.
      report_json.parse_log()
      report_json.assert_num_host_to_device_transfer_events(4)

  @testing_utils.run_v2_only
  def testFitWithMetrics(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2])
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.0001)
      m.compile(opt, loss='mse', metrics=['accuracy'], steps_per_execution=2)

      # Fit the weights to the dataset
      history = m.fit(test_dataset(), steps_per_epoch=2, epochs=2)

      self.assertEqual(list(history.history.keys()), ['loss', 'accuracy'])
      self.assertEqual(type(history.history['loss']), list)
      self.assertEqual(type(history.history['accuracy']), list)
      self.assertEqual(len(history.history['loss']), 2)
      self.assertEqual(type(history.history['loss'][0]), float)
      self.assertEqual(len(history.history['accuracy']), 2)
      self.assertEqual(type(history.history['loss'][1]), float)
      self.assertEqual(type(history.history['accuracy'][0]), float)
      self.assertEqual(type(history.history['accuracy'][1]), float)

  @testing_utils.run_v2_only
  def testEval_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile("sgd", loss='mse')

      # Fit the weights to the dataset
      result = m.evaluate(test_dataset(length=96))

    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [8, 8, 2], w=0.4)
    m_cpu = keras.Model(inputs=input_layer, outputs=x)
    m_cpu.compile("sgd", loss='mse')
    cpu_result = m.evaluate(test_dataset(length=96))

    self.assertAllClose(result, cpu_result)

  @testing_utils.run_v2_only
  def testCallOrder(self):
    # Test which verifies that we can call evaluate/predict before fit.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile(optimizer="rmsprop", loss='mse')

      # Fit the weights to the dataset
      m.evaluate(test_dataset(length=96))
      m.predict(test_inference_dataset(length=96))
      m.fit(test_dataset(length=96))
      # No exception.

  @testing_utils.run_v2_only
  def testPredict_CpuMatch(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [8, 8, 2], w=0.4)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      # Generate predictions
      ipu_out = m.predict(test_inference_dataset(length=96))

    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [8, 8, 2], w=0.4)
    m_cpu = keras.Model(inputs=input_layer, outputs=x)
    cpu_out = m_cpu.predict(test_inference_dataset(length=96))

    self.assertAllClose(cpu_out, ipu_out)

  @testing_utils.run_v2_only
  def testTrainMultipleInput(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_a = keras.layers.Input(shape=(32))
      input_b = keras.layers.Input(shape=(16))

      block_a = simple_model(input_a, [8, 8], w=0.4)
      block_b = simple_model(input_b, [8, 8], w=0.4)

      concat_ab = keras.layers.concatenate([block_a, block_b])

      block_c = simple_model(concat_ab, [32, 2])
      block_d = simple_model(concat_ab, [32, 1])

      m = keras.Model(inputs=[input_a, input_b], outputs=[block_c, block_d])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile("sgd", loss=['mse', 'mse'], steps_per_execution=2)

      ds = test_dataset_two_input_output(
          length=96,
          batch_size=4,
          input_names=[input_a.name, input_b.name],
          target_names=[
              block_c.name.partition("/")[0],
              block_d.name.partition("/")[0]
          ])

      m.fit(ds)

  @testing_utils.run_v2_only
  def testTrainMultipleInputMap(self):
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_a = keras.layers.Input(shape=(32))
      input_b = keras.layers.Input(shape=(16))

      block_a = simple_model(input_a, [8, 8], w=0.4)
      block_b = simple_model(input_b, [8, 8], w=0.4)

      concat_ab = keras.layers.concatenate([block_a, block_b])

      block_c = simple_model(concat_ab, [32, 2])
      block_d = simple_model(concat_ab, [32, 1])

      m = keras.Model(inputs=[input_a, input_b], outputs=[block_c, block_d])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile("sgd", loss=['mse', 'mse'], metrics=['accuracy'])

      ds = test_dataset_two_input_output_np(
          length=96,
          input_names=[input_a.name, input_b.name],
          target_names=[
              block_c.name.partition("/")[0],
              block_d.name.partition("/")[0]
          ])
      m.fit(*ds, batch_size=4)

  @testing_utils.run_v2_only
  def testPredictNumpyData(self):
    xs = np.stack([np.ones(32, dtype=np.float32) * i for i in range(48)])
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 1], w=1)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()
      ipu_out = m.predict(xs, batch_size=2)

    # CPU
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [32, 32, 1], w=1)
    m = keras.Model(inputs=input_layer, outputs=x)
    cpu_out = m.predict(xs, batch_size=2)

    self.assertEqual(cpu_out.shape, ipu_out.shape)
    self.assertAllClose(cpu_out, ipu_out)

  @testing_utils.run_v2_only
  def testPredictNumpyDataTwoOutput(self):
    xs = np.stack([np.ones(32, dtype=np.float32) * i for i in range(48)])

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 1], w=1)
      m = keras.Model(inputs=input_layer, outputs=[x, x])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      ipu_out = m.predict(xs, batch_size=2)

    # CPU
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [32, 32, 1], w=1)
    m = keras.Model(inputs=input_layer, outputs=[x, x])
    cpu_out = m.predict(xs, batch_size=2)

    for t_cpu, t_ipu in zip(cpu_out, ipu_out):
      self.assertAllClose(t_cpu, t_ipu)

  @testing_utils.run_v2_only
  def testPredictNumpyData3D(self):
    xs = np.stack([np.ones(32, dtype=np.float32) * i for i in range(48)])

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 48], w=1)
      x = keras.layers.Reshape((4, 4, 3))(x)
      m = keras.Model(inputs=input_layer, outputs=x)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      ipu_out = m.predict(xs, batch_size=2)

    # CPU
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [32, 32, 48], w=1)
    x = keras.layers.Reshape((4, 4, 3))(x)
    m = keras.Model(inputs=input_layer, outputs=x)
    cpu_out = m.predict(xs, batch_size=2)

    self.assertEqual(cpu_out.shape, ipu_out.shape)
    self.assertAllClose(cpu_out, ipu_out)

  @testing_utils.run_v2_only
  def testPredictNumpyDataTwoOutput3D(self):
    xs = np.stack([np.ones(32, dtype=np.float32) * i for i in range(48)])

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 48], w=1)
      x = keras.layers.Reshape((4, 4, 3))(x)
      m = keras.Model(inputs=input_layer, outputs=[x, x])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      ipu_out = m.predict(xs, batch_size=2)

    # CPU
    xs = np.stack([np.ones(32, dtype=np.float32) * i for i in range(48)])
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [32, 32, 48], w=1)
    x = keras.layers.Reshape((4, 4, 3))(x)
    m = keras.Model(inputs=input_layer, outputs=[x, x])
    cpu_out = m.predict(xs, batch_size=2)

    self.assertEqual(np.shape(cpu_out), np.shape(ipu_out))
    for t_cpu, t_ipu in zip(cpu_out, ipu_out):
      self.assertAllClose(t_cpu, t_ipu)

  @testing_utils.run_v2_only
  def testFitVanillaKerasMatch(self):
    # IPU Model.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      input_layer = keras.layers.Input(shape=(32))
      x = simple_model(input_layer, [32, 32, 1], w=1)
      m = keras.Model(inputs=input_layer, outputs=x)

      m.compile('sgd', 'mse')
      ipu_out = m.fit(test_dataset(length=96), epochs=2)

    # CPU Model.
    input_layer = keras.layers.Input(shape=(32))
    x = simple_model(input_layer, [32, 32, 1], w=1)
    m = keras.Model(inputs=input_layer, outputs=x)

    m.compile('sgd', 'mse')
    cpu_out = m.fit(test_dataset(length=96), epochs=2)

    # Compare.
    self.assertAllClose(ipu_out.history['loss'], cpu_out.history['loss'])

  @testing_utils.run_v2_only
  def testTrainMultipleInputMultipleOutput(self):
    # 3 inputs, 2 outputs.
    def data_fn():
      x1 = np.ones((32), dtype=np.float64)
      x2 = np.ones((32), dtype=np.float64)
      x3 = np.ones((32), dtype=np.float64)
      y1 = np.ones((1), dtype=np.float64)
      y2 = np.ones((1), dtype=np.float64)
      ds_x = tf.data.Dataset.from_tensors((x1, x2, x3))
      ds_y = tf.data.Dataset.from_tensors((y1, y2))
      ds_xy = tf.data.Dataset.zip(
          (ds_x, ds_y)).repeat(32).batch(4, drop_remainder=True)
      return ds_xy

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
                                   activation=keras.activations.relu,
                                   name="output1")(cat)
      dense_4 = keras.layers.Dense(1,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu,
                                   name="output2")(cat)

      return ((input_1, input_2, input_3), (dense_3, dense_4))

    # IPU Test.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

      out = model.fit(data_fn(), steps_per_epoch=1, epochs=2)

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'], metrics=['accuracy'])

    cpu_out = cpu_model.fit(data_fn(), steps_per_epoch=1, epochs=2)

    # Comparison.
    self.assertEqual(len(out.history), len(cpu_out.history))

    # Check per output loss and metrics exist.
    self.assertTrue("output1_loss" in out.history)
    self.assertTrue("output2_loss" in out.history)
    self.assertTrue("output1_accuracy" in out.history)
    self.assertTrue("output2_accuracy" in out.history)

    for key in out.history:
      self.assertAllClose(out.history[key], cpu_out.history[key])

  @testing_utils.run_v2_only
  def testNestedClasses(self):
    init = keras.initializers.Constant(1)

    # 3 inputs, 2 outputs.
    def data_fn():
      x1 = np.ones((64, 32), dtype=np.float32)
      x2 = np.ones((64, 32), dtype=np.float32)
      x3 = np.ones((64, 32), dtype=np.float32)

      y1 = np.ones((64, 1), dtype=np.float32)
      y2 = np.ones((64, 3), dtype=np.float32)

      return (x1, x2, x3), (y1, y2)

    # pylint: disable=abstract-method
    class MyDenseModel(keras.Model):
      def __init__(self, units):
        super().__init__()
        self.dense1 = keras.layers.Dense(units,
                                         kernel_initializer=init,
                                         activation=keras.activations.relu)
        self.dense2 = keras.layers.Dense(units,
                                         kernel_initializer=init,
                                         activation=keras.activations.softmax)

      # pylint: disable=arguments-differ
      def call(self, inputs):
        in0, in1 = inputs
        x = self.dense1(in0)
        return x, self.dense2(in1)

    class MyLayer(keras.layers.Layer):
      def __init__(self):
        super().__init__()
        self.concat = keras.layers.Concatenate()
        self.dense1 = keras.layers.Dense(1,
                                         kernel_initializer=init,
                                         activation=keras.activations.relu)
        self.dense2 = keras.layers.Dense(3,
                                         kernel_initializer=init,
                                         activation=keras.activations.softmax)

      # pylint: disable=arguments-differ
      def call(self, inputs):
        cat = self.concat(inputs)
        return ((self.dense1(cat),), self.dense2(cat))

    def model_fn():
      input_1 = keras.Input(32)
      input_2 = keras.Input(32)
      input_3 = keras.Input(32)

      dense_1, dense_2 = MyDenseModel(16)([input_1, input_2])
      output = MyLayer()([dense_1, dense_2, input_3])

      return ((input_1, input_2, input_3), ((output[0][0], output[1])))

    # IPU Test.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'])

      out = model.fit(*data_fn(), batch_size=4)

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])

    cpu_out = cpu_model.fit(*data_fn(), batch_size=4)

    # Comparison.
    self.assertEqual(np.shape(cpu_out), np.shape(out))
    self.assertAllClose(out.history['loss'], cpu_out.history['loss'])

  @testing_utils.run_v2_only
  def testPredictMultipleOutput(self):
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

      return ((input_1, input_2, input_3), ((dense_3, dense_4)))

    # IPU Test.
    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse'])

      ipu_predict_out = model.predict(predict_input_fn(), batch_size=4)

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse'])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # Comparison.
    self.assertAllClose(cpu_predict_out, ipu_predict_out)

  @testing_utils.run_v2_only
  def testPredictMultipleOutputDifferentShapes(self):
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
      dense_5 = keras.layers.Dense(2,
                                   kernel_initializer=init,
                                   activation=keras.activations.relu)(cat)

      return ((input_1, input_2, input_3), (dense_3, (dense_4, dense_5)))

    # CPU Test.
    cpu_model = keras.Model(*model_fn())
    cpu_model.compile('sgd', ['mse', 'mse', 'mse'])

    cpu_predict_out = cpu_model.predict(predict_input_fn(), batch_size=4)

    # IPU Test.
    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      model = keras.Model(*model_fn())
      model.compile('sgd', ['mse', 'mse', 'mse'])

      ipu_predict_out = model.predict(predict_input_fn(), batch_size=4)

    self.assertAllClose(cpu_predict_out, ipu_predict_out)

  @testing_utils.run_v2_only
  def testAutocast_ComplexDatasetStructure(self):
    base_layer_utils.enable_v2_dtype_behavior()

    def f():
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

    cfg = IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      m = keras.Model(*f())

      opt = keras.optimizer_v2.gradient_descent.SGD(learning_rate=0.001)
      m.compile(opt, loss='mse')

      # Input data
      x1 = np.ones((32), dtype=np.float64)
      x2 = np.ones((32), dtype=np.float64)
      x3 = np.ones((32), dtype=np.float64)
      y1 = np.ones((1), dtype=np.float64)
      y2 = np.ones((1), dtype=np.float64)
      ds_x = tf.data.Dataset.from_tensors((x1, x2, x3))
      ds_y = tf.data.Dataset.from_tensors((y1, y2))
      ds_xy = tf.data.Dataset.zip(
          (ds_x, ds_y)).repeat(32).batch(4, drop_remainder=True)
      ds_x_tuple = tf.data.Dataset.zip(
          (ds_x,)).repeat(32).batch(4, drop_remainder=True)

      m.fit(ds_xy)
      m.predict(ds_x_tuple)
      m.evaluate(ds_xy)

      # No exceptions thrown

  @testing_utils.run_v2_only
  def testUint8(self):
    dataset = tf.data.Dataset.from_tensor_slices(np.array(range(30)))
    dataset = dataset.map(lambda x: tf.cast(x, dtype=np.uint8)).batch(
        1, drop_remainder=True).batch(1, drop_remainder=True)

    strategy = ipu.ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      i = keras.layers.Input(shape=[1])
      ci = keras.layers.Lambda(lambda x: tf.cast(x, dtype=np.float16))(i)
      o = keras.layers.Dense(1, kernel_initializer='ones')(ci)
      m = keras.Model(i, o)

      cfg = IPUConfig()
      cfg.auto_select_ipus = 1
      cfg.configure_ipu_system()

      m.compile(steps_per_execution=10)
      output = m.predict(dataset)
      self.assertAllClose(output.flatten(), range(30))


if __name__ == '__main__':
  tf.test.main()
