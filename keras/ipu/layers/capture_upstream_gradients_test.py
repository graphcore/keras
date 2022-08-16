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

from functools import partial

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig

import keras
from keras.ipu.layers import CaptureUpstreamGradients
from keras.ipu.layers import CaptureActivationGradients


class SGDCaptureOptimizer(keras.optimizer_v2.gradient_descent.SGD):
  def __init__(self, learning_rate, captured_grad_outfeed=None):
    super().__init__(learning_rate=learning_rate)

    self.captured_grad_outfeed = captured_grad_outfeed

  def apply_gradients(self, grads_and_vars, captured_grads=None):  # pylint: disable=arguments-differ
    if captured_grads and self.captured_grad_outfeed and \
      not self.captured_grad_outfeed.enqueued:
      self.captured_grad_outfeed.enqueue(captured_grads)

    return super().apply_gradients(grads_and_vars)

  @property
  def captured_grad_outfeed(self):
    return self._captured_grad_outfeed

  @captured_grad_outfeed.setter
  def captured_grad_outfeed(self, val):
    if val and not isinstance(val, ipu_outfeed_queue.IPUOutfeedQueue):
      raise ValueError(
          "Expected an instance of "
          "tensorflow.python.ipu.ipu_outfeed_queue.IPUOutfeedQueue")

    self._captured_grad_outfeed = val

  @property
  def supports_captured_grads(self):
    return True


def data_fn(n=16, dtype=np.float32):
  return np.ones((n, 4), dtype=dtype), \
          2.0 * np.ones((n, 4), dtype=dtype)


def _dense(capture_layer_type, with_capture, layer_input, d):
  init = keras.initializers.Constant(1.0)

  if with_capture:
    if capture_layer_type == CaptureUpstreamGradients:
      x = CaptureUpstreamGradients(
          keras.layers.Dense(d, activation=None,
                             kernel_initializer=init))(layer_input)
      x = keras.layers.Activation(keras.activations.relu)(x)
    elif capture_layer_type == CaptureActivationGradients:
      x = CaptureActivationGradients(
          keras.layers.Dense(d, activation='relu',
                             kernel_initializer=init))(layer_input)
    else:
      raise ValueError("Unknown gradient capture layer type.")
  else:
    x = keras.layers.Dense(d, activation=None,
                           kernel_initializer=init)(layer_input)
    x = keras.layers.Activation(keras.activations.relu)(x)

  return x


def _model_fn(capture_layer_type, with_capture=True, d=4):
  input_layer = keras.layers.Input(shape=(d))
  x = _dense(capture_layer_type, with_capture, input_layer, d)
  return keras.Model(input_layer, x)


def _pipeline_fn(capture_layer_type, with_capture=True, d=4):
  input_layer = keras.layers.Input(shape=(d))

  with keras.ipu.PipelineStage(0):
    x = _dense(capture_layer_type, with_capture, input_layer, d)

  with keras.ipu.PipelineStage(1):
    x = _dense(capture_layer_type, with_capture, x, d)

  return keras.Model(input_layer, x)


def opt_fn(lr=0.01):
  outfeed = ipu_outfeed_queue.IPUOutfeedQueue()
  return SGDCaptureOptimizer(lr, captured_grad_outfeed=outfeed), outfeed


def manually_compute_model_fn_captured_grads(strategy):
  dense = keras.layers.Dense(
      4, activation=None, kernel_initializer=keras.initializers.Constant(1.0))
  activation = keras.layers.Activation(keras.activations.relu)

  @tf.function(jit_compile=True)
  def f(x, t):
    y = dense(x)
    a = activation(y)

    # Compute dl/da - gradient of MSE
    dl_da = (2.0 / x.shape[0]) * (a - t)

    # Compute da/dy
    da_dy = tf.raw_ops.ReluGrad(gradients=dl_da, features=y)

    # Compute dl/dy
    dl_dy = tf.reduce_sum(dl_da * da_dy, axis=1)

    return dl_dy

  return strategy.run(f, [*data_fn()])


def manually_compute_pipeline_fn_captured_grads(strategy):
  dense_1 = keras.layers.Dense(
      4, activation=None, kernel_initializer=keras.initializers.Constant(1.0))
  activation_1 = keras.layers.Activation(keras.activations.relu)

  dense_2 = keras.layers.Dense(
      4, activation=None, kernel_initializer=keras.initializers.Constant(1.0))
  activation_2 = keras.layers.Activation(keras.activations.relu)

  @tf.function(jit_compile=True)
  def f(x, t):
    y_1 = dense_1(x)
    a_1 = activation_1(y_1)

    y_2 = dense_2(a_1)
    a_2 = activation_2(y_2)

    # Compute dl/da2 - gradient of MSE
    dl_da2 = (2.0 / x.shape[0]) * (a_2 - t)

    # Compute da2/dy2
    da2_dy2 = tf.raw_ops.ReluGrad(gradients=dl_da2, features=y_2)

    # Compute dy2/dw2
    dy2_dw2 = tf.matmul(da2_dy2, tf.transpose(dense_2.kernel))

    # Compute dw2/da1
    dy2_da1 = tf.raw_ops.ReluGrad(gradients=dy2_dw2, features=y_1)

    return tf.reduce_sum(da2_dy2, axis=1), tf.reduce_sum(dy2_da1, axis=1)

  return strategy.run(f, [*data_fn()])


MODEL_FN_CASES = [
    partial(_model_fn, CaptureUpstreamGradients),
    partial(_model_fn, CaptureActivationGradients)
]

PIPELINE_FN_CASES = [
    partial(_pipeline_fn, CaptureUpstreamGradients),
    partial(_pipeline_fn, CaptureActivationGradients)
]


class CaptureUpstreamGradientsTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters(MODEL_FN_CASES)
  def testExplicitWrapping(self, model_fn):
    config = IPUConfig()
    config.auto_select_ipus = 1
    config.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Create and fit model with gradient capture layers.
      opt, outfeed = opt_fn()
      m = model_fn()
      m.compile(opt, 'mse')

      history = m.fit(*data_fn(), batch_size=4, steps_per_epoch=1, epochs=1)

      # Dequeue captured grads and ensure we only have entries for one op.
      captured_grads = outfeed.dequeue()

      self.assertEqual(len(captured_grads), 1)
      captured_grads_values = list(captured_grads.values())[0]

      # Manually compute the expected gradient values for those grads
      # that have been captured.
      dl_dy = manually_compute_model_fn_captured_grads(strategy)

      self.assertAllEqual(np.reshape(dl_dy, 16),
                          np.reshape(captured_grads_values, 16))

      # Create and fit model without gradient capture layers.
      m = model_fn(with_capture=False)
      m.compile('sgd', 'mse')

      history_no_capture = m.fit(*data_fn(),
                                 batch_size=4,
                                 steps_per_epoch=1,
                                 epochs=1)

      # Verify losses are consistent.
      self.assertAllEqual(history.history['loss'],
                          history_no_capture.history['loss'])

  @parameterized.parameters(PIPELINE_FN_CASES)
  def testExplicitWrappingPipeline(self, pipeline_fn):
    config = IPUConfig()
    config.auto_select_ipus = 2
    config.configure_ipu_system()

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      # Create and fit pipelined model with gradient capture layers.
      opt, outfeed = opt_fn()
      m = pipeline_fn()
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               device_mapping=[0, 1])
      m.compile(opt, 'mse', steps_per_execution=4)

      history = m.fit(*data_fn(), batch_size=4, epochs=1)

      # Dequeue captured grads and ensure we only have entries for one op.
      captured_grads = outfeed.dequeue()

      self.assertEqual(len(captured_grads), 2)
      captured_grads_values = list(captured_grads.values())

      # Manually compute the expected gradient values for those grads
      # that have been captured.
      stage_2_act_grad, stage_1_act_grad = \
        manually_compute_pipeline_fn_captured_grads(strategy)

      self.assertAllEqual(np.reshape(captured_grads_values[0], 16),
                          np.reshape(stage_1_act_grad / 4, 16))

      self.assertAllEqual(np.reshape(captured_grads_values[1], 16),
                          np.reshape(stage_2_act_grad * 4, 16))

      # Create and pipelined fit model without gradient capture layers.
      m = pipeline_fn(with_capture=False)
      m.set_pipelining_options(gradient_accumulation_steps_per_replica=4,
                               device_mapping=[0, 1])
      m.compile('sgd', 'mse', steps_per_execution=4)

      history_no_capture = m.fit(*data_fn(), batch_size=4, epochs=1)

      # Verify losses are consistent.
      self.assertAllEqual(history.history['loss'],
                          history_no_capture.history['loss'])


if __name__ == "__main__":
  tf.test.main()
