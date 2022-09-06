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

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow.python.eager import def_function
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ipu import loops
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.eager import backprop as ipu_backprop
from tensorflow.python.ipu.ipu_strategy import IPUStrategyV1
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses

import keras
from keras import layers
from keras.ipu.layers import CaptureUpstreamGradients
from keras.ipu.layers import CaptureActivationGradients
from keras.ipu.optimizers import ALSOptimizer
from keras.ipu.optimizers import ALSGradientAccumulationOptimizer
from keras.optimizer_v2 import adam as adam_v2
from keras.optimizer_v2 import gradient_descent as gradient_descent_v2

from keras.ipu.optimizers import als_optimizer_test_utils as als_tu

TEST_CASES = als_tu.generate_test_cases()


class ALSOptimizerTest(tf.test.TestCase, parameterized.TestCase):
  def testInvalidInitialLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "initial_loss_scaling_factor must be a power of two"):
      _ = ALSOptimizer(opt, initial_loss_scaling_factor=5)

  def testInvalidUpdateFrequency(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "update_frequency must be nonzero and positive"):
      _ = ALSOptimizer(opt, update_frequency=0)

  def testInvalidIncreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(ValueError,
                                "increase_factor must be a power of two"):
      _ = ALSOptimizer(opt, increase_factor=3)

  def testCannotSetHistogram(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = ALSOptimizer(opt)
    with self.assertRaisesRegex(ValueError,
                                "histogram is a read only property."):
      opt_wrapper.histogram = None

  def testCannotSetNormalizedHistogram(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = ALSOptimizer(opt)
    with self.assertRaisesRegex(
        ValueError, "normalized_histogram is a read only property."):
      opt_wrapper.normalized_histogram = None

  def testCannotSetLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    opt_wrapper = ALSOptimizer(opt)
    with self.assertRaisesRegex(
        ValueError, "loss_scaling_factor is a read only property."):
      opt_wrapper.loss_scaling_factor = 1

  def testCannotSetDecreaseFactor(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(ValueError,
                                "decrease_factor is a read only property."):
      opt = ALSOptimizer(opt)
      opt.decrease_factor = 2

  def testInvalidMaxLSF(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "max_loss_scaling_factor must be a power of two"):
      _ = ALSOptimizer(opt, max_loss_scaling_factor=5)

  def testRatioThresholdTooLow(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "ratio_threshold must be greater than zero and less than one"):
      _ = ALSOptimizer(opt, ratio_threshold=-1)

  def testRatioThresholdTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "ratio_threshold must be greater than zero and less than one"):
      _ = ALSOptimizer(opt, ratio_threshold=2)

  def testInitialLSFTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError, "initial_loss_scaling_factor must be less than"
        " max_loss_scaling_factor"):
      _ = ALSOptimizer(opt,
                       initial_loss_scaling_factor=4,
                       max_loss_scaling_factor=2)

  def testInitialLSFAndIncreaseFactorTooHigh(self):
    opt = gradient_descent_v2.SGD(0.1)
    with self.assertRaisesRegex(
        ValueError,
        "initial_loss_scaling_factor x increase_factor must be less "
        "than max_loss_scaling_factor"):
      _ = ALSOptimizer(opt,
                       initial_loss_scaling_factor=1,
                       increase_factor=4,
                       max_loss_scaling_factor=4)

  @parameterized.named_parameters(*TEST_CASES)
  def testSimpleTraining(self,
                         optimizer_type,
                         optimizer_args,
                         optimizer_kwargs,
                         als_kwargs,
                         wrapper_type,
                         ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      if ga_steps_per_replica > 1:
        opt_wrapper = ALSGradientAccumulationOptimizer(opt_wrapper,
                                                       ga_steps_per_replica)

      dense = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type)

      @def_function.function(jit_compile=True)
      def f(x, t, _):
        with ipu_backprop.GradientCaptureTape() as tape:
          y = dense(x)
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense.trainable_weights
        gv = als_tu.get_grads_and_vars(ga_steps_per_replica, v, opt_wrapper, l)
        opt_wrapper.apply_gradients(gv, captured_grads=tape.captured_gradients)
        return x, t, l

      @def_function.function(jit_compile=True)
      def loop_fn(x, t):
        _, _, l = loops.repeat(ga_steps_per_replica, f, inputs=[x, t, 0.0])
        return l

      model_losses = []
      for _ in range(3):
        res = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS])
        model_losses.append(res)

    last_loss = float('inf')
    for r in model_losses:
      self.assertTrue(np.isfinite(r))
      self.assertLess(r, last_loss)
      last_loss = r

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturation(self,
                                     optimizer_type,
                                     optimizer_args,
                                     optimizer_kwargs,
                                     als_kwargs,
                                     wrapper_type,
                                     ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      if ga_steps_per_replica > 1:
        opt_wrapper = ALSGradientAccumulationOptimizer(opt_wrapper,
                                                       ga_steps_per_replica)

      dense0 = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type)
      dense1 = als_tu.dense_fn(tf.float32)

      @def_function.function(jit_compile=True)
      def f(x, t, _):
        with ipu_backprop.GradientCaptureTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        gv = als_tu.get_grads_and_vars(ga_steps_per_replica, v, opt_wrapper, l)
        opt_wrapper.apply_gradients(gv, captured_grads=tape.captured_gradients)
        return x, t, l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      @def_function.function(jit_compile=True)
      def loop_fn(x, t):
        _, _, l = loops.repeat(ga_steps_per_replica, f, inputs=[x, t, 0.0])
        return l

      for _ in range(opt_wrapper.update_frequency):
        l = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS_HUGE])
        self.assertTrue(np.isfinite(l))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[1], hist[0])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF decreases after the next epoch.
      _ = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS_HUGE])
      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturation(self,
                                        optimizer_type,
                                        optimizer_args,
                                        optimizer_kwargs,
                                        als_kwargs,
                                        wrapper_type,
                                        ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      if ga_steps_per_replica > 1:
        opt_wrapper = ALSGradientAccumulationOptimizer(opt_wrapper,
                                                       ga_steps_per_replica)

      dense0 = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type)
      dense1 = als_tu.dense_fn(tf.float32)

      @def_function.function(jit_compile=True)
      def f(x, t, _):
        with ipu_backprop.GradientCaptureTape() as tape:
          y = dense1(dense0(x))
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables + dense1.trainable_variables
        gv = als_tu.get_grads_and_vars(ga_steps_per_replica, v, opt_wrapper, l)
        opt_wrapper.apply_gradients(gv, captured_grads=tape.captured_gradients)
        return x, t, l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      @def_function.function(jit_compile=True)
      def loop_fn(x, t):
        _, _, l = loops.repeat(ga_steps_per_replica, f, inputs=[x, t, 0.0])
        return l

      for _ in range(opt_wrapper.update_frequency):
        l = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS])
        self.assertTrue(np.isfinite(l))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should not have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[0], hist[1])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF increases after the next epoch.
      _ = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS])
      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturationNoStatAccum(self,
                                                optimizer_type,
                                                optimizer_args,
                                                optimizer_kwargs,
                                                als_kwargs,
                                                wrapper_type,
                                                ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # For this particular test, using a one-shot collection of gradients,
    # coupled with using only captured gradients is not sufficient to trigger
    # overflow. The reason for this is that when we compute statistics over
    # gradients obtained from get_gradients, they are summed by batch, whereas
    # captured gradients are not. They are directly lifted from the op grad.
    captured_only = wrapper_type == CaptureUpstreamGradients
    if captured_only:
      return

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(
          opt, **als_kwargs, accumulate_statistics_over_update_period=False)

      if ga_steps_per_replica > 1:
        opt_wrapper = ALSGradientAccumulationOptimizer(opt_wrapper,
                                                       ga_steps_per_replica)

      dense0 = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type)

      @def_function.function(jit_compile=True)
      def f(x, t, _):
        with ipu_backprop.GradientCaptureTape() as tape:
          y = dense0(x)
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables
        gv = als_tu.get_grads_and_vars(ga_steps_per_replica, v, opt_wrapper, l)
        opt_wrapper.apply_gradients(gv, captured_grads=tape.captured_gradients)
        return x, t, l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      @def_function.function(jit_compile=True)
      def loop_fn(x, t):
        _, _, l = loops.repeat(ga_steps_per_replica, f, inputs=[x, t, 0.0])
        return l

      for _ in range(opt_wrapper.update_frequency):
        l = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS_HUGE])
        self.assertTrue(np.isfinite(l))

        # In this loop, we expect the histogram to always be zeros.
        hist = strategy.run(h)
        self.assertAllEqual(hist, np.zeros_like(hist))

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF decreases after the next epoch.
      _ = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS_HUGE])
      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturationNoStatAccum(self,
                                                   optimizer_type,
                                                   optimizer_args,
                                                   optimizer_kwargs,
                                                   als_kwargs,
                                                   wrapper_type,
                                                   ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(
          opt,
          **als_kwargs,
          accumulate_statistics_over_update_period=False,
          captured_grads_only=captured_only)

      if ga_steps_per_replica > 1:
        opt_wrapper = ALSGradientAccumulationOptimizer(opt_wrapper,
                                                       ga_steps_per_replica)

      dense0 = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type)

      @def_function.function(jit_compile=True)
      def f(x, t, _):
        with ipu_backprop.GradientCaptureTape() as tape:
          y = dense0(x)
          l = losses.mean_squared_error(labels=t, predictions=y)

        v = dense0.trainable_variables
        gv = als_tu.get_grads_and_vars(ga_steps_per_replica, v, opt_wrapper, l)
        opt_wrapper.apply_gradients(gv, captured_grads=tape.captured_gradients)
        return x, t, l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      @def_function.function(jit_compile=True)
      def loop_fn(x, t):
        _, _, l = loops.repeat(ga_steps_per_replica, f, inputs=[x, t, 0.0])
        return l

      for _ in range(opt_wrapper.update_frequency):
        l = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS])
        self.assertTrue(np.isfinite(l))

        # In this loop, we expect the histogram to always be zeros.
        hist = strategy.run(h)
        self.assertAllEqual(hist, np.zeros_like(hist))

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that the LSF increases after the next epoch.
      _ = strategy.run(loop_fn, args=[als_tu.DATA, als_tu.TARGETS])
      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testSimpleTrainingKeras(self,
                              optimizer_type,
                              optimizer_args,
                              optimizer_kwargs,
                              als_kwargs,
                              wrapper_type,
                              ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      input_layer = layers.Input(shape=als_tu.DATA.shape[1])
      dense = als_tu.dense_fn(tf.float16, wrapper_type)(input_layer)

      m = keras.Model(inputs=input_layer, outputs=dense)

      m.compile(optimizer=opt_wrapper,
                loss='mse',
                steps_per_execution=ga_steps_per_replica)

      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=ga_steps_per_replica)

      history = m.fit(als_tu.DATA,
                      als_tu.TARGETS,
                      epochs=3,
                      batch_size=als_tu.BATCH_SIZE,
                      steps_per_epoch=ga_steps_per_replica,
                      verbose=False)

    last_loss = float('inf')
    for l in history.history['loss']:
      self.assertTrue(np.isfinite(l))
      self.assertLess(l, last_loss)
      last_loss = l

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithSaturationKeras(self,
                                          optimizer_type,
                                          optimizer_args,
                                          optimizer_kwargs,
                                          als_kwargs,
                                          wrapper_type,
                                          ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      input_layer = layers.Input(shape=als_tu.DATA.shape[1])

      dense0 = als_tu.dense_fn(tf.float16,
                               wrapper_type=wrapper_type)(input_layer)
      dense1 = als_tu.dense_fn(tf.float32)(dense0)

      m = keras.Model(inputs=input_layer, outputs=dense1)

      m.compile(optimizer=opt_wrapper,
                loss='mse',
                steps_per_execution=ga_steps_per_replica)

      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=ga_steps_per_replica)

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency):
        history = m.fit(als_tu.DATA,
                        als_tu.TARGETS_HUGE,
                        epochs=1,
                        batch_size=als_tu.BATCH_SIZE,
                        steps_per_epoch=ga_steps_per_replica,
                        verbose=False)

        self.assertTrue(np.isfinite(history.history['loss'][0]))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[1], hist[0])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # The LSF should have decreased after this next epoch.
      history = m.fit(als_tu.DATA,
                      als_tu.TARGETS_HUGE,
                      epochs=1,
                      batch_size=als_tu.BATCH_SIZE,
                      steps_per_epoch=ga_steps_per_replica,
                      verbose=False)

      lsf = strategy.run(g)
      self.assertLess(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.decrease_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  @parameterized.named_parameters(*TEST_CASES)
  def testDistributionWithoutSaturationKeras(self,
                                             optimizer_type,
                                             optimizer_args,
                                             optimizer_kwargs,
                                             als_kwargs,
                                             wrapper_type,
                                             ga_steps_per_replica=1):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    # In these tests, if we are using CaptureUpstreamGradients then we'll end
    # up with duplicate gradients in the histogram. So we run ALS using only
    # the captured gradients.
    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      input_layer = layers.Input(shape=als_tu.DATA.shape[1])

      dense0 = als_tu.dense_fn(tf.float16,
                               wrapper_type=wrapper_type)(input_layer)
      dense1 = als_tu.dense_fn(tf.float32)(dense0)

      m = keras.Model(inputs=input_layer, outputs=dense1)

      m.compile(optimizer=opt_wrapper,
                loss='mse',
                steps_per_execution=ga_steps_per_replica)

      m.set_gradient_accumulation_options(
          gradient_accumulation_steps_per_replica=ga_steps_per_replica)

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency):
        history = m.fit(als_tu.DATA,
                        als_tu.TARGETS,
                        epochs=1,
                        batch_size=als_tu.BATCH_SIZE,
                        steps_per_epoch=ga_steps_per_replica,
                        verbose=False)

        self.assertTrue(np.isfinite(history.history['loss'][0]))

        # We expect gradients only for the float16 dense layer to be taken into
        # consideration in the histogram. In this case, most units gradients
        # in this layer should not have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[0], hist[1])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # The LSF should have increased after this next epoch.
      history = m.fit(als_tu.DATA,
                      als_tu.TARGETS,
                      epochs=1,
                      batch_size=als_tu.BATCH_SIZE,
                      steps_per_epoch=ga_steps_per_replica,
                      verbose=False)

      lsf = strategy.run(g)
      self.assertGreater(lsf, opt_wrapper.initial_loss_scaling_factor)

      # Check that it's the expected value.
      expected_lsf = \
        opt_wrapper.initial_loss_scaling_factor * opt_wrapper.increase_factor
      self.assertAllClose(lsf, expected_lsf)

      # Check that the histogram has been reset as there has
      # been an LSF update.
      hist = strategy.run(h)
      self.assertAllEqual(hist, np.zeros_like(hist))

  def testUpperLSFCap(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = gradient_descent_v2.SGD(0.01)
      opt_wrapper = ALSOptimizer(opt,
                                 increase_factor=2,
                                 initial_loss_scaling_factor=1,
                                 max_loss_scaling_factor=16,
                                 update_frequency=1)

      v = variables.Variable(1.0, dtype=tf.float16)

      @def_function.function(jit_compile=True)
      def f():
        with GradientTape() as tape:
          y = 3 * v
          l = losses.mean_squared_error(labels=array_ops.ones_like(v),
                                        predictions=y)
        opt_wrapper.minimize(l, [v], tape=tape)
        return l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return v.assign(1.0)

      # We expect the LSF to increase and cap at 8.
      for expected_lsf in [2, 4, 8, 8]:
        # "Train"
        _ = strategy.run(f)

        # Check the LSF.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, expected_lsf)

        # Reset var.
        _ = strategy.run(h)

  def testLowerLSFCap(self):
    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = gradient_descent_v2.SGD(0.01)
      opt_wrapper = ALSOptimizer(opt,
                                 increase_factor=2,
                                 initial_loss_scaling_factor=16,
                                 update_frequency=1)

      v_init = np.finfo(np.float16).max
      v = variables.Variable(v_init, dtype=tf.float16, shape=())

      @def_function.function(jit_compile=True)
      def f():
        with GradientTape() as tape:
          y = 3 * v
          l = losses.mean_squared_error(labels=array_ops.ones_like(v),
                                        predictions=y)
        opt_wrapper.minimize(l, [v], tape=tape)
        return l

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return v.assign(v_init)

      # We expect the LSF to decrease and cap at 1.
      for expected_lsf in [8, 4, 2, 1, 1]:
        # "Train"
        _ = strategy.run(f)

        # Check the LSF.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, expected_lsf)

        # Reset var.
        _ = strategy.run(h)


if __name__ == "__main__":
  tf.test.main()
