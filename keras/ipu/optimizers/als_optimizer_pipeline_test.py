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

import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow.python.eager import def_function
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ipu_strategy import IPUStrategyV1

import keras
from keras import layers
from keras.ipu.extensions.functional_extensions import PipelineStage
from keras.ipu.layers import CaptureUpstreamGradients
from keras.ipu.optimizers import ALSOptimizer

from keras.ipu.optimizers import als_optimizer_test_utils as als_tu

TEST_CASES = als_tu.generate_test_cases(no_ga=True)


def _pipeline_fn(wrapper_type=None, init=1.0):
  input_layer = layers.Input(als_tu.DATA.shape[1], dtype=als_tu.DATA.dtype)

  with PipelineStage(0):
    x = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type,
                        init=init)(input_layer)
    x = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type, init=init)(x)

  with PipelineStage(1):
    x = als_tu.dense_fn(tf.float16, wrapper_type=wrapper_type, init=init)(x)
    x = als_tu.dense_fn(tf.float16, init=init)(x)

  return keras.Model(input_layer, x)


class ALSOptimizerPipelineTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(TEST_CASES)
  def testSimpleTraining(self, optimizer_type, optimizer_args,
                         optimizer_kwargs, als_kwargs, wrapper_type,
                         ga_steps_per_replica):  # pylint: disable=unused-argument
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 16
    cfg.configure_ipu_system()

    captured_only = wrapper_type == CaptureUpstreamGradients

    strategy = IPUStrategyV1()
    with strategy.scope():
      opt = optimizer_type(*optimizer_args, **optimizer_kwargs)
      opt_wrapper = ALSOptimizer(opt,
                                 **als_kwargs,
                                 captured_grads_only=captured_only)

      steps = 4 * ga_steps_per_replica

      m = _pipeline_fn(wrapper_type=wrapper_type)

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=steps)

      m.compile(optimizer=opt_wrapper, loss='mse', steps_per_execution=steps)

      history = m.fit(
          als_tu.DATA,
          als_tu.TARGETS,
          epochs=2,  # Converges after two executions.
          verbose=False)

    last_loss = float('inf')
    for l in history.history['loss']:
      self.assertTrue(np.isfinite(l))
      self.assertLess(l, last_loss)
      last_loss = l

  @parameterized.named_parameters(TEST_CASES)
  def testDistributionWithSaturation(self, optimizer_type, optimizer_args,
                                     optimizer_kwargs, als_kwargs,
                                     wrapper_type, ga_steps_per_replica):  # pylint: disable=unused-argument
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
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
          opt, **als_kwargs, captured_grads_only=captured_only)  # See above.

      steps = 8

      m = _pipeline_fn(wrapper_type=wrapper_type, init=4.0)

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=steps)

      m.compile(optimizer=opt_wrapper, loss='mse', steps_per_execution=steps)

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
                        verbose=False)

        self.assertTrue(np.isfinite(history.history['loss'][0]))

        # We expect gradients only for the float16 dense layers to be taken
        # into consideration in the histogram. In this case, most units
        # gradients in these layers should have overflowed.
        hist = strategy.run(h)
        self.assertGreater(hist[1], hist[0])

        # Check the LSF hasn't changed yet.
        lsf = strategy.run(g)
        self.assertAllEqual(lsf, opt_wrapper.initial_loss_scaling_factor)

      # The LSF should have decreased after this next epoch.
      history = m.fit(als_tu.DATA,
                      als_tu.TARGETS_HUGE,
                      epochs=1,
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

  @parameterized.named_parameters(TEST_CASES)
  def testDistributionWithoutSaturation(self, optimizer_type, optimizer_args,
                                        optimizer_kwargs, als_kwargs,
                                        wrapper_type, ga_steps_per_replica):  # pylint: disable=unused-argument
    cfg = IPUConfig()
    cfg.auto_select_ipus = 2
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

      steps = 8

      m = _pipeline_fn(wrapper_type=wrapper_type, init=0.0)

      m.set_pipelining_options(gradient_accumulation_steps_per_replica=steps)

      m.compile(optimizer=opt_wrapper, loss='mse', steps_per_execution=steps)

      @def_function.function(jit_compile=True)
      def g():
        return opt_wrapper.loss_scaling_factor

      @def_function.function(jit_compile=True)
      def h():
        return opt_wrapper.histogram

      for _ in range(opt_wrapper.update_frequency):
        history = m.fit(als_tu.DATA, als_tu.TARGETS, epochs=1, verbose=False)

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
      history = m.fit(als_tu.DATA, als_tu.TARGETS, epochs=1, verbose=False)

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


if __name__ == "__main__":
  tf.test.main()
