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
"""Tests specific to `Sequential` model."""

import numpy as np

import tensorflow.compat.v2 as tf

import keras
from keras import keras_parameterized
from keras import testing_utils

from tensorflow.python import ipu


class TestSequential(keras_parameterized.TestCase):
  def setUp(self):
    super().setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super().tearDown()

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_basic_methods(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=2))
    model.add(keras.layers.Dropout(0.3, name='dp'))
    model.add(
        keras.layers.Dense(2,
                           kernel_regularizer='l2',
                           kernel_constraint='max_norm'))
    self.assertEqual(len(model.layers), 3)
    self.assertEqual(len(model.weights), 2 * 2)
    self.assertEqual(model.get_layer(name='dp').name, 'dp')

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_input_defined_first_layer(self):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(2,), name='input_layer'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Dropout(0.3, name='dp'))
    model.add(
        keras.layers.Dense(2,
                           kernel_regularizer='l2',
                           kernel_constraint='max_norm'))
    self.assertLen(model.layers, 3)
    self.assertLen(model.weights, 2 * 2)
    self.assertEqual(model.get_layer(name='dp').name, 'dp')

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_single_layer_in_init(self):
    model = keras.models.Sequential(keras.layers.Dense(1))
    self.assertLen(model.layers, 1)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_pop(self):
    batch_size = 5
    num_classes = 2

    model = testing_utils.get_small_sequential_mlp(num_classes, num_classes)
    self.assertIsInstance(
        model, keras.ipu.extensions.sequential_extensions.SequentialExtension)
    model.compile(loss='mse', optimizer='rmsprop')
    x = np.random.random((batch_size, num_classes))
    y = np.random.random((batch_size, num_classes))
    model.fit(x, y, batch_size=batch_size, epochs=1)
    model.pop()
    self.assertEqual(len(model.layers), 1)
    self.assertEqual(model.output_shape, (batch_size, num_classes))
    model.compile(loss='mse', optimizer='rmsprop')
    y = np.random.random((batch_size, num_classes))
    model.fit(x, y, batch_size=batch_size, epochs=1)

    # Test popping single-layer model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(num_classes))
    model.pop()
    self.assertEqual(model.layers, [])
    self.assertEqual(model.outputs, None)

    # Invalid use case
    model = keras.models.Sequential()
    with self.assertRaises(TypeError):
      model.pop()

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_deferred_build_with_np_arrays(self):
    input_dim = 3
    batch_size = 5
    num_classes = 2

    model = testing_utils.get_small_sequential_mlp(num_classes, num_classes)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=[keras.metrics.CategoricalAccuracy()],
                  run_eagerly=testing_utils.should_run_eagerly())
    self.assertEqual(len(model.layers), 2)
    with self.assertRaisesRegex(
        ValueError, 'Weights for model .* have not yet been created'):
      len(model.weights)
    self.assertFalse(model.built)

    x = np.random.random((batch_size, input_dim))
    y = np.random.random((batch_size, num_classes))
    model.fit(x, y, batch_size=batch_size, epochs=1)
    self.assertTrue(model.built)
    self.assertEqual(len(model.weights), 2 * 2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_deferred_build_with_dataset_iterators(self):
    num_hidden = 5
    input_dim = 3
    num_classes = 2
    num_samples = 50
    steps_per_epoch = 10

    model = testing_utils.get_small_sequential_mlp(num_hidden, num_classes)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=[keras.metrics.CategoricalAccuracy()])
    self.assertEqual(len(model.layers), 2)
    with self.assertRaisesRegex(
        ValueError, 'Weights for model .* have not yet been created'):
      len(model.weights)
    self.assertFalse(model.built)

    x = tf.ones((num_samples, input_dim))
    y = tf.zeros((num_samples, num_classes))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(10, drop_remainder=True)

    model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch)
    self.assertTrue(model.built)
    self.assertEqual(len(model.weights), 2 * 2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_training_and_eval_methods_on_symbolic_tensors(self):
    def get_model():
      model = testing_utils.get_small_sequential_mlp(10, 4)
      model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
      return model

    inputs = tf.zeros(shape=(30, 3))
    targets = tf.zeros(shape=(30, 4))

    model = get_model()
    model.fit(inputs, targets, epochs=10, steps_per_epoch=30)

    model = get_model()
    model.evaluate(inputs, targets, steps=2, verbose=0)

    model = get_model()
    model.predict(inputs, steps=2)

    model = get_model()
    model.fit(inputs,
              targets,
              epochs=1,
              steps_per_epoch=2,
              verbose=0,
              validation_data=(inputs, targets),
              validation_steps=2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_invalid_use_cases(self):
    # Added objects must be layer instances
    with self.assertRaises(TypeError):
      model = keras.models.Sequential()
      model.add(None)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_nested_sequential_trainability(self):
    input_dim = 20
    num_units = 10
    num_classes = 2

    inner_model = keras.models.Sequential()
    inner_model.add(keras.layers.Dense(num_units, input_shape=(input_dim,)))

    model = keras.models.Sequential()
    model.add(inner_model)
    model.add(keras.layers.Dense(num_classes))

    self.assertEqual(len(model.layers), 2)

    self.assertEqual(len(model.trainable_weights), 4)
    inner_model.trainable = False
    self.assertEqual(len(model.trainable_weights), 2)
    inner_model.trainable = True
    self.assertEqual(len(model.trainable_weights), 4)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_update_disabling(self):
    val_a = np.random.random((32, 4))
    val_out = np.random.random((32, 4))

    model = keras.models.Sequential()
    model.add(keras.layers.BatchNormalization(input_shape=(4,)))

    model.trainable = False
    model.compile('sgd', 'mse')

    x1 = model.predict(val_a)
    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    self.assertAllClose(x1, x2, atol=1e-7)

    model.trainable = True
    model.compile('sgd', 'mse')

    model.train_on_batch(val_a, val_out)
    x2 = model.predict(val_a)
    assert np.abs(np.sum(x1 - x2)) > 1e-5

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_deferred_build_serialization(self):
    num_hidden = 5
    input_dim = 3
    batch_size = 5
    num_classes = 2

    model = testing_utils.get_small_sequential_mlp(num_hidden, num_classes)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=[keras.metrics.CategoricalAccuracy()],
                  run_eagerly=testing_utils.should_run_eagerly())
    self.assertFalse(model.built)

    x = np.random.random((batch_size, input_dim))
    y = np.random.random((batch_size, num_classes))
    model.train_on_batch(x, y)
    self.assertTrue(model.built)

    config = model.get_config()
    new_model = keras.models.Sequential.from_config(config)
    new_model.compile(loss='mse',
                      optimizer='rmsprop',
                      metrics=[keras.metrics.CategoricalAccuracy()],
                      run_eagerly=testing_utils.should_run_eagerly())
    x = np.random.random((batch_size, input_dim))
    y = np.random.random((batch_size, num_classes))
    new_model.train_on_batch(x, y)
    self.assertEqual(len(new_model.layers), 2)
    self.assertEqual(len(new_model.weights), 4)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_shape_inference_deferred(self):
    model = testing_utils.get_small_sequential_mlp(4, 5)
    output_shape = model.compute_output_shape((None, 7))
    self.assertEqual(tuple(output_shape.as_list()), (None, 5))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_build_deferred(self):
    model = testing_utils.get_small_sequential_mlp(4, 5)

    model.build((None, 10))
    self.assertTrue(model.built)
    self.assertEqual(len(model.weights), 4)

    # Test with nested model
    model = testing_utils.get_small_sequential_mlp(4, 3)
    inner_model = testing_utils.get_small_sequential_mlp(4, 5)
    model.add(inner_model)

    model.build((None, 10))
    self.assertTrue(model.built)
    self.assertEqual(len(model.weights), 8)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_deferred_manual_build(self):
    model = testing_utils.get_small_sequential_mlp(4, 5)
    self.assertFalse(model.built)
    model(tf.zeros([1, 2]))
    self.assertTrue(model.built)
    model.compile('rmsprop',
                  loss='mse',
                  run_eagerly=testing_utils.should_run_eagerly())
    model.train_on_batch(np.zeros((1, 2)), np.zeros((1, 5)))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_sequential_nesting(self):
    model = testing_utils.get_small_sequential_mlp(4, 3)
    inner_model = testing_utils.get_small_sequential_mlp(4, 5)
    model.add(inner_model)

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  run_eagerly=testing_utils.should_run_eagerly())
    x = np.random.random((2, 6))
    y = np.random.random((2, 5))
    model.fit(x, y, batch_size=2, epochs=1)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_multi_output_layer_not_accepted(self):
    class MultiOutputLayer(keras.layers.Layer):
      def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs, inputs

    with self.assertRaisesRegex(ValueError,
                                'should have a single output tensor'):
      keras.Sequential([MultiOutputLayer(input_shape=(3,))])

    with self.assertRaisesRegex(ValueError,
                                'should have a single output tensor'):
      keras.Sequential(
          [keras.layers.Dense(1, input_shape=(3,)),
           MultiOutputLayer()])

    # Should also raise error in a deferred build mode
    with self.assertRaisesRegex(ValueError,
                                'should have a single output tensor'):
      keras.Sequential([MultiOutputLayer()])(np.zeros((10, 10)))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_layer_add_after_compile_deferred(self):
    model = keras.Sequential([keras.layers.Dense(3)])
    self.assertFalse(model.built)

    model.compile('adam', loss='mse')
    model.fit(np.random.random((1, 3)), np.random.random((1, 3)), batch_size=1)
    self.assertTrue(model.built)

    model.add(keras.layers.Dense(3))

    model.compile('adam', loss='mse')
    model.fit(np.random.random((1, 3)), np.random.random((1, 3)), batch_size=1)
    self.assertTrue(model.built)

  def test_sequential_layer_tracking(self):
    """Test that Sequential only tracks layers added in init or `.add`."""
    layer = keras.layers.Dense(1)
    model = keras.Sequential([layer])
    self.assertEqual(
        list(model._flatten_layers(include_self=False, recursive=False))[-1],  # pylint: disable=protected-access
        layer)

    model.a = [keras.layers.Dense(3)
               ]  # should not be added to the layers list.
    self.assertEqual(
        list(model._flatten_layers(include_self=False, recursive=False))[-1],  # pylint: disable=protected-access
        layer)

    layer2 = keras.layers.Dense(2)
    model.add(layer2)
    self.assertEqual(
        list(model._flatten_layers(include_self=False, recursive=False))[-1],  # pylint: disable=protected-access
        layer2)

    model.a = [keras.layers.Dense(3)
               ]  # should not be added to the layers list.
    self.assertEqual(
        list(model._flatten_layers(include_self=False, recursive=False))[-1],  # pylint: disable=protected-access
        layer2)

    model.pop()
    self.assertEqual(
        list(model._flatten_layers(include_self=False, recursive=False))[-1],  # pylint: disable=protected-access
        layer)

  def test_config_preserves_input_layer(self):
    model = keras.Sequential([
        keras.Input((None,), name='my_embedding_input', dtype='int32'),
        keras.layers.Embedding(32, 32),
        keras.layers.Dense(3),
    ])
    config = model.get_config()
    new_model = keras.Sequential.from_config(config)
    self.assertTrue(new_model.built)
    layers = list(
        new_model._flatten_layers(include_self=False, recursive=False))  # pylint: disable=protected-access
    self.assertEqual(layers[0].dtype, 'int32')
    self.assertEqual(layers[0].name, 'my_embedding_input')

  def test_name_unicity(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(3, name='specific_name'))
    with self.assertRaisesRegex(ValueError, 'should have unique names'):
      model.add(keras.layers.Dense(3, name='specific_name'))

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_tf_module_call(self):
    class MyModule(tf.Module):
      def __init__(self):
        self.v = tf.Variable(2.)

      def __call__(self, x):
        return self.v * x

    model = keras.Sequential()
    model.add(MyModule())
    model.compile('sgd', 'mse')
    x, y = np.ones((10, 1)), np.ones((10, 1))
    model.fit(x, y, batch_size=2)
    self.assertLen(model.trainable_variables, 1)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_tf_module_error(self):
    class MyModule(tf.Module):
      def __init__(self):
        self.v = tf.Variable(2.)

    model = keras.Sequential()
    with self.assertRaisesRegex(ValueError, 'is not defined'):
      model.add(MyModule())


class TestSequentialEagerIntegration(keras_parameterized.TestCase):
  def setUp(self):
    super().setUp()
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.ipu_model.compile_ipu_code = False
    cfg.ipu_model.tiles_per_ipu = 1
    cfg.configure_ipu_system()
    self._ipu_strategy = ipu.ipu_strategy.IPUStrategyV1()
    self._ipu_strategy_scope = self._ipu_strategy.scope()
    self._ipu_strategy_scope.__enter__()

  def tearDown(self):
    self._ipu_strategy_scope.__exit__(None, None, None)
    super().tearDown()

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_defun_on_call(self):
    # Check that one can subclass Sequential and place the `call` in a `defun`.
    class MySequential(keras.Sequential):
      def __init__(self, name=None):
        super().__init__(name=name)
        self.call = tf.function(self.call)

    model = MySequential()
    model.add(keras.layers.Dense(4, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  run_eagerly=testing_utils.should_run_eagerly())

    x = np.random.random((2, 6))
    y = np.random.random((2, 5))
    model.fit(x, y, epochs=1, batch_size=2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_build_before_fit(self):
    model = testing_utils.get_small_sequential_mlp(4, 5)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  run_eagerly=testing_utils.should_run_eagerly())

    model.build((None, 6))

    x = np.random.random((2, 6))
    y = np.random.random((2, 5))
    model.fit(x, y, epochs=1, batch_size=2)

  @keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                           always_skip_v1=True)
  def test_build_empty_network(self):
    x = np.random.random((2, 6))
    y = np.random.random((2, 5))
    model = keras.Sequential()

    # Make sure an empty sequential model can still work with build().
    model.build((None, 6))
    self.assertTrue(model.built)

    model.add(keras.layers.Dense(5, input_shape=(6,)))

    model.compile(loss='mse',
                  optimizer='rmsprop',
                  run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, batch_size=2)

    model.pop()
    self.assertFalse(model.built)

    model.build((None, 6))
    self.assertTrue(model.built)


if __name__ == '__main__':
  tf.test.main()
