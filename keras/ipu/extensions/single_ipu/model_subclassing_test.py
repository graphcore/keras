# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Model subclassing."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras import combinations
from keras import keras_parameterized
from keras import testing_utils
from keras.tests import model_subclassing_test_util as model_util
from tensorflow.python.framework import test_util
from tensorflow.python.training.tracking import data_structures
from tensorflow.python import ipu

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


@keras_parameterized.run_all_keras_modes(always_skip_eager=True,
                                         always_skip_v1=True)
class ModelSubclassingTest(keras_parameterized.TestCase):
  def setUp(self):
    super(ModelSubclassingTest, self).setUp()  # pylint: disable=super-with-arguments
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
    super(ModelSubclassingTest, self).tearDown()  # pylint: disable=super-with-arguments

  def test_build_with_dtype(self):
    class TypeCheckLayer(keras.layers.Layer):
      # This layer will error if called subsequently with different dtypes.
      def call(self, x):  # pylint: disable=arguments-differ
        if not hasattr(self, "x_dtype"):
          self.x_dtype = x.dtype
        elif x.dtype != self.x_dtype:
          raise TypeError(f"The dtype of `x` ({x.dtype.name}) does not match "
                          f"the dtype of this layer ({self.x_dtype.name}).")
        return x

    class TestModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.layer = TypeCheckLayer()

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.layer(inputs)

    batch_size = 1
    x_dtype = np.int32
    x_shape = [batch_size, 32]
    x = np.ones(x_shape, dtype=x_dtype)

    # Exception raised if build is called without specifying a dtype.
    model = TestModel()
    model.build(x_shape)
    with self.assertRaisesRegex(TypeError, "The dtype of `x`"):
      model.predict(x, batch_size=batch_size)

    # Runs successfully if build is called with a dtype.
    model = TestModel()
    x_input = keras.Input(shape=x_shape, dtype=x_dtype)
    model.build(x_input)
    model.predict(x, batch_size=batch_size)

  def test_custom_build(self):
    class DummyModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.uses_custom_build = False

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.dense1(inputs)

      def build(self, input_shape):
        self.uses_custom_build = True

    test_model = DummyModel()
    dummy_data = tf.ones((32, 50))
    test_model(dummy_data)
    self.assertTrue(test_model.uses_custom_build, 'Model should use user '
                    'defined build when called.')

  def test_attribute_conflict_error(self):
    class ModelWithProperty(keras.Model):  # pylint: disable=abstract-method
      @property
      def read_only(self):
        return 1.

    m = ModelWithProperty()
    with self.assertRaisesRegex(AttributeError, 'read_only'):
      m.read_only = 2.

  def test_custom_build_with_fit(self):
    class DummyModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Dense(10, activation='relu')

      def build(self, input_shape):
        self.layer2 = keras.layers.Dense(1, activation='relu')

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.layer2(self.layer1(inputs))

    model = DummyModel()
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2, epochs=2)
    self.assertLen(model.layers, 3)
    self.assertLen(model.trainable_variables, 4)

  def test_dataset_dict_with_fit(self):
    class MyModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(1)
        self.dense2 = keras.layers.Dense(1)
        self.add = keras.layers.Add()

      def call(self, x):  # pylint: disable=arguments-differ
        return self.add([self.dense1(x['a']), self.dense2(x['b'])])

    model = MyModel()
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())

    data = tf.data.Dataset.from_tensor_slices(({
        'a': np.ones((32, 10)),
        'b': np.ones((32, 20))
    }, np.ones((32, 1)))).batch(2, drop_remainder=True)
    model.fit(data, epochs=2)

  def test_invalid_input_shape_build(self):
    num_classes = 2
    input_dim = 50

    model = testing_utils.SmallSubclassMLP(num_hidden=32,
                                           num_classes=num_classes,
                                           use_dp=True,
                                           use_bn=True)

    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    with self.assertRaisesRegex(ValueError,
                                'input shape is not one of the valid types'):
      model.build(input_shape=tf.compat.v1.Dimension(input_dim))

  def test_embed_dtype_with_subclass_build(self):
    class Embedding(keras.layers.Layer):
      """An Embedding layer."""
      def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

      def build(self, _):
        self.embedding = self.add_variable(
            'embedding_kernel',
            shape=[self.vocab_size, self.embedding_dim],
            dtype=np.float32,
            initializer=tf.compat.v1.random_uniform_initializer(-0.1, 0.1),
            trainable=True)

      def call(self, x):  # pylint: disable=arguments-differ
        return tf.compat.v1.nn.embedding_lookup(self.embedding, x)

    class EmbedModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed1 = Embedding(vocab_size, embed_size)

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.embed1(inputs)

    model = EmbedModel(100, 20)
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    with self.assertRaisesRegex(
        ValueError, 'if your layers do not support float type inputs'):
      model.build(input_shape=(35, 20))

  def test_single_time_step_rnn_build(self):
    dim = 4
    timesteps = 1
    batch_input_shape = (None, timesteps, dim)
    units = 3

    class SimpleRNNModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.lstm = keras.layers.LSTM(units)

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.lstm(inputs)

    model = SimpleRNNModel()
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build(batch_input_shape)
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')
    model(tf.ones((32, timesteps, dim)))

  def test_single_io_subclass_build(self):
    num_classes = 2
    input_dim = 50
    batch_size = None

    model = testing_utils.SmallSubclassMLP(num_hidden=32,
                                           num_classes=num_classes,
                                           use_dp=True,
                                           use_bn=True)

    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build(input_shape=(batch_size, input_dim))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')
    model(tf.ones((32, input_dim)))

  def test_single_io_dimension_subclass_build(self):
    num_classes = 2
    input_dim = tf.compat.v1.Dimension(50)
    batch_size = tf.compat.v1.Dimension(None)

    model = testing_utils.SmallSubclassMLP(num_hidden=32,
                                           num_classes=num_classes,
                                           use_dp=True,
                                           use_bn=True)

    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build(input_shape=(batch_size, input_dim))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')
    model(tf.ones((32, input_dim)))

  def test_multidim_io_subclass_build(self):
    num_classes = 10
    # Input size, e.g. image
    batch_size = 32
    input_shape = (32, 32, 3)

    model = model_util.SimpleConvTestModel(num_classes)
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    batch_input_shape = (batch_size,) + input_shape
    model.build(input_shape=batch_input_shape)
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')

    model(tf.ones(batch_input_shape))

  def test_tensorshape_io_subclass_build(self):
    num_classes = 10
    # Input size, e.g. image
    batch_size = None
    input_shape = (32, 32, 3)

    model = model_util.SimpleConvTestModel(num_classes)
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build(input_shape=tf.TensorShape((batch_size,) + input_shape))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')

    model(tf.ones((32,) + input_shape))

  def test_subclass_save_model(self):
    num_classes = 10
    # Input size, e.g. image
    batch_size = None
    input_shape = (32, 32, 3)

    model = model_util.SimpleConvTestModel(num_classes)
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build(input_shape=tf.TensorShape((batch_size,) + input_shape))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')
    weights = model.get_weights()

    tf_format_name = os.path.join(self.get_temp_dir(), 'ckpt')
    model.save_weights(tf_format_name)
    if h5py is not None:
      hdf5_format_name = os.path.join(self.get_temp_dir(), 'weights.h5')
      model.save_weights(hdf5_format_name)

    model = model_util.SimpleConvTestModel(num_classes)
    model.build(input_shape=tf.TensorShape((batch_size,) + input_shape))
    if h5py is not None:
      model.load_weights(hdf5_format_name)
      self.assertAllClose(weights, model.get_weights())
    model.load_weights(tf_format_name)
    self.assertAllClose(weights, model.get_weights())

  def test_multi_io_subclass_build(self):
    batch_size = None
    num_samples = 1000
    input_dim = 50
    model = model_util.get_multi_io_subclass_model()
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    batch_input_shape = tf.TensorShape((batch_size, input_dim))
    model.build(input_shape=[batch_input_shape, batch_input_shape])
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')
    x1 = tf.ones((num_samples, input_dim))
    x2 = tf.ones((num_samples, input_dim))
    model([x1, x2])

  def test_summary(self):
    class ToString:
      def __init__(self):
        self.contents = ''

      def __call__(self, msg):
        self.contents += msg + '\n'

    # Single-io
    model = testing_utils.SmallSubclassMLP(num_hidden=32,
                                           num_classes=4,
                                           use_bn=True,
                                           use_dp=True)
    model(np.ones((3, 4)))  # need to build model first
    print_fn = ToString()
    model.summary(print_fn=print_fn)
    self.assertIn('Trainable params: 356', print_fn.contents)

    # Multi-io
    model = model_util.get_multi_io_subclass_model(num_classes=(5, 6),
                                                   use_bn=True,
                                                   use_dp=True)
    model([np.ones((3, 4)), np.ones((3, 4))])  # need to build model first
    print_fn = ToString()
    model.summary(print_fn=print_fn)
    self.assertIn('Trainable params: 587', print_fn.contents)

    # Single-io with unused layer
    model = testing_utils.SmallSubclassMLP(num_hidden=32,
                                           num_classes=4,
                                           use_bn=True,
                                           use_dp=True)
    model.unused_layer = keras.layers.Dense(10)
    model(np.ones((3, 4)))  # need to build model first
    print_fn = ToString()
    model.summary(print_fn=print_fn)
    self.assertIn('Trainable params: 356', print_fn.contents)
    self.assertIn('0 (unused)', print_fn.contents)

  def test_no_dependency(self):
    class Foo(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.isdep = keras.layers.Dense(1)
        self.notdep = data_structures.NoDependency(keras.layers.Dense(2))
        self.notdep_var = data_structures.NoDependency(
            tf.Variable(1., name='notdep_var'))

    m = Foo()
    self.assertEqual([m.isdep, m.notdep], m.layers)
    self.assertEqual(1, len(m._checkpoint_dependencies))  # pylint: disable=protected-access
    self.assertIs(m.isdep, m._checkpoint_dependencies[0].ref)  # pylint: disable=protected-access
    self.assertEqual('notdep_var:0', m.notdep_var.name)

  def test_extra_variable(self):
    class ExtraVar(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(1)
        self.var = tf.Variable(1.)
        self.not_trainable_var = tf.Variable(2., trainable=False)

      def call(self, inputs):  # pylint: disable=arguments-differ
        return self.dense(inputs + self.var)

    m = ExtraVar()
    self.assertTrue(m.trainable)
    self.assertEqual([m.dense], m.layers)
    self.assertEqual([m.var, m.not_trainable_var], m.variables)
    self.assertEqual([m.var], m.trainable_variables)
    self.assertEqual([m.not_trainable_var], m.non_trainable_variables)
    self.assertLen(m.get_weights(), 2)
    m.trainable = False
    self.assertEqual([m.var, m.not_trainable_var], m.variables)
    self.assertEqual([], m.trainable_variables)
    self.assertEqual([m.var, m.not_trainable_var], m.non_trainable_variables)
    self.assertLen(m.get_weights(), 2)
    m.trainable = True

    m(tf.ones([1, 1]))

    self.assertEqual([m.dense.kernel, m.dense.bias], m.dense.variables)
    self.assertEqual([m.dense.kernel, m.dense.bias], m.dense.weights)

    self.assertLen(m.get_weights(), 4)
    self.assertEqual(
        [m.dense.kernel, m.dense.bias, m.var, m.not_trainable_var],
        m.variables)
    self.assertEqual([m.dense.kernel, m.dense.bias, m.var],
                     m.trainable_variables)
    self.assertEqual([m.not_trainable_var], m.non_trainable_variables)

    m.dense.trainable = False
    self.assertEqual(
        [m.dense.kernel, m.dense.bias, m.var, m.not_trainable_var],
        m.variables)
    self.assertEqual([m.var], m.trainable_variables)
    self.assertEqual([m.dense.kernel, m.dense.bias, m.not_trainable_var],
                     m.non_trainable_variables)
    self.assertLen(m.get_weights(), 4)

  def test_add_weight_in_model(self):
    class MyModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.b = self.add_weight('bias', (10,))
        self.c = self.add_weight('bias2', (10,), trainable=False)

      def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs + self.b + self.c

    x = tf.convert_to_tensor(np.ones((10, 10), 'float32'))
    model = MyModel()
    model(x)
    self.assertEqual(1, len(model.trainable_weights))
    self.assertEqual(1, len(model.non_trainable_weights))
    self.assertEqual(2, len(model.weights))

    class MyModelCustomBuild(keras.Model):  # pylint: disable=abstract-method
      def build(self, input_shape):
        self.b = self.add_weight('bias', (10,))
        self.c = self.add_weight('bias2', (10,), trainable=False)

      def call(self, inputs):  # pylint: disable=arguments-differ
        return inputs + self.b + self.c

    x = tf.convert_to_tensor(np.ones((10, 10), 'float32'))
    model = MyModelCustomBuild()
    model(x)
    self.assertEqual(1, len(model.trainable_weights))
    self.assertEqual(1, len(model.non_trainable_weights))
    self.assertEqual(2, len(model.weights))

  def test_add_update_in_model(self):
    class MyModel(keras.Model):  # pylint: disable=abstract-method
      def __init__(self):
        super().__init__()
        self.b = self.add_weight('bias', (10,))
        self.c = self.add_weight('bias2', (10,))

      def call(self, inputs):  # pylint: disable=arguments-differ
        # Unconditional
        self.add_update(self.b.assign(self.b * 2))
        # Conditional
        self.add_update(self.c.assign(inputs[1, :]))
        return inputs + self.b + self.c

    x = tf.convert_to_tensor(np.ones((10, 10), 'float32'))
    model = MyModel()
    model(x)

    if tf.executing_eagerly():
      self.assertEqual(0, len(model.updates))
    else:
      self.assertEqual(2, len(model.updates))


@combinations.generate(combinations.combine(mode=['graph']))
class CustomCallSignatureTests(tf.test.TestCase, parameterized.TestCase):
  def setUp(self):
    super(CustomCallSignatureTests, self).setUp()  # pylint: disable=super-with-arguments
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
    super(CustomCallSignatureTests, self).tearDown()  # pylint: disable=super-with-arguments

  def test_no_inputs_in_signature(self):
    model = model_util.CustomCallModel()
    first = tf.ones([2, 3])
    second = tf.ones([2, 5])
    output = model(first, second)
    self.evaluate([v.initializer for v in model.variables])
    expected_output = self.evaluate(model.dense1(first) + model.dense2(second))
    self.assertAllClose(expected_output, self.evaluate(output))
    output = model(first, second, fiddle_with_output='yes')
    self.assertAllClose(10. * expected_output, self.evaluate(output))
    output = model(first, second=second, training=False)
    self.assertAllClose(expected_output, self.evaluate(output))

  def test_training_args_call_build(self):
    input_dim = 2

    model = model_util.TrainingNoDefaultModel()
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build((None, input_dim))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')

  def test_training_and_mask_args_call_build(self):
    input_dim = 2

    model = model_util.TrainingMaskingModel()
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    model.build((None, input_dim))
    self.assertTrue(model.weights, ('Model should have weights now that it '
                                    'has been properly built.'))
    self.assertTrue(model.built,
                    'Model should be built after calling `build`.')

  def test_custom_call_kwargs_and_build(self):
    first_input_shape = (2, 3)
    second_input_shape = (2, 5)

    model = model_util.CustomCallModel()
    self.assertFalse(model.built, 'Model should not have been built')
    self.assertFalse(model.weights, ('Model should have no weights since it '
                                     'has not been built.'))
    with self.assertRaisesRegex(
        ValueError, 'cannot build your model if it has positional'):
      model.build(input_shape=[first_input_shape, second_input_shape])

  def test_kwargs_in_signature(self):
    class HasKwargs(keras.Model):  # pylint: disable=abstract-method
      def call(self, x, y=3, **kwargs):  # pylint: disable=arguments-differ,unused-argument
        return x

    model = HasKwargs()
    arg = tf.ones([1])
    model(arg, a=3)
    if not tf.executing_eagerly():
      self.assertLen(model.inputs, 1)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def test_training_no_default(self):
    if not tf.executing_eagerly():
      return
    model = model_util.TrainingNoDefaultModel()
    arg = tf.ones([1, 1])
    model(arg, True)

  def test_positional_arg_in_call(self):
    class ModelWithPositionalArgs(keras.Model):  # pylint: disable=abstract-method
      def call(self, x, x2, x3=None):  # pylint: disable=arguments-differ,unused-argument
        return x + x2

    x = np.ones((10, 1))
    y = np.ones((10, 1))
    m = ModelWithPositionalArgs()
    m.compile('sgd', 'mse')
    with self.assertRaisesRegex(ValueError, r'Models passed to `fit`'):
      m.fit(x, y, batch_size=2)
    with self.assertRaisesRegex(ValueError, r'Models passed to `evaluate`'):
      m.evaluate(x, y, batch_size=2)
    with self.assertRaisesRegex(ValueError, r'Models passed to `predict`'):
      m.predict(x, batch_size=2)
    with self.assertRaisesRegex(ValueError,
                                r'Models passed to `train_on_batch`'):
      m.train_on_batch(x, y)
    with self.assertRaisesRegex(ValueError,
                                r'Models passed to `test_on_batch`'):
      m.test_on_batch(x, y)
    with self.assertRaisesRegex(ValueError,
                                r'Models passed to `predict_on_batch`'):
      m.predict_on_batch(x)


if __name__ == '__main__':
  tf.test.main()
