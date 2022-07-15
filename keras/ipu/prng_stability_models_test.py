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

import keras
from keras import testing_utils
from keras.utils import np_utils
from keras.optimizer_v2 import adam
from keras.optimizer_v2 import gradient_descent
from keras.optimizer_v2 import optimizer_v2
from keras.datasets import imdb
from keras.datasets import mnist
from keras.layers.preprocessing import normalization
from tensorflow.python import ipu
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.client import session
from tensorflow.python.framework import test_util
from tensorflow.python.training.optimizer import Optimizer


# Optimizer wrapper that captures intermediate kernel/weight values into the
# given outfeed. This lets us compare the value of weights across replicas.
class KernelLoggingOptimizer(Optimizer):
  def __init__(self, outfeed_queue, wrapped_optimizer, model=None):
    super().__init__(False, "KernelLoggingOptimizer")

    self._wrapped_optimizer = wrapped_optimizer
    self._outfeed_queue = outfeed_queue
    self._model = model

    self._using_v2_optimizer = isinstance(self._wrapped_optimizer,
                                          optimizer_v2.OptimizerV2)

  def compute_gradients(self, loss, var_list=None, **kwargs):  # pylint: disable=arguments-differ,unused-argument
    if isinstance(self._wrapped_optimizer, optimizer_v2.OptimizerV2):
      grads = self._wrapped_optimizer.get_gradients(
          loss, self._model.trainable_weights)
      grads_and_vars = list(zip(grads, self._model.trainable_weights))
    else:
      grads_and_vars = self._wrapped_optimizer.compute_gradients(
          loss, var_list=var_list, **kwargs)

    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    kernels = []
    for _, var in list(reversed(grads_and_vars)):
      if "kernel" in var.name:
        kernels.append(var)
    outfeed = self._outfeed_queue.enqueue(kernels)

    with tf.control_dependencies([outfeed]):
      if self._using_v2_optimizer:
        return self._wrapped_optimizer.apply_gradients(grads_and_vars)

      return self._wrapped_optimizer.apply_gradients(grads_and_vars,
                                                     global_step, name)

  def _apply_dense(self, grad, var):
    return self._wrapped_optimizer._apply_dense(grad, var)  # pylint: disable=protected-access

  def _resource_apply_dense(self, grad, handle):
    return self._wrapped_optimizer._resource_apply_dense(grad, handle)  # pylint: disable=protected-access

  def _apply_sparse(self, grad, var):
    return self._wrapped_optimizer._apply_sparse(grad, var)  # pylint: disable=protected-access

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._wrapped_optimizer._resource_apply_sparse(  # pylint: disable=protected-access
        grad, handle, indices)

  def get_name(self):
    return self._wrapped_optimizer.get_name()

  def get_slot(self, var, name):
    return self._wrapped_optimizer.get_slot(var, name)

  def get_slot_names(self):
    return self._wrapped_optimizer.get_slot_names()

  def variables(self):
    return self._wrapped_optimizer.variables()


def assert_all_weights_replica_identical(test_case, var, replica_count):
  """ Utility for checking that the values logged in KernelLoggingOptimizer are
  replica identical."""
  test_case.assertGreater(len(var), 0, "No weights for variable.")

  for weights in var:
    test_case.assertEqual(
        len(weights), replica_count,
        f"Expected {replica_count} weights but have {len(weights)}.")

    replica1 = weights[0]
    equal_weights = map(lambda x: (replica1 == x).all(), weights[1:])  # pylint: disable=cell-var-from-loop
    test_case.assertTrue(all(equal_weights),
                         "Expected all weights to be replica identical.")


ipu_count = 4


def create_test_config():
  cfg = ipu.config.IPUConfig()
  cfg.auto_select_ipus = ipu_count
  tu.add_hw_ci_connection_options(cfg)
  return cfg


def stochastic_rounding_modes():
  test_modes = (ipu.config.StochasticRoundingBehaviour.ON,
                ipu.config.StochasticRoundingBehaviour.REPLICA_IDENTICAL_ONLY)
  return tuple(
      ("_StochasticRounding_" + mode.name, mode) for mode in test_modes)


# This test is intended to verify that we get the same weight values produced on each replica
# when running simple models with the experimental.enable_prng_stability flag enabled.
@parameterized.named_parameters(*stochastic_rounding_modes())
class PrngStabilityTest(tf.test.TestCase):
  @staticmethod
  def make_mnist_dataset(y_as_categorical=False):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0
    x_train = x_train[..., np.newaxis]

    x_train = x_train.astype('float32')
    y_train = np_utils.to_categorical(
        y_train) if y_as_categorical else y_train.astype('float32')
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)

    return train_ds.repeat()

  @staticmethod
  def make_imdb_dataset(max_features, minibatch_size):
    (x_train, y_train), (_, _) = imdb.load_data(num_words=max_features)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat()
    dataset = dataset.map(lambda x, y: (x, tf.cast(y, np.int32)))
    return dataset.batch(minibatch_size, drop_remainder=True)

  def setUpTest(self, stochastic_rounding_mode):
    cfg = create_test_config()
    cfg.floating_point_behaviour.esr = stochastic_rounding_mode
    cfg.experimental.enable_prng_stability = True
    cfg.configure_ipu_system()

    keras.backend.set_floatx('float16')

    self.outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  def createLoggingOptimizer(self, wrapped_optimizer, model):
    return KernelLoggingOptimizer(self.outfeed, wrapped_optimizer, model)

  def assertAllWeightsReplicaIdentical(self, expected_replicas=2):
    out_queue = self.outfeed.dequeue()
    self.assertTrue(out_queue, "Expected some logged variables.")

    for var in out_queue:
      var = var.numpy()
      assert_all_weights_replica_identical(self, var, expected_replicas)

  @tu.test_uses_ipus(num_ipus=ipu_count)
  @testing_utils.run_v2_only
  def testLinearRegression(self, stochastic_rounding_mode):

    self.setUpTest(stochastic_rounding_mode)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      np.random.seed(1234)
      horsepower = np.random.rand(320, 1).astype(np.float32)
      mpg = np.random.rand(320, 1).astype(np.float32)

      normalizer = normalization.Normalization(input_shape=[
          1,
      ], axis=None)

      model = keras.Sequential([normalizer, keras.layers.Dense(units=1)])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.set_pipeline_stage_assignment([0, 1])

      model.compile(optimizer=self.createLoggingOptimizer(
          adam.Adam(learning_rate=0.1), model),
                    loss='mean_absolute_error',
                    steps_per_execution=64)
      model.fit(horsepower, mpg, epochs=3, steps_per_epoch=64)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=ipu_count)
  @testing_utils.run_v2_only
  def testImdbRnn(self, stochastic_rounding_mode):
    self.setUpTest(stochastic_rounding_mode)

    max_features = 10000
    minibatch_size = 32

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = self.make_imdb_dataset(max_features, minibatch_size)

      model = keras.Sequential([
          keras.layers.Input(shape=(80),
                             dtype=np.int32,
                             batch_size=minibatch_size),
          keras.layers.Embedding(max_features, 128),
          keras.layers.Bidirectional(keras.layers.LSTM(64)),
          keras.layers.Dense(64, activation='relu'),
          keras.layers.Dense(1)
      ])
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=16)
      model.set_pipeline_stage_assignment([0, 0, 1, 1])

      model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    optimizer=self.createLoggingOptimizer(
                        adam.Adam(1e-4), model),
                    steps_per_execution=64)
      model.fit(dataset, steps_per_epoch=64, epochs=3)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=ipu_count)
  @testing_utils.run_v2_only
  def testImdb(self, stochastic_rounding_mode):
    self.setUpTest(stochastic_rounding_mode)

    max_features = 20000
    minibatch_size = 32
    gradient_accumulation_steps_per_replica = 16

    def get_model():
      input_layer = keras.layers.Input(shape=(80),
                                       dtype=np.int32,
                                       batch_size=minibatch_size)
      with keras.ipu.PipelineStage(0):
        x = keras.layers.Embedding(max_features, 128)(input_layer)

      with keras.ipu.PipelineStage(1):
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(1)(x)

      return keras.Model(input_layer, x)

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = self.make_imdb_dataset(max_features, minibatch_size)

      model = get_model()
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=
                                   gradient_accumulation_steps_per_replica)

      model.compile(steps_per_execution=384,
                    loss='binary_crossentropy',
                    optimizer=self.createLoggingOptimizer(
                        adam.Adam(0.005), model))
      model.fit(dataset, steps_per_epoch=768, epochs=2)

      self.assertAllWeightsReplicaIdentical()

  @tu.test_uses_ipus(num_ipus=ipu_count)
  @testing_utils.run_v2_only
  def testMnistCnn(self, stochastic_rounding_mode):
    self.setUpTest(stochastic_rounding_mode)

    def create_model():
      return keras.Sequential([
          keras.layers.Conv2D(64,
                              kernel_size=3,
                              activation="relu",
                              input_shape=(28, 28, 1)),
          keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
          keras.layers.Flatten(),
          keras.layers.Dense(10, activation='softmax')
      ])

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = self.make_mnist_dataset(y_as_categorical=True)

      steps_per_epoch = 4
      model = create_model()
      model.compile(loss='categorical_crossentropy',
                    optimizer=self.createLoggingOptimizer(
                        gradient_descent.SGD(), model),
                    steps_per_execution=steps_per_epoch)
      model.fit(dataset, epochs=3, steps_per_epoch=steps_per_epoch)

      self.assertAllWeightsReplicaIdentical(expected_replicas=ipu_count)

  @testing_utils.run_v2_only
  @tu.test_uses_ipus(num_ipus=ipu_count)
  def testMnist(self, stochastic_rounding_mode):
    self.setUpTest(stochastic_rounding_mode)

    def create_model():
      return keras.Sequential([
          keras.layers.Flatten(),
          keras.layers.Dense(128, activation='relu'),
          keras.layers.Dense(10, activation='softmax')
      ])

    strategy = ipu.ipu_strategy.IPUStrategy()
    with strategy.scope():
      dataset = self.make_mnist_dataset()

      model = create_model()
      model.set_pipelining_options(gradient_accumulation_steps_per_replica=4)
      model.set_pipeline_stage_assignment([0, 1, 1])

      steps_per_epoch = 16
      model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=self.createLoggingOptimizer(
                        gradient_descent.SGD(), model),
                    steps_per_execution=steps_per_epoch)
      model.fit(dataset, epochs=3, steps_per_epoch=steps_per_epoch)

      self.assertAllWeightsReplicaIdentical()


if __name__ == "__main__":
  tf.test.main()
