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
import numpy as np

import popdist
import popdist.tensorflow

import tensorflow.compat.v2 as tf

import keras
from keras import Model, Input
from keras import layers
from keras.engine import data_adapter
from keras.optimizer_v2 import gradient_descent
from tensorflow.python import ipu
from tensorflow.python.ipu.distributed.host_collective_ops import broadcast
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import test_utils as tu
from tensorflow.python.ipu.distributed import popdist_strategy
from tensorflow.python.framework import test_util as tf_test_util


def simple_model():
  random_seed = 1234

  np.random.seed(random_seed)
  tf_test_util.random_seed.set_seed(random_seed)

  bias = keras.initializers.Constant(value=popdist.getInstanceIndex())

  inputs = Input(shape=(32,))
  outputs = layers.Dense(1, bias_initializer=bias, name='test_bias')(inputs)

  return Model(inputs, outputs)


def test_dataset(length=None, batch_size=1, x_val=1.0, y_val=0.2):
  constant_d = tf.constant(x_val, shape=[32])
  constant_l = tf.constant(y_val, shape=[1])

  ds = tf.data.Dataset.from_tensors((constant_d, constant_l))
  ds = ds.repeat(length)
  ds = ds.shard(num_shards=popdist.getNumInstances(),
                index=popdist.getInstanceIndex())
  ds = ds.batch(batch_size, drop_remainder=True)

  return ds


class DistributedTF2Test(tf.test.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.strategy = popdist_strategy.PopDistStrategy()

  def assert_all_instances_equal(self, local_value, name=None):
    # Assert that the current instance has the same value as the root instance.
    local_tensor = tf.constant(local_value)
    root_tensor = broadcast(local_tensor, root_rank=0)
    np.testing.assert_equal(local_value, root_tensor.numpy(), name)

  def assert_all_instances_not_equal(self, local_value):
    local_tensor = tf.constant(local_value)
    root_tensor = broadcast(local_tensor, root_rank=0)

    if popdist.getInstanceIndex() == 0:
      return

    assert not np.equal(local_value, root_tensor.numpy()).any()

  def prepare_model(self):
    # Make sure we have different parameters on each index
    bias = keras.initializers.Constant(value=popdist.getInstanceIndex())

    return keras.models.Sequential([
        keras.layers.Conv2D(4,
                            3,
                            activation='relu',
                            bias_initializer=bias,
                            name='test_bias'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
    ])

  def prepare_dataset(self):
    def generator():
      for _ in range(100):
        yield np.random.rand(4, 4, 1), np.random.randint(1, 2, size=1)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((4, 4, 1), (1,)),
    )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.OFF
    dataset = dataset.with_options(options)

    dataset = dataset.shard(num_shards=popdist.getNumInstances(),
                            index=popdist.getInstanceIndex())
    dataset = dataset.batch(10, drop_remainder=True)

    return dataset

  @tu.test_uses_ipus(num_ipus=8)
  def test_tf2_distributed_ipu_strategy(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    steps_to_run = 10
    batch_size = 8

    strategy = ipu_strategy.IPUStrategy()

    with strategy.scope():
      dataset = test_dataset(popdist.getNumInstances() * batch_size *
                             steps_to_run,
                             batch_size=batch_size)
      model = simple_model()
      optimizer = gradient_descent.SGD(learning_rate=0.01)
      loss_fn = keras.losses.MeanSquaredError()

      model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    steps_per_execution=steps_to_run)

      # Build the model separately so we can assert that the biases are
      # broadcasted properly before training.
      model.build((1, 32))

      layer = model.get_layer(name='test_bias')
      self.assert_all_instances_not_equal(layer.get_weights()[1])

      history = model.fit(dataset, epochs=1)

      # Make sure the losses and weights are not equal
      self.assert_all_instances_not_equal(history.history['loss'])
      self.assert_all_instances_not_equal(layer.get_weights()[1])

  @tu.test_uses_ipus(num_ipus=8)
  def test_tf2_distributed_popdist_strategy(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    steps_to_run = 10
    batch_size = 8

    with self.strategy.scope():
      dataset = test_dataset(popdist.getNumTotalReplicas() * batch_size *
                             steps_to_run,
                             batch_size=batch_size)
      dataset = dataset.shard(num_shards=popdist.getNumInstances(),
                              index=popdist.getInstanceIndex())
      steps_per_execution = len(dataset) // popdist.getNumLocalReplicas()
      model = simple_model()
      optimizer = gradient_descent.SGD(learning_rate=0.01)
      loss_fn = keras.losses.MeanSquaredError()

      model.compile(optimizer=optimizer,
                    loss=loss_fn,
                    steps_per_execution=steps_per_execution)

      # Build the model separately so we can assert that the biases are
      # broadcasted properly before training.
      model.build((1, 32))

      layer = model.get_layer(name='test_bias')
      self.assert_all_instances_equal(layer.get_weights()[1])

      history = model.fit(dataset, epochs=1)

      # Make sure the losses and weights are identical as we reduce over all
      # IPUs
      self.assert_all_instances_equal(history.history['loss'])

      for v in model.trainable_variables:
        self.assert_all_instances_equal(v)

  @tu.test_uses_ipus(num_ipus=8)
  def test_single_multi_replica_training_step(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    with self.strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      optimizer = gradient_descent.SGD(learning_rate=learning_rate)

      w = tf.Variable(initial_w)

      @tf.function(jit_compile=True)
      def step_fn(x):
        with tf.GradientTape() as tape:
          loss = w * x
        optimizer.minimize(loss, var_list=[w], tape=tape)

        return loss

      @tf.function(jit_compile=True)
      def step_fn_wrapper(x):
        per_replica_loss = self.strategy.run(step_fn, args=[x])
        loss_reduced = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_loss)

        return loss_reduced

      num_replicas = popdist.getNumTotalReplicas()
      reference_w = initial_w

      for x in range(10):
        self.assertEqual(reference_w, w.numpy())
        with tf.device("/device:IPU:0"):
          loss_final = step_fn_wrapper(tf.constant(tf.cast(x, tf.float32)))
        self.assertEqual(num_replicas * reference_w * x, loss_final)

        # L(x) = num_replicas * w * x
        # dL(x)/dw = num_replicas * x
        # w := w - learning_rate * num_replicas * x
        reference_w -= learning_rate * num_replicas * x

  @tu.test_uses_ipus(num_ipus=8)
  def test_single_multi_replica_training_step_keras(self):
    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    with self.strategy.scope():
      learning_rate = 0.5
      initial_w = 2.0
      model = keras.Sequential([
          keras.layers.Dense(
              1,
              kernel_initializer=keras.initializers.Constant(initial_w),
              use_bias=False)
      ])
      optimizer = gradient_descent.SGD(learning_rate=learning_rate)

      @tf.function(jit_compile=True)
      def loss_fn(_, y_pred):
        return y_pred

      reference_w = initial_w
      model.compile(loss=loss_fn, optimizer=optimizer, steps_per_execution=1)
      model.build((1, 1))

      for x in range(10):
        self.assertEqual(reference_w,
                         model.trainable_variables[0][0][0].numpy())
        history = model.fit(
            np.array([[x]], np.float32).repeat(popdist.getNumLocalReplicas(),
                                               axis=0),
            np.array([[x]], np.float32).repeat(popdist.getNumLocalReplicas(),
                                               axis=0),
            batch_size=1,
            epochs=1)
        self.assertEqual(reference_w * x, history.history['loss'][0])

        # L(x) = w * x
        # dL(x)/dw = x
        # w := w - learning_rate * x
        reference_w -= learning_rate * x

  @tu.test_uses_ipus(num_ipus=8)
  def test_single_training_step_equal_in_tf_and_keras(self):
    # This test verifies that a training loop in raw TensorFlow and Keras yield
    # the same losses, gradients and weight updates.

    def initialize_model_with_seed():
      # Make sure we initialize the kernels in a reproducible manner, create
      # an initializer with a constant seed.
      initializer = keras.initializers.GlorotNormal(seed=1234)

      return keras.models.Sequential([
          keras.layers.Conv2D(4,
                              3,
                              kernel_initializer=initializer,
                              use_bias=False,
                              activation='relu'),
          keras.layers.Flatten(),
          keras.layers.Dense(2, kernel_initializer=initializer,
                             use_bias=False),
      ])

    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    with self.strategy.scope():
      learning_rate = 0.01

      model_tf = initialize_model_with_seed()
      model_keras = initialize_model_with_seed()
      optimizer = gradient_descent.SGD(learning_rate=learning_rate)

      @tf.function(jit_compile=True)
      def step_fn_tf(x, y):
        with tf.GradientTape() as tape:
          output = model_tf(x)
          loss = keras.losses.sparse_categorical_crossentropy(y_true=y,
                                                              y_pred=output,
                                                              from_logits=True)
          loss = tf.nn.compute_average_loss(
              loss, global_batch_size=popdist.getNumTotalReplicas())
        optimizer.minimize(loss,
                           var_list=model_tf.trainable_variables,
                           tape=tape)

        return loss

      @tf.function(jit_compile=True)
      def run_training_step_tf(x, y):
        per_replica_loss = self.strategy.run(step_fn_tf, args=[x, y])
        loss_reduced = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_loss)

        return loss_reduced

      loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      model_keras.compile(optimizer=optimizer,
                          loss=loss_fn,
                          steps_per_execution=1)

      def run_training_step_keras(x, y):
        history = model_keras.fit(x, y, batch_size=1, epochs=1)

        return history.history['loss'][0]

      # First generate some random data using numpy, so we can reuse the same
      # data for both TF and Keras in order to force reproducibility.
      input_sample = np.random.uniform(0, 1, (1, 4, 4, 1))
      output_sample = np.random.randint(1, 2, size=1)

      # The Keras `.fit()` API requires the input to be replicated `num_replica`
      # times.
      x_tf = tf.constant(tf.cast(input_sample, tf.float32))
      y_tf = tf.constant(tf.cast(output_sample, tf.float32))
      x_keras = tf.constant(
          tf.cast(
              np.repeat(input_sample, popdist.getNumLocalReplicas(), axis=0),
              tf.float32))
      y_keras = tf.constant(
          tf.cast(
              np.repeat(output_sample, popdist.getNumLocalReplicas(), axis=0),
              tf.float32))

      # First test whether a single distributed training step yields the same
      # loss values for both TensorFlow and Keras.
      with tf.device("/device:IPU:0"):
        loss_final_tf = run_training_step_tf(x_tf, y_tf)
        loss_final_keras = run_training_step_keras(x_keras, y_keras)

      self.assertEqual(loss_final_tf, loss_final_keras)

      # Assert that both models have the same weights after the first backwards
      # pass.
      for i, _ in enumerate(model_tf.trainable_variables):
        np.testing.assert_equal(model_tf.trainable_variables[i].numpy(),
                                model_keras.trainable_variables[i].numpy())

      @tf.function(jit_compile=True)
      def step_fn_eval_tf(x, y):
        output = model_tf(x, training=False)
        loss = keras.losses.sparse_categorical_crossentropy(y_true=y,
                                                            y_pred=output,
                                                            from_logits=True)
        loss = tf.nn.compute_average_loss(
            loss, global_batch_size=popdist.getNumTotalReplicas())

        return loss

      @tf.function(jit_compile=True)
      def run_eval_step_tf(x, y):
        per_replica_loss = self.strategy.run(step_fn_eval_tf, args=[x, y])
        loss_reduced = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_loss)

        return loss_reduced

      def run_eval_step_keras(x, y):
        scores = model_keras.evaluate(x,
                                      y,
                                      steps=popdist.getNumTotalReplicas())

        return scores

      x_keras_eval = tf.constant(
          tf.cast(
              np.repeat(input_sample, popdist.getNumTotalReplicas(), axis=0),
              tf.float32))
      y_keras_eval = tf.constant(
          tf.cast(
              np.repeat(output_sample, popdist.getNumTotalReplicas(), axis=0),
              tf.float32))

      with tf.device("/device:IPU:0"):
        val_loss_final_tf = run_eval_step_tf(x_tf, y_tf)
        val_loss_final_keras = run_eval_step_keras(x_keras_eval, y_keras_eval)

      self.assertEqual(val_loss_final_tf, val_loss_final_keras)

  def single_training_step_equal_keras(self):
    # This test verifies that a training loop in raw TensorFlow 1 and Keras
    # yield the same losses, gradients and weight updates.
    num_iterations = popdist.getNumLocalReplicas()
    learning_rate = 0.5
    batch_size = 2

    def initialize_model_with_seed():
      # Make sure we initialize the kernels in a reproducible manner, create
      # an initializer with a constant seed.
      initializer = keras.initializers.GlorotNormal(seed=1234)

      return keras.models.Sequential([
          keras.layers.Flatten(),
          keras.layers.Dense(2, kernel_initializer=initializer,
                             use_bias=False),
      ])

    # Instantiate a custom optimizer that allows us to keep track of the
    # gradients in `model.fit()`.
    class ModelKeras(keras.Sequential):
      def __init__(self, layer_list):
        super().__init__(layer_list)
        self.outfeed_queue_gradients = ipu.ipu_outfeed_queue.IPUOutfeedQueue()
        self.outfeed_queue_losses = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

      def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
          y_pred = self(x, training=True)
          loss = self.compiled_loss(y,
                                    y_pred,
                                    regularization_losses=self.losses)

        # Save the loss to an outfeed queue so we can use it later.
        self.outfeed_queue_losses.enqueue(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Save the gradients to an outfeed queue so we can use them later.
        self.outfeed_queue_gradients.enqueue(gradients)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    # First generate some random data using numpy, so we can reuse the same
    # data for both TF1 and Keras in order to force reproducibility.
    np.random.seed(1234)
    input_sample = np.random.uniform(
        0, 1, (batch_size * popdist.getNumLocalReplicas(), 4, 4, 1)).astype(
            np.float32)
    output_sample = np.random.randint(
        1, 2,
        size=batch_size * popdist.getNumLocalReplicas()).astype(np.float32)

    config = ipu.config.IPUConfig()
    popdist.tensorflow.set_ipu_config(config, ipus_per_replica=1)
    config.configure_ipu_system()

    with self.strategy.scope():
      dataset = tf.data.Dataset.from_tensor_slices(
          (input_sample, output_sample))
      dataset = dataset.repeat()
      dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

      optimizer = gradient_descent.SGD(learning_rate=learning_rate)
      loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

      model_keras = ModelKeras(initialize_model_with_seed())
      model_keras.compile(optimizer=optimizer,
                          loss=loss_fn,
                          steps_per_execution=1)
      model_keras.build((1, 4, 4, 1))
      history = model_keras.fit(dataset,
                                steps_per_epoch=num_iterations,
                                epochs=1)
      loss = history.history['loss'][0]

    # Extract weights to numpy now we are still in eager mode, this will not be
    # possible afterwards.
    weights = [var.numpy() for var in model_keras.trainable_variables]
    gradients = [
        g.numpy() for g in model_keras.outfeed_queue_gradients.dequeue()
    ]
    losses = [l.numpy() for l in model_keras.outfeed_queue_losses.dequeue()]

    return loss, gradients, losses, weights


if __name__ == "__main__":
  tf.test.main()
