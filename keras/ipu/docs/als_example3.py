# MNIST training with ALS.
# Uses gradients w.r.t activations and only collects statistics
# every 5 iterations.

import tensorflow as tf
from tensorflow.python import ipu

import keras
from keras.datasets import mnist
from keras.ipu.optimizers import ALSOptimizer
from keras.ipu.layers import CaptureActivationGradients
from keras.optimizer_v2 import gradient_descent

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()


# Create a simple model.
def create_model():
  input_layer = keras.layers.Input((28, 28, 1))

  with keras.ipu.PipelineStage(0):
    x = CaptureActivationGradients(
        keras.layers.Conv2D(32,
                            dtype=tf.float16,
                            kernel_size=(3, 3),
                            activation="relu"))(input_layer)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = CaptureActivationGradients(
        keras.layers.Conv2D(64,
                            dtype=tf.float16,
                            kernel_size=(3, 3),
                            activation="relu"))(x)

  with keras.ipu.PipelineStage(1):
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = CaptureActivationGradients(keras.layers.Dense(10,
                                                      activation="softmax"))(x)

  m = keras.Model(inputs=input_layer, outputs=x)
  m.set_pipelining_options(gradient_accumulation_steps_per_replica=8)

  return m


# Create a dataset for the model.
def create_dataset():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0
  x_train = tf.expand_dims(x_train, -1)

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


# Create the optimizer.
def create_optimizer():
  o = gradient_descent.SGD(0.01)
  return ALSOptimizer(o,
                      update_frequency=5,
                      accumulate_statistics_over_update_period=False)


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model inside the strategy.
  model = create_model()

  # Create the optimizer and ALS wrapper.
  opt = create_optimizer()

  # Compile the model for training.
  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer=opt,
                metrics=["accuracy"],
                steps_per_execution=8)

  model.fit(dataset, epochs=50, steps_per_epoch=8)
