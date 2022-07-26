import tensorflow as tf
from tensorflow.python import ipu

import keras
from keras.datasets import mnist

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()


# Create a dataset for the model.
def create_dataset():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


class MyModel(keras.Model):
  def __init__(self):
    super().__init__(self)
    self.dense_layer_1 = keras.layers.Dense(8)
    self.dense_layer_2 = keras.layers.Dense(8)
    self.concat_layer = keras.layers.Concatenate()
    self.dense_layer_3 = keras.layers.Dense(1)

  def call(self, inputs):
    # Invoke layers inside PipelineStage scopes to assign the layer invocations
    # to the specified pipeline stage.
    with keras.ipu.PipelineStage(0):
      x = self.dense_layer_1(inputs)
    with keras.ipu.PipelineStage(1):
      x1 = self.dense_layer_2(x)
      x2 = self.dense_layer_2(x)
    with keras.ipu.PipelineStage(2):
      x1 = self.dense_layer_2(x1)
      x2 = self.dense_layer_2(x2)
      x = self.concat_layer([x1, x2])
    with keras.ipu.PipelineStage(3):
      x = self.dense_layer_3(x)

    return x


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Construct the model inside the strategy.
  model = MyModel()

  # Compile the model for training.
  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer='rmsprop',
                metrics=["accuracy"],
                steps_per_execution=256)

  model.set_pipelining_options(gradient_accumulation_steps_per_replica=16,
                               device_mapping=[0, 1, 1, 0])
  model.fit(dataset, epochs=2, steps_per_epoch=128)
