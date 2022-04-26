import tensorflow as tf
from tensorflow.python import ipu

import keras
from keras.datasets import mnist

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()


# Create a simple model.
def create_model():
  return keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10)
  ])


# Create a dataset for the model.
def create_dataset():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model inside the strategy.
  model = keras.Sequential([
      keras.layers.Dense(8),  # Pipeline stage 0.
      keras.layers.Dense(16),  # Pipeline stage 0.
      keras.layers.Dense(16),  # Pipeline stage 1.
      keras.layers.Dense(1),  # Pipeline stage 1.
  ])

  model.set_pipeline_stage_assignment([0, 0, 1, 1])

  # Compile the model for training.
  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer='rmsprop',
                metrics=["accuracy"],
                steps_per_execution=256)

  model.set_pipelining_options(
      gradient_accumulation_steps_per_replica=16,
      pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Interleaved)

  model.fit(dataset, epochs=2, steps_per_epoch=128)
