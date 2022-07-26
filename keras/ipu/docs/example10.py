from tensorflow.python import ipu

import keras

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()


# Create a simple model.
def create_model():
  return keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10)
  ])


# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():

  model = create_model()

  # Set the infeed and outfeed options.
  model.set_infeed_queue_options(prefetch_depth=2)
  model.set_outfeed_queue_options(buffer_depth=2)
