import os
import shutil

import numpy as np
from tensorflow.python import ipu

import keras

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/003'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)

batch_size = 1
input_shape = (batch_size, 4)
# Number of IPU-optimized loop's iterations.
iterations = 16

# Configure the IPU for compilation.
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

# Always create Keras models inside an IPU strategy.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Always set `batch_size` if model has explicit input layers.
  input1 = keras.layers.Input(shape=input_shape[1:],
                              batch_size=batch_size,
                              name="input_1")
  input2 = keras.layers.Input(shape=input_shape[1:],
                              batch_size=batch_size,
                              name="input_2")
  x = keras.layers.Add()([input1, input2])
  model = keras.Model(inputs=[input1, input2], outputs=x)

  model.build([input_shape, input_shape])
  # Call compile to set the number of iterations of the inference loop.
  # It can be used to tweak the inference latency.
  model.compile(steps_per_execution=iterations)

# Export as a SavedModel.
runtime_func = model.export_for_ipu_serving(saved_model_directory)
# Alternatively: `runtime_func = serving.export_keras(model, saved_model_directory)`
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
# This should print the numbers from 2 to 17.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  for i in range(iterations):
    input1_data = np.ones(input_shape, dtype=np.float32) * i
    input2_data = np.ones(input_shape, dtype=np.float32) * 2
    print(runtime_func(input1_data, input2_data))
