import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu

import keras

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/011'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)

batch_size = 1
input_shape = (batch_size, 4)
iterations = 16

# Configure the IPU for compilation.
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 4
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()


# The preprocessing step is performed fully on the IPU.
def preprocessing_step(lhs_input, rhs_input):
  abs_layer = keras.layers.Lambda(tf.abs)
  return abs_layer(lhs_input), abs_layer(rhs_input)


# The postprocessing step is performed fully on the IPU.
def postprocessing(model_result):
  reduce_layer = keras.layers.Lambda(tf.reduce_sum)
  return reduce_layer(model_result)


# Always create Keras models inside an IPU strategy.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Always set `batch_size` if model has explicit input layers.
  input1_ph = keras.layers.Input(shape=input_shape[1:],
                                 batch_size=batch_size,
                                 name="input_1")
  input2_ph = keras.layers.Input(shape=input_shape[1:],
                                 batch_size=batch_size,
                                 name="input_2")
  with keras.ipu.PipelineStage(0):
    input1, input2 = preprocessing_step(input1_ph, input1_ph)

  with keras.ipu.PipelineStage(1):
    x = keras.layers.Multiply()([input1, input2])

  with keras.ipu.PipelineStage(2):
    x = keras.layers.Add()([x, input2])

  with keras.ipu.PipelineStage(3):
    x = postprocessing(x)

  model = keras.Model(inputs=[input1_ph, input2_ph], outputs=x)
  model.set_pipelining_options(device_mapping=[0, 1, 2, 3])

  model.build([input_shape, input_shape])
  # Call compile to set the number of times each pipeline stage is executed.
  # It can be used to minimize the latency a bit.
  model.compile(steps_per_execution=iterations)

# Export as a SavedModel.
runtime_func = model.export_for_ipu_serving(saved_model_directory)
# Alternatively: `runtime_func = serving.export_keras(model, saved_model_directory)`
print("SavedModel written to", saved_model_directory)

# You can test the exported executable using returned runtime_func
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  for i in range(iterations):
    input1_data = np.ones(input_shape, dtype=np.float32) * i
    input2_data = np.ones(input_shape, dtype=np.float32) * 2
    print(runtime_func(input1_data, input2_data))
