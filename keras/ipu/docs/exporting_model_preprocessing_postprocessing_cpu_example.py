import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu

import keras

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/009'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)

batch_size = 1
input_shape = (batch_size, 6)
# Number of IPU-optimized iterations.
iterations = 16

# Configure the IPU for compilation.
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = ipu.config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

# Prepare the `preprocessing_step` function signature.
preprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                              dtype=tf.string),
                                tf.TensorSpec(shape=input_shape,
                                              dtype=tf.string))
# Prepare the `postprocessing_step` function signature.
postprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                               dtype=np.float32),)


# The preprocessing step is performed fully on the CPU.
@tf.function(input_signature=preprocessing_step_signature)
def preprocessing_step(lhs_input, rhs_input):
  transform_fn = lambda input: tf.constant(
      1.0) if input == "graphcore" else tf.random.uniform(shape=tuple(),
                                                          dtype=np.float32)
  transform_string = lambda input: tf.stack([
      tf.stack([transform_fn(elem) for elem in tf.unstack(rank1)])
      for rank1 in tf.unstack(input)
  ])
  return transform_string(lhs_input), transform_string(rhs_input)


# The postprocessing step is performed fully on the CPU.
@tf.function(input_signature=postprocessing_step_signature)
def postprocessing_step(model_result):
  return tf.abs(model_result)


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
  # Call `compile` to set the number of iterations of the inference loop.
  # It can be used to tweak the inference latency.
  model.compile(steps_per_execution=iterations)

# Export as a SavedModel.
runtime_func = model.export_for_ipu_serving(
    saved_model_directory,
    preprocessing_step=preprocessing_step,
    postprocessing_step=postprocessing_step)
# Alternatively: `runtime_func = serving.export_keras(
#   model,
#   saved_model_directory,
#   preprocessing_step=preprocessing_step,
#   postprocessing_step=postprocessing_step)`
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  input1_data = tf.constant(
      ["graphcore", "red", "blue", "yellow", "graphcore", "purple"],
      shape=input_shape,
      dtype=tf.string)
  input2_data = tf.constant(
      ["apple", "banana", "graphcore", "orange", "pineapple", "graphcore"],
      shape=input_shape,
      dtype=tf.string)
  print(runtime_func(input1_data, input2_data))
