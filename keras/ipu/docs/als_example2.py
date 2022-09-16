# Simple training example using ALS with gradient accumulation.

import numpy as np

from tensorflow.python import ipu

import keras
from keras.ipu.optimizers import ALSOptimizer
from keras.optimizer_v2.gradient_descent import SGD

config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  input_layer = keras.layers.Input(32)
  x = keras.layers.Dense(16)(input_layer)
  x = keras.layers.Dense(8)(x)

  m = keras.Model(input_layer, x)

  opt = SGD(0.01)
  opt = ALSOptimizer(opt)

  m.compile(loss='mse', optimizer=opt, steps_per_execution=8)
  m.set_gradient_accumulation_options(
      gradient_accumulation_steps_per_replica=4)

  in_data = np.ones((128, 32), dtype=np.float16)
  targets = 2.0 * np.ones((128, 8), dtype=np.float16)

  m.fit(in_data, targets, epochs=10, steps_per_epoch=8)
