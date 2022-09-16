# Simple training example using ALS.

import numpy as np

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import loops

import keras
from keras.ipu.optimizers import ALSOptimizer
from keras.optimizer_v2.gradient_descent import SGD

config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  opt = SGD(0.01)
  opt = ALSOptimizer(opt)

  layer_0 = keras.layers.Dense(16)
  layer_1 = keras.layers.Dense(8)
  mse = keras.losses.MeanSquaredError(
      reduction=keras.losses.losses_utils.ReductionV2.SUM)

  @tf.function(jit_compile=True)
  def f(x, t, _):
    y = layer_1(layer_0(x))
    l = mse(y_true=t, y_pred=y)

    v = layer_0.trainable_variables + layer_1.trainable_variables
    g = opt.get_gradients(l, v)
    opt.apply_gradients(zip(g, v))

    return x, t, l

  @tf.function(jit_compile=True)
  def training_loop(x, t):
    _, _, l = loops.repeat(10, f, inputs=[x, t, 0.0])
    return l

  in_data = np.ones((128, 32), dtype=np.float16)
  targets = 2.0 * np.ones((128, 8), dtype=np.float16)

  strategy.run(training_loop, args=[in_data, targets])
