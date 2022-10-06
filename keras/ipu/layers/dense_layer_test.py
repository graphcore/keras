# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.ops.f8_ops import create_metadata, convert_to_f8, Format
from tensorflow.python.framework import ops

import keras
from keras.ipu.layers import Dense
from keras import testing_utils


class CoreLayersTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters("float8", "float16", "float32")
  @testing_utils.run_v2_only
  def test_dense(self, tensor_type):
    cfg = IPUConfig()
    cfg.ipu_model.tiles_per_ipu = 8
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()
    strategy = ipu_strategy.IPUStrategyV1()

    with strategy.scope():
      if tensor_type == "float8":
        arr = convert_to_f8(np.array([[1., 2.], [3., -1.], [-3., -5.]]),
                            create_metadata(Format.F143, 0))
        arr_keras = ops.convert_to_tensor(
            np.array([[1., 2.], [3., -1.], [-3., -5.]]).astype("float16"))
      else:
        arr = ops.convert_to_tensor(
            np.array([[1., 2.], [3., -1.], [-3., -5.]]).astype(tensor_type))
        arr_keras = ops.convert_to_tensor(
            np.array([[1., 2.], [3., -1.], [-3., -5.]]).astype(tensor_type))

      keras_layer = keras.layers.Dense(units=3)
      ipu_layer = Dense(units=3)
      ipu_layer.set_weights(keras_layer.get_weights())

      expected_output = keras_layer(arr_keras)

      # Need a dummy initial call to be able to set the weights for the layer
      _ = ipu_layer(arr)
      ipu_layer.set_weights(keras_layer.get_weights())
      # Sanity check that the weights are the same for both layers
      self.assertAllEqual(ipu_layer.get_weights()[0],
                          keras_layer.get_weights()[0])
      self.assertAllEqual(ipu_layer.get_weights()[1],
                          keras_layer.get_weights()[1])
      result = ipu_layer(arr)
      if tensor_type == "float8":
        # Check that we indeed use fp8.
        self.assertNotAllClose(result, expected_output, rtol=1e-5)
        # Check that the results are still approximately similar.
        self.assertAllClose(result, expected_output, rtol=1e-1)
      else:
        self.assertAllClose(result, expected_output, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
