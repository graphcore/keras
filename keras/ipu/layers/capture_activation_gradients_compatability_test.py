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

import tensorflow.compat.v2 as tf

import keras
from keras import activations
from keras.ipu.layers import CaptureActivationGradients
from keras.ipu.layers.capture_upstream_gradients import _ActivationCapture


class CustomLayerNoAct(keras.layers.Layer):
  def __init__(self, **kwargs):  # pylint: disable=arguments-differ
    super().__init__(**kwargs)

  def call(self, inputs):  # pylint: disable=arguments-differ
    return inputs

  def get_config(self):  # pylint: disable=useless-super-delegation
    return super().get_config()

  @classmethod
  def from_config(cls, config):
    return CustomLayerNoAct(**config)


class CustomLayer1Act(keras.layers.Layer):
  def __init__(self, activation='relu', **kwargs):
    super().__init__(**kwargs)
    self._activation = activations.get(activation)

  def call(self, inputs):  # pylint: disable=arguments-differ
    return self._activation(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({'activation': self._activation})
    return config

  @classmethod
  def from_config(cls, config):
    return CustomLayer1Act(**config)


class CustomLayer2Act(keras.layers.Layer):
  def __init__(self, activation='relu', second_activation='sigmoid', **kwargs):
    super().__init__(**kwargs)
    self._activation = activations.get(activation)
    self._second_activation = activations.get(second_activation)

  def call(self, inputs):  # pylint: disable=arguments-differ
    return self._activation(inputs) + self._second_activation(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        'activation': self._activation,
        'second_activation': self._second_activation
    })
    return config

  @classmethod
  def from_config(cls, config):
    return CustomLayer2Act(**config)


class CustomLayer3Act(keras.layers.Layer):
  def __init__(self,
               activation='relu',
               activation_second='sigmoid',
               third_activation_fn='linear',
               **kwargs):
    super().__init__(**kwargs)
    self._activation = activations.get(activation)
    self._activation_second = activations.get(activation_second)
    self._third_activation_fn = activations.get(third_activation_fn)

  def call(self, inputs):  # pylint: disable=arguments-differ
    return self._activation(inputs) + self._activation_second(inputs) + \
      self._third_activation_fn(inputs)

  def get_config(self):
    config = super().get_config()
    config.update({
        'activation': self._activation,
        'activation_second': self._activation_second,
        'third_activation_fn': self._third_activation_fn
    })
    return config

  @classmethod
  def from_config(cls, config):
    return CustomLayer3Act(**config)


def case(l_type, args, num_acts):
  return {
      'testcase_name': l_type.__name__,
      'layer_type': l_type,
      'layer_args': tuple(args),
      'num_activations': num_acts
  }


# Custom layers.
LAYER_CASES = [
    case(CustomLayerNoAct, (), 0),
    case(CustomLayer1Act, (), 1),
    case(CustomLayer2Act, (), 2),
    case(CustomLayer3Act, (), 3),
]

# Core layers.
LAYER_CASES += [
    case(keras.layers.Dense, (1,), 1),
    case(keras.layers.Embedding, (2, 1), 0),
    case(keras.layers.Masking, (), 0),
    case(keras.layers.Lambda, (lambda x: x,), 0),
]

# Convolution layers.
LAYER_CASES += [
    case(l, (1, 3), 1) for l in [
        keras.layers.Conv1D,
        keras.layers.Conv2D,
        keras.layers.Conv3D,
        keras.layers.Conv1DTranspose,
        keras.layers.Conv2DTranspose,
        keras.layers.Conv3DTranspose,
        keras.layers.SeparableConv1D,
        keras.layers.SeparableConv2D,
        keras.layers.DepthwiseConv2D,
    ]
]

# Pooling layers.
LAYER_CASES += [
    case(l, (), 0) for l in [
        keras.layers.MaxPooling1D,
        keras.layers.MaxPooling2D,
        keras.layers.MaxPooling3D,
        keras.layers.AveragePooling1D,
        keras.layers.AveragePooling2D,
        keras.layers.AveragePooling3D,
        keras.layers.GlobalMaxPooling1D,
        keras.layers.GlobalMaxPooling2D,
        keras.layers.GlobalMaxPooling3D,
        keras.layers.GlobalAveragePooling1D,
        keras.layers.GlobalAveragePooling2D,
        keras.layers.GlobalAveragePooling3D,
    ]
]

# Recurrent layers.
LAYER_CASES += [
    case(keras.layers.LSTM, (2,), 2),
    case(keras.layers.GRU, (2,), 2),
    case(keras.layers.SimpleRNN, (2,), 1),
    case(keras.layers.TimeDistributed, (keras.layers.Dense(1),), 0),
    case(keras.layers.Bidirectional, (keras.layers.SimpleRNN(2),), 0),
    case(keras.layers.ConvLSTM1D, (2, 2), 2),
    case(keras.layers.ConvLSTM2D, (2, 2), 2),
    case(keras.layers.ConvLSTM3D, (2, 2), 2),
]

# Preprocessing layers.
LAYER_CASES += [
    case(keras.layers.TextVectorization, (), 0),
    case(keras.layers.Normalization, (), 0),
    case(keras.layers.Discretization, (), 0),
    case(keras.layers.CategoryEncoding, (2,), 0),
    case(keras.layers.Hashing, (2,), 0),
    case(keras.layers.StringLookup, (), 0),
    case(keras.layers.IntegerLookup, (), 0),
    case(keras.layers.Resizing, (2, 2), 0),
    case(keras.layers.Rescaling, (2,), 0),
    case(keras.layers.CenterCrop, (2, 2), 0),
    case(keras.layers.RandomCrop, (2, 2), 0),
    case(keras.layers.RandomFlip, (), 0),
    case(keras.layers.RandomTranslation, (1, 1), 0),
    case(keras.layers.RandomRotation, (1.0,), 0),
    case(keras.layers.RandomZoom, (1,), 0),
    case(keras.layers.RandomHeight, (1,), 0),
    case(keras.layers.RandomWidth, (1,), 0),
    case(keras.layers.RandomContrast, (1,), 0),
]

# Normalization layers.
LAYER_CASES += [
    case(l, (), 0) for l in [
        keras.layers.BatchNormalization,
        keras.layers.LayerNormalization,
    ]
]

# Regularization layers.
LAYER_CASES += [
    case(l, (0.5,), 0) for l in [
        keras.layers.Dropout, keras.layers.SpatialDropout1D,
        keras.layers.SpatialDropout2D, keras.layers.SpatialDropout3D,
        keras.layers.GaussianDropout, keras.layers.GaussianNoise,
        keras.layers.ActivityRegularization, keras.layers.AlphaDropout
    ]
]

# Attention layers.
LAYER_CASES += [
    case(keras.layers.MultiHeadAttention, (2, 2), 0),
    case(keras.layers.Attention, (), 0),
    case(keras.layers.AdditiveAttention, (), 0)
]

# Reshaping layers.
LAYER_CASES += [
    case(keras.layers.Reshape, ((2, 2),), 0),
    case(keras.layers.RepeatVector, (2,), 0),
    case(keras.layers.Permute, ((1, 2),), 0)
]

LAYER_CASES += [
    case(l, (), 0) for l in [
        keras.layers.Flatten, keras.layers.Cropping1D, keras.layers.Cropping2D,
        keras.layers.Cropping3D, keras.layers.UpSampling1D, keras.layers.
        UpSampling2D, keras.layers.UpSampling3D, keras.layers.ZeroPadding1D,
        keras.layers.ZeroPadding2D, keras.layers.ZeroPadding3D
    ]
]

# Merging layers.
LAYER_CASES += [case(keras.layers.Dot, (1,), 0)]

LAYER_CASES += [
    case(l, (), 0) for l in [
        keras.layers.Concatenate, keras.layers.Average, keras.layers.Maximum,
        keras.layers.Minimum, keras.layers.Add, keras.layers.Subtract,
        keras.layers.Multiply
    ]
]

# Locally connected layers.
LAYER_CASES += [
    case(l, (1, 1), 1) for l in
    [keras.layers.LocallyConnected1D, keras.layers.LocallyConnected2D]
]


class CaptureUpstreamGradientsTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.named_parameters(*LAYER_CASES)
  def testLayerCompatability(self, layer_type, layer_args, num_activations):
    # Create a layer to test.
    l = layer_type(*layer_args)

    # If the layer does not have user-specifiable activations, then trying to
    # wrap the layer should raise an exception.
    if num_activations == 0:
      with self.assertRaisesRegex(ValueError,
                                  "does not use an activation function"):
        _ = CaptureActivationGradients(l)
      return

    # This layer should be wrappable.
    self.assertTrue(CaptureActivationGradients.layer_is_supported(l))

    l_wrapped = CaptureActivationGradients(l)

    self.assertEqual(len(l_wrapped._wrapped_activations), num_activations)  # pylint: disable=protected-access

    for a in l_wrapped._wrapped_activations.values():  # pylint: disable=protected-access
      self.assertEqual(a.__class__, _ActivationCapture)


if __name__ == "__main__":
  tf.test.main()
