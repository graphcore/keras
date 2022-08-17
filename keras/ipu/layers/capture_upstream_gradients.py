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
"""
Keras layer for capturing upstream gradients.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import inspect

from tensorflow.python.ipu.ops.grad_util_ops import capture_upstream_gradients
from tensorflow.python.platform import tf_logging as logging

from keras import activations
from keras.engine.base_layer import Layer
from keras.layers import Activation
from keras.utils import generic_utils


class _GradientCaptureLayerBase(Layer):
  def __init__(self, layer, tag):
    _GradientCaptureLayerBase._check_is_layer(layer)

    self._layer = layer
    self._tag = tag

    super().__init__(dtype=layer.dtype, name=tag)

  def call(self, inputs, *args, **kwargs):
    raise NotImplementedError(
        "_GradientCaptureLayerBase cannot be called. Child classes must "
        "override its call method.")

  def build(self, input_shape):  # pylint: disable=useless-super-delegation
    return super().build(input_shape)

  @staticmethod
  def _check_is_layer(layer):
    if not isinstance(layer, Layer):
      raise ValueError(
          "Expected an instance of a keras.engine.base_layer.Layer")

  @staticmethod
  def _filter_call_kwargs(layer, kwargs):
    if kwargs and 'training' in kwargs:
      spec = inspect.getfullargspec(layer.__class__.call)
      kw = spec.kwonlyargs
      if not kw:
        return {}

      if not 'training' in kw:
        logging.warn(
            f"Layer {layer.name} does not take a 'training' kwarg to its call "
            "method, but one was passed to _GradientCaptureLayerBase.call. "
            "This keyword argument cannot be delegated to the wrapped layer's "
            "call method, so is being dropped.")
        return {k: kwargs[k] for k in kwargs if k != 'training'}
    return kwargs

  def get_config(self):
    config = super().get_config()
    config.update({
        'wrapped_layer_type': self.layer.__class__,
        'tag': self.tag
    })

    return config

  @property
  def tag(self):
    return self._tag

  @property
  def layer(self):
    return self._layer

  @property
  def non_trainable_variables(self):
    return self.layer.non_trainable_variables

  @property
  def trainable_variables(self):
    return self.layer.trainable_variables

  @property
  def variables(self):
    return self.layer.variables

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights

  @property
  def weights(self):
    return self.layer.weights


class CaptureUpstreamGradients(_GradientCaptureLayerBase):
  """A Keras Layer wrapper that captures incoming (upstream) gradients on the
  backward pass of a training step.

  Captured gradients are passed to the optimizers `apply_gradients` method via
  its `captured_grads` keyword argument (if it takes this keyword argument),
  for use in the gradient update stage.

  Example:
  .. code-block:: python
    class SGDCaptureOptimizer(SGD):
      def apply_gradients(self, grads_and_vars, captured_grads=None):
        if captured_grads:
          print(captured_grads)
        return super().apply_gradients(grads_and_vars)

      @property
      def supports_captured_grads(self):
        return True

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input = keras.layers.Input(...)

      num_units = ...

      x = CaptureUpstreamGradients(keras.layers.Dense(
        num_units, activation=None))(input)

      x = keras.layers.Activation(keras.activations.relu)(x)

      opt = SGDCaptureOptimizer()

      m = keras.Model(input, x)
      m.compile(opt, 'mse')

      m.fit(...)

  In the above example, the gradient of the `Activation` layer will be captured
  and passed to `SGDCaptureOptimizer.apply_gradients` via a dictionary of
  captured gradients.

  Note that it is not the output of the layer that is wrapped, it is the layer
  instance itself that is wrapped.
  """
  def __init__(self, layer, tag=None):
    """Construct an instance of `CaptureUpstreamGradients` wrapping an instance
    of another Keras layer for which incoming gradients are to be captured.

    Args:
        layer (keras.Layer): An instance of a Keras layer to be wrapped.
        tag (str, optional): An optional tag for use in the gradient dictionary
          passed to an optimizer that supports captured graients.
          Defaults to None.
    """
    tag = tag if not tag is None else layer.name + "_cug"
    super().__init__(layer, tag)

  def call(self, inputs, *args, **kwargs):
    filtered_kwargs = _GradientCaptureLayerBase._filter_call_kwargs(
        self.layer, kwargs)

    return capture_upstream_gradients(self.layer.call(inputs, *args,
                                                      **filtered_kwargs),
                                      tag=self.tag)

  def build(self, input_shape):
    super().build(input_shape)
    return self.layer.build(input_shape)

  def get_config(self):
    config = super().get_config()

    config.update({
        'wrapped_layer_config': self.layer.get_config(),
    })

    return config

  @classmethod
  def from_config(cls, config):
    layer_config = config['wrapped_layer_config']
    layer_type = config['wrapped_layer_type']
    layer = layer_type.from_config(layer_config)

    tag = config['tag']

    return CaptureUpstreamGradients(layer, tag=tag)


class _ActivationCapture(Activation):
  def __init__(self, activation, tag, **kwargs):
    super().__init__(activation, **kwargs)
    self._original_activation = activation
    self._tag = tag

  def call(self, inputs):
    x = capture_upstream_gradients(inputs, tag=self._tag)
    return super().call(x)

  def get_config(self):
    config = super().get_config()
    config.update({
        'activation': self._original_activation,
        'tag': self._tag,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return _ActivationCapture(**config)


class CaptureActivationGradients(_GradientCaptureLayerBase):
  """A Keras Layer wrapper that captures gradients with respect to
  activations on the backward pass of a training step.

  Captured gradients are passed to the optimizers `apply_gradients` method via
  its `captured_grads` keyword argument (if it takes this keyword argument),
  for use in the gradient update stage.

  Example:
  .. code-block:: python
    class SGDCaptureOptimizer(SGD):
      def apply_gradients(self, grads_and_vars, captured_grads=None):
        if captured_grads:
          print(captured_grads)
        return super().apply_gradients(grads_and_vars)

      @property
      def supports_captured_grads(self):
        return True

    strategy = ipu_strategy.IPUStrategyV1()
    with strategy.scope():
      input = keras.layers.Input(...)

      num_units = ...

      x = CaptureActivationGradients(keras.layers.Dense(
        num_units, activation=keras.activations.relu))(input)

      opt = SGDCaptureOptimizer()

      m = keras.Model(input, x)
      m.compile(opt, 'mse')

      m.fit(...)

  In the above example, the gradient of the `Dense` layer's activation will be
  captured and passed to `SGDCaptureOptimizer.apply_gradients` via a dictionary
  of captured gradients.

  Note that it is not the output of the layer that is wrapped, it is the layer
  instance itself that is wrapped.
  """
  def __init__(self, layer, tag=None):
    """Construct an instance of `CaptureActivationGradients` wrapping an
    instance of another Keras layer for which the gradients w.r.t its
    activations are to be captured.

    Args:
        layer (keras.Layer): An instance of a Keras layer to be wrapped.
        tag (str, optional): An optional tag for use in the gradient dictionary
          passed to an optimizer that supports captured graients.
          Defaults to None.
    """
    tag = tag if not tag is None else layer.name + "_cag"
    super().__init__(layer, tag)

    # Check that we actually have activations in this layer.
    if not CaptureActivationGradients.layer_is_supported(layer):
      raise ValueError(
          f"Layer {layer.name} does not use an activation function "
          "so is not supported by "
          "keras.ipu.layers.CaptureActivationGradients")

    layer_config = self.layer.get_config()
    activation_keys = \
      CaptureActivationGradients._find_activation_config_items(layer_config)

    # Inject the capture_upstream_gradients ops into the layer config.
    self._unwrapped_activations = dict()
    self._wrapped_activations = dict()
    for k in activation_keys:
      self._unwrapped_activations[k] = layer_config[k]
      activation = activations.deserialize(self._unwrapped_activations[k])

      wrapped_activation = _ActivationCapture(activation,
                                              tag=f"{self.tag}_{k}")
      self._wrapped_activations[k] = wrapped_activation
      layer_config[k] = wrapped_activation

    # Recreate the inner layer from the updated config.
    with generic_utils.custom_object_scope(self._wrapped_activations):
      self._layer = self.layer.__class__.from_config(layer_config)

  def call(self, inputs, *args, **kwargs):
    filtered_kwargs = _GradientCaptureLayerBase._filter_call_kwargs(
        self.layer, kwargs)
    return self.layer.call(inputs, *args, **filtered_kwargs)

  def build(self, input_shape):
    super().build(input_shape)
    return self.layer.build(input_shape)

  @staticmethod
  def _find_activation_config_items(config):
    if not isinstance(config, dict):
      raise ValueError("Expected a layer configuration dictionary.")

    def _valid_activation(k):
      try:
        _ = activations.deserialize(config[k])
      except:  # pylint: disable=bare-except
        return False

      return True

    return [k for k in config if 'activation' in k and _valid_activation(k)]

  @staticmethod
  def layer_is_supported(layer):
    """Determines if a given instance of a Keras `Layer` is supported by
    `CaptureActivationGradients`.

    Args:
        layer (keras.Layer): The `keras.Layer` instance to test for
          compatability.

    Returns:
        bool: `True` if supported.
    """
    _GradientCaptureLayerBase._check_is_layer(layer)
    return bool(
        CaptureActivationGradients._find_activation_config_items(
            layer.get_config()))

  def get_config(self):
    config = super().get_config()

    original_config = self._layer.get_config()
    for k, a in self._unwrapped_activations.items():
      original_config[k] = a

    config.update({
        'wrapped_layer_config': original_config,
    })

    return config

  @classmethod
  def from_config(cls, config):
    layer_type = config['wrapped_layer_type']
    layer_config = config['wrapped_layer_config']
    layer = layer_type.from_config(layer_config)

    return StoreActivationGradients(layer)
