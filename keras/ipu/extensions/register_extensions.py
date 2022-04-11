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
# ==============================================================================
from tensorflow.python.ipu import keras_extensions

from keras.engine.base_layer import KerasExtension as _KerasExtension
from keras.ipu.extensions.functional_extensions import FunctionalExtension as _FunctionalExtension
from keras.ipu.extensions.sequential_extensions import SequentialExtension as _SequentialExtension
from keras.ipu.extensions.model_extensions import ModelExtension as _ModelExtension
from keras.engine.functional import Functional as _Functional
from keras.engine.sequential import Sequential as _Sequential
from keras.engine.training import Model as _Model

# Insert the extensions for the Keras classes.
# Note: insert Sequential before Functional as Sequential models inherit from
# Functional models.
keras_extensions._extensions_manager._register_extension(  # pylint: disable=protected-access
    _Sequential, _KerasExtension, _SequentialExtension)
keras_extensions._extensions_manager._register_extension(  # pylint: disable=protected-access
    _Functional, _KerasExtension, _FunctionalExtension)
keras_extensions._extensions_manager._register_extension(  # pylint: disable=protected-access
    _Model, _KerasExtension, _ModelExtension)
