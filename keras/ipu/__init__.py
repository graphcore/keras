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
"""IPU specific Keras integration.
"""

from tensorflow.python.util.tf_export import keras_export

# pylint: disable=unused-import
from keras.ipu.extensions.functional_extensions import PipelineStage
from keras.ipu.extensions.functional_extensions import FunctionalLayerPipelineStageAssignment
from keras.ipu.extensions.sequential_extensions import SequentialLayerPipelineStageAssignment
from keras.ipu.extensions.model_extensions import ModelLayerPipelineStageAssignment
# pylint: enable=unused-import

__ipu__built__ = '1'
keras_export('keras.ipu.__ipu__built__').export_constant(
    __name__, '__ipu__built__')
