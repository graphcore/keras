# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
IPU specific Keras extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# pylint: disable=unused-import
from keras.ipu.extensions.pipeline_stage_assignment import PipelineStage
from keras.ipu.extensions.pipeline_stage_assignment import ModelLayerPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import FunctionalLayerPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import SequentialLayerPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import NestedModelPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import FunctionalNestedModelPipelineStageAssignment
from keras.ipu.extensions.pipeline_stage_assignment import SequentialNestedModelPipelineStageAssignment

from keras.ipu.extensions.functional_extensions import FunctionalExtension
from keras.ipu.extensions.sequential_extensions import SequentialExtension
from keras.ipu.extensions.model_extensions import ModelExtension

from keras.ipu.extensions.extensions_base import ReplicatedMetricReductionMethod
# pylint: enable=unused-import
