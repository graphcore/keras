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

from absl.testing import parameterized

import tensorflow as tf
from keras import testing_utils
from keras import ipu as keras_ipu


class APITest(tf.test.TestCase, parameterized.TestCase):
  @testing_utils.run_v2_only
  def testKerasExport(self):
    self.assertTrue(tf.keras.ipu)
    self.assertEqual(tf.keras.ipu.__ipu__built__, '1')
    self.assertEqual(tf.keras.ipu.PipelineStage, keras_ipu.PipelineStage)
    self.assertEqual(tf.keras.ipu.FunctionalLayerPipelineStageAssignment,
                     keras_ipu.FunctionalLayerPipelineStageAssignment)
    self.assertEqual(tf.keras.ipu.SequentialLayerPipelineStageAssignment,
                     keras_ipu.SequentialLayerPipelineStageAssignment)
    self.assertEqual(tf.keras.ipu.ModelLayerPipelineStageAssignment,
                     keras_ipu.ModelLayerPipelineStageAssignment)


if __name__ == '__main__':
  tf.test.main()
