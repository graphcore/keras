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

from tensorflow.python.ipu.ops.f8_ops import f8_matmul, create_metadata, QuarterTensor, convert_to_f8, Format
import tensorflow as tf

from keras.layers import Dense as KerasDense


class Dense(KerasDense):
  """The keras Dense layer with the additional option of using FP8 matrix multiplication.

  When passing a `tf.Tensor` as input this layer
  behaves identically to keras.layer.Dense.

  To use FP8 multiplication simply pass an instance of QuarterTensor as input.
  The easiest way to do this is by using the `convert_to_f8` function:
  ```
  from tensorflow.python.ipu.ops.f8_ops import create_metadata, convert_to_f8, Format
  from keras.ipu.layers import Dense

  input_array = [[1., 2.], [3., -1.]]
  input_tensor = convert_to_f8(np.array(input_array,
                               create_metadata(Format.F143))
  output = Dense(units=3)(input_tensor)
  ```
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def call(self, inputs):
    """Mostly copied from keras.layers.Dense
    """
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype \
        and not isinstance(inputs, QuarterTensor):
      # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
      inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, tf.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        # pylint: disable=no-value-for-parameter
        inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = tf.SparseTensor(indices=inputs.indices,
                              values=inputs.indices[:, 1],
                              dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = tf.nn.embedding_lookup_sparse(self.kernel,
                                                ids,
                                                weights,
                                                combiner='sum')
      else:
        #####################################################
        # This part is different from the keras.layers.Dense.
        if isinstance(inputs, QuarterTensor):
          kernel = convert_to_f8(self.kernel, create_metadata(Format.F143, 0))
          outputs = f8_matmul(lhs=inputs, rhs=kernel)
        else:
          outputs = tf.raw_ops.MatMul(a=inputs, b=self.kernel)
        #####################################################
    # Broadcast kernel to inputs.
    else:
      outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not tf.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      bias = self.bias
      if isinstance(inputs, QuarterTensor):
        bias = tf.cast(bias, "float16")
      outputs = tf.nn.bias_add(outputs, bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs
