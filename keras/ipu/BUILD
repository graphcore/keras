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

load("@org_keras//keras:keras.bzl", "tf_py_test")
load("@local_poprun_configure//:poprun_build_defs.bzl", "mpirun_py_test", "poprun_py_test")

package(
    default_visibility = [
        "//keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "ipu",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = ["//keras/ipu/extensions"],
)

tf_py_test(
    name = "pipeline_test",
    size = "large",
    srcs = ["pipeline_test.py"],
    shard_count = 27,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "sequential_pipeline_test",
    size = "large",
    srcs = ["sequential_pipeline_test.py"],
    shard_count = 8,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "large",
    srcs = ["sequential_test.py"],
    shard_count = 12,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "functional_model_test",
    size = "large",
    srcs = ["functional_model_test.py"],
    shard_count = 12,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "pipeline_save_restore_test",
    size = "large",
    srcs = ["pipeline_save_restore_test.py"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "sequential_pipeline_save_restore_test",
    size = "large",
    srcs = ["sequential_pipeline_save_restore_test.py"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "synthetic_data_test",
    size = "large",
    srcs = ["synthetic_data_test.py"],
    tags = ["hw_poplar_test_2_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "feed_options_test",
    size = "large",
    srcs = ["feed_options_test.py"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "execution_parameters_test",
    size = "large",
    srcs = ["execution_parameters_test.py"],
    shard_count = 13,
    tags = ["hw_poplar_test_8_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "model_replication_mnist_test",
    size = "large",
    srcs = ["model_replication_mnist_test.py"],
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "model_replication_test",
    size = "large",
    srcs = ["model_replication_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test_2_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "models_comparison_test",
    size = "large",
    srcs = ["models_comparison_test.py"],
    shard_count = 3,
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "pipeline_model_replication_test",
    size = "large",
    srcs = ["pipeline_model_replication_test.py"],
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "pipeline_sequential_replication_test",
    size = "large",
    srcs = ["pipeline_sequential_replication_test.py"],
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "sequential_replication_test",
    size = "large",
    srcs = ["sequential_replication_test.py"],
    shard_count = 2,
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "gradient_accumulation_test",
    size = "large",
    srcs = ["gradient_accumulation_test.py"],
    shard_count = 12,
    tags = ["hw_poplar_test_2_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "asynchronous_callbacks_test",
    size = "large",
    srcs = ["asynchronous_callbacks_test.py"],
    shard_count = 6,
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "prng_stability_models_test",
    size = "large",
    srcs = ["prng_stability_models_test.py"],
    shard_count = 4,
    tags = ["hw_poplar_test_4_ipus"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

poprun_py_test(
    name = "poprun_test",
    size = "large",
    srcs = ["poprun_test.py"],
    main = "poprun_test.py",
    num_instances = 2,
    num_replicas = 2,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

poprun_py_test(
    name = "distributed_test",
    size = "large",
    srcs = ["distributed_test.py"],
    main = "distributed_test.py",
    num_instances = 2,
    num_replicas = 8,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

mpirun_py_test(
    name = "mixed_precision_test",
    size = "large",
    srcs = ["mixed_precision_test.py"],
    main = "mixed_precision_test.py",
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)
