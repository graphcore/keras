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
    name = "optimizers",
    srcs = [
        "__init__.py",
        "adam.py",
        "als_optimizer.py",
        "gradient_accumulation_optimizer.py",
        "gradient_descent.py",
        "optimizer_v2_wrapper.py",
        "parameter_unscaling_optimizer.py",
        "rmsprop.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:engine",
    ],
)

py_library(
    name = "als_optimizer_test_utils",
    testonly = 1,
    srcs = ["als_optimizer_test_utils.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "als_optimizer_test",
    size = "large",
    srcs = [
        "als_optimizer_test.py",
    ],
    python_version = "PY3",
    shard_count = 32,
    deps = [
        ":als_optimizer_test_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "als_optimizer_specializations_test",
    size = "large",
    srcs = [
        "als_optimizer_specializations_test.py",
    ],
    python_version = "PY3",
    shard_count = 6,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "als_optimizer_pipeline_test",
    size = "large",
    srcs = [
        "als_optimizer_pipeline_test.py",
    ],
    python_version = "PY3",
    shard_count = 32,
    deps = [
        ":als_optimizer_test_utils",
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)

tf_py_test(
    name = "optimizer_serialization_test",
    size = "small",
    srcs = [
        "optimizer_serialization_test.py",
    ],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras/api:keras_api",
    ],
)
