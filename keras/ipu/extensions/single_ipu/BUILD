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

tf_py_test(
    name = "callbacks_test",
    size = "large",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    shard_count = 25,
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
    name = "correctness_test",
    size = "large",
    srcs = ["correctness_test.py"],
    python_version = "PY3",
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
    name = "deferred_sequential_test",
    size = "large",
    srcs = ["deferred_sequential_test.py"],
    python_version = "PY3",
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
    name = "functional_test",
    size = "large",
    srcs = ["functional_test.py"],
    python_version = "PY3",
    shard_count = 25,
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
    name = "metrics_correctness_test",
    size = "large",
    srcs = ["metrics_correctness_test.py"],
    python_version = "PY3",
    shard_count = 25,
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
    name = "metrics_test",
    size = "large",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    shard_count = 5,
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
    python_version = "PY3",
    shard_count = 5,
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras",
        "//keras:combinations",
        "//keras/api:keras_api",
    ],
)

# tf_py_test(
#     name = "training_test",
#     size = "large",
#     srcs = ["training_test.py"],
#     python_version = "PY3",
#     shard_count = 20,
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras:combinations",
#         "//keras/api:keras_api",
#     ],
# )
