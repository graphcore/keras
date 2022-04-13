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
    name = "extensions",
    srcs = [
        "__init__.py",
        "data_adapter.py",
        "data_feed_manager.py",
        "extensions_base.py",
        "extensions_util.py",
        "functional_extensions.py",
        "model_extensions.py",
        "polling_thread.py",
        "register_extensions.py",
        "sequential_extensions.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//:expect_absl_installed",
        "//:expect_numpy_installed",
        "//:expect_tensorflow_installed",
        "//keras:combinations",
        "//keras:engine",
    ],
)

tf_py_test(
    name = "extensions_test",
    size = "large",
    srcs = ["extensions_test.py"],
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
    name = "extensions_save_load_test",
    size = "large",
    srcs = ["extensions_save_load_test.py"],
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
    name = "data_adapter_test",
    size = "large",
    srcs = ["data_adapter_test.py"],
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
    name = "sequential_pipeline_api_test",
    size = "large",
    srcs = ["sequential_pipeline_api_test.py"],
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
    name = "functional_pipeline_api_test",
    size = "large",
    srcs = ["functional_pipeline_api_test.py"],
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

# tf_py_test(
#     name = "model_pipeline_api_test",
#     size = "large",
#     srcs = ["model_pipeline_api_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras:combinations",
#         "//keras/api:keras_api",
#     ],
# )
