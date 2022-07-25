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

load("//keras/ipu/docs/keras_docs_py_test:keras_docs_py_test.bzl", "keras_docs_py_test")

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

keras_docs_py_test(
    name = "example1",
    size = "large",
    srcs = ["example1.py"],
    main = "example1.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example2",
    size = "large",
    srcs = ["example2.py"],
    main = "example2.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example3",
    size = "large",
    srcs = ["example3.py"],
    main = "example3.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example4",
    size = "large",
    srcs = ["example4.py"],
    main = "example4.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example5",
    size = "large",
    srcs = ["example5.py"],
    main = "example5.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example6",
    size = "large",
    srcs = ["example6.py"],
    main = "example6.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example7",
    size = "large",
    srcs = ["example7.py"],
    main = "example7.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example8",
    size = "large",
    srcs = ["example8.py"],
    main = "example8.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example9",
    size = "large",
    srcs = ["example9.py"],
    main = "example9.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example10",
    size = "large",
    srcs = ["example10.py"],
    main = "example10.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example11",
    size = "large",
    srcs = ["example11.py"],
    main = "example11.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "example12",
    size = "large",
    srcs = ["example12.py"],
    main = "example12.py",
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_model_example",
    size = "large",
    srcs = ["exporting_model_example.py"],
    main = "exporting_model_example.py",
    num_ipus = 1,
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_pipelined_model_example",
    size = "large",
    srcs = ["exporting_pipelined_model_example.py"],
    main = "exporting_pipelined_model_example.py",
    num_ipus = 2,
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_pipelined_model_preprocessing_postprocessing_example",
    size = "large",
    srcs = ["exporting_pipelined_model_preprocessing_postprocessing_example.py"],
    main = "exporting_pipelined_model_preprocessing_postprocessing_example.py",
    num_ipus = 4,
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_pipelined_model_preprocessing_postprocessing_cpu_example",
    size = "large",
    srcs = ["exporting_pipelined_model_preprocessing_postprocessing_cpu_example.py"],
    main = "exporting_pipelined_model_preprocessing_postprocessing_cpu_example.py",
    num_ipus = 2,
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_model_preprocessing_postprocessing_example",
    size = "large",
    srcs = ["exporting_model_preprocessing_postprocessing_example.py"],
    main = "exporting_model_preprocessing_postprocessing_example.py",
    num_ipus = 1,
    deps = ["//keras"],
)

keras_docs_py_test(
    name = "exporting_model_preprocessing_postprocessing_cpu_example",
    size = "large",
    srcs = ["exporting_model_preprocessing_postprocessing_cpu_example.py"],
    main = "exporting_model_preprocessing_postprocessing_cpu_example.py",
    num_ipus = 1,
    deps = ["//keras"],
)
