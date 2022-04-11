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

# tf_py_test(
#     name = "extensions_test",
#     size = "large",
#     srcs = ["extensions_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )

# tf_py_test(
#     name = "extensions_save_load_test",
#     size = "large",
#     srcs = ["extensions_save_load_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )

# tf_py_test(
#     name = "data_adapter_test",
#     size = "large",
#     srcs = ["data_adapter_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )

# tf_py_test(
#     name = "sequential_pipeline_api_test",
#     size = "large",
#     srcs = ["sequential_pipeline_api_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )

# tf_py_test(
#     name = "functional_pipeline_api_test",
#     size = "large",
#     srcs = ["functional_pipeline_api_test.py"],
#     python_version = "PY3",
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )

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
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )
