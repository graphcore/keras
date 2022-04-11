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

# tf_py_test(
#     name = "automatic_loss_scaling_optimizer_test",
#     size = "large",
#     srcs = ["automatic_loss_scaling_optimizer_test.py"],
#     shard_count = 8,
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
#     name = "pipeline_test",
#     size = "large",
#     srcs = ["pipeline_test.py"],
#     shard_count = 27,
#     tags = ["isolated_poplar_tests"],
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
#     name = "pipeline_input_validation_test",
#     size = "medium",
#     srcs = ["pipeline_input_validation_test.py"],
#     tags = ["isolated_poplar_tests"],
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
#     name = "sequential_pipeline_test",
#     size = "large",
#     srcs = ["sequential_pipeline_test.py"],
#     shard_count = 8,
#     tags = ["isolated_poplar_tests"],
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
#     name = "sequential_test",
#     size = "large",
#     srcs = ["sequential_test.py"],
#     shard_count = 12,
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
#     name = "functional_model_test",
#     size = "large",
#     srcs = ["functional_model_test.py"],
#     shard_count = 12,
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
#     name = "pipeline_save_restore_test",
#     size = "large",
#     srcs = ["pipeline_save_restore_test.py"],
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
#     name = "sequential_pipeline_save_restore_test",
#     size = "large",
#     srcs = ["sequential_pipeline_save_restore_test.py"],
#     tags = ["isolated_poplar_tests"],
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
#     name = "synthetic_data_test",
#     size = "large",
#     srcs = ["synthetic_data_test.py"],
#     tags = ["hw_poplar_test_2_ipus"],
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
#     name = "optimizers_v2_test",
#     size = "large",
#     srcs = ["optimizers_v2_test.py"],
#     shard_count = 8,
#     tags = ["hw_poplar_test_2_ipus"],
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
#     name = "feed_options_test",
#     size = "large",
#     srcs = ["feed_options_test.py"],
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
#     name = "execution_parameters_test",
#     size = "large",
#     srcs = ["execution_parameters_test.py"],
#     shard_count = 13,
#     tags = ["hw_poplar_test_8_ipus"],
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
#     name = "model_replication_mnist_test",
#     size = "large",
#     srcs = ["model_replication_mnist_test.py"],
#     tags = ["hw_poplar_test_4_ipus"],
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
#     name = "model_replication_test",
#     size = "large",
#     srcs = ["model_replication_test.py"],
#     shard_count = 2,
#     tags = ["hw_poplar_test_2_ipus"],
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
#     name = "models_comparison_test",
#     size = "large",
#     srcs = ["models_comparison_test.py"],
#     shard_count = 2,
#     tags = ["hw_poplar_test_4_ipus"],
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
#     name = "pipeline_model_replication_test",
#     size = "large",
#     srcs = ["pipeline_model_replication_test.py"],
#     tags = ["hw_poplar_test_4_ipus"],
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
#     name = "pipeline_sequential_replication_test",
#     size = "large",
#     srcs = ["pipeline_sequential_replication_test.py"],
#     tags = ["hw_poplar_test_4_ipus"],
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
#     name = "sequential_replication_test",
#     size = "large",
#     srcs = ["sequential_replication_test.py"],
#     shard_count = 2,
#     tags = ["hw_poplar_test_4_ipus"],
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
#     name = "gradient_accumulation_test",
#     size = "large",
#     srcs = ["gradient_accumulation_test.py"],
#     shard_count = 2,
#     tags = ["hw_poplar_test_2_ipus"],
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
#     name = "asynchronous_callbacks_test",
#     size = "large",
#     srcs = ["asynchronous_callbacks_test.py"],
#     shard_count = 6,
#     tags = ["hw_poplar_test_4_ipus"],
#     deps = [
#         "//:expect_absl_installed",
#         "//:expect_numpy_installed",
#         "//:expect_tensorflow_installed",
#         "//keras",
#         "//keras/api:keras_api",
#         "//keras:combinations",
#     ],
# )
