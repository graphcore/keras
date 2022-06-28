# Keras: Deep Learning for humans

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

This repository hosts the development of the Keras library.
Read the documentation at [keras.io](https://keras.io/).

## IPU Keras

This version of Keras has been modified to optimise the performance of Keras
models running on Graphcore IPUs.

See the [IPU TensorFlow user guide](https://docs.graphcore.ai/projects/tensorflow-user-guide) for how to use Keras with IPUs.

This section explains how to build the wheel file, and how to run tests.

### Prerequisites
Bazel 3.7.2 is required for building wheel files and running tests.

For running the tests, you will need a wheel file for a compatible version of IPU TensorFlow, and a
matching Poplar installation (from the same Poplar SDK).

### TL;DR
```
lang=sh
# Prerequisites:
# Make sure you have the prerequisites listed in the above section.
# Navigate to your Keras repository.

# Configure:
export TF_POPLAR_BASE=<absolute-path-to-poplar-install>
python configure.py <path-to-tf-wheel>

# Build a wheel:
bazel run //keras/tools/pip_package:build_pip_package <absolute-path-to-output-directory> <tf-version-number>

# Run all non-HW tests:
bazel test $(bazel query //keras/...)
 --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors

# Run all HW tests:
bazel test $(bazel query //keras/...)
 --test_tag_filters=hw_poplar_test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --max_compilation_threads=1" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="1200,1200,1200,1200" --test_output=errors

```

### Configuring
Before building a wheel file or running tests, you will first need to run the `configure` script found in the root of the
repository. This creates a `.bazelrc.user` which specifies bazel configuration options required to
run the tests. One thing this does is set up the test environment so that TensorFlow and Poplar
are available.

Before running the script you need to set `TF_POPLAR_BASE` to the absolute path to your Poplar
installation. The script also requires positional arguments. Call `bash configure --help` for more
information.

```
lang=sh
export TF_POPLAR_BASE=<absolute-path-to-poplar-install>
bash ./configure <path-to-tf-wheel> <path-to-keras-wheel>
```

After calling the configure script, you can add customisations to the `.bazelrc.user` file.

### Building a wheel
The bazel script target `//keras/tools/pip_package:build_pip_package` is used to
build wheel files. It links a subset of the Python source files into a build directory,
and uses setuptools to build a wheel file from them.

The script requires positional arguments. For more information pass `--help` using any of the
methods below.

The script can be run completely through bazel using the following command:

```
lang=sh
bazel run //keras/tools/pip_package:build_pip_package -- <args>
```

Using `bazel run` means the script will be run from within the `bazel-bin` directory, so it's
recommended that you specify an absolute path for the `output_directory` argument. Alternatively,
you can build the script using bazel, then run it manually with the following commands:

```
lang=sh
bazel build //keras/tools/pip_package:build_pip_package
bash bazel-bin/keras/tools/pip_package/build_pip_package <args>
```

### Running the tests
The following command can be used to run all of the tests. Unless you know what they do, it is
recommended that you pass all of the arguments shown in the command when running tests.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors $(bazel query //keras/...)

```

You can specify a different bazel test target to run a more specific set of tests.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors //keras/ipu:sequential_test
```

You can also run a single test by specifying `--test_arg`.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --use_ipu_model --max_compilation_threads=1 --ipu_model_tiles=8" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="240,360,900,3600" --test_output=errors //keras/ipu:sequential_test --test_arg IPUModelTest.testEmptyModelCreation
```

#### Running HW tests
To run hardware tests, you need to pass a different set of arguments to bazel test.

```
lang=sh
bazel test --test_env=TF_POPLAR_FLAGS="--max_infeed_threads=8 --max_compilation_threads=1" --test_env=TF_POPLAR_VLOG_LEVEL=1 --test_size_filters=small,medium,large --test_timeout="1200,1200,1200,1200" --test_output=errors --test_tag_filters=hw_poplar_test_{A}_ipus --test_env="TF_IPU_COUNT={B}" --jobs {C} $(bazel query //keras/...)

```
The `{A}` in `--test_tag_filters=hw_poplar_test_{A}_ipus` specifies which set of tests to run, the valid values are 1, 2, 4, 8 and 16.
The `{B}` in `--test_env="TF_IPU_COUNT={B}"` specifies how many IPUs there are in total in the system.
The `{C}` in `--jobs {C}` specifies how many parallel tests to run. This value should be set to `B / A`.

Note that `--use_ipu_model` and `--ipu_model_tiles` have been omitted from `--test_env=TF_POPLAR_FLAGS`.

## About Keras

Keras is a deep learning API written in Python,
running on top of the machine learning platform [TensorFlow](https://github.com/tensorflow/tensorflow).
It was developed with a focus on enabling fast experimentation.
*Being able to go from idea to result as fast as possible is key to doing good research.*

Keras is:

- **Simple** -- but not simplistic. Keras reduces developer *cognitive load*
to free you to focus on the parts of the problem that really matter.
- **Flexible** -- Keras adopts the principle of *progressive disclosure of complexity*:
simple workflows should be quick and easy, while arbitrarily advanced workflows
should be *possible* via a clear path that builds upon what you've already learned.
- **Powerful** -- Keras provides industry-strength performance and scalability:
it is used by organizations and companies including NASA, YouTube, or Waymo.

---

## Keras & TensorFlow 2

[TensorFlow 2](https://www.tensorflow.org/) is an end-to-end, open-source machine learning platform.
You can think of it as an infrastructure layer for
[differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming).
It combines four key abilities:

- Efficiently executing low-level tensor operations on CPU, GPU, or TPU.
- Computing the gradient of arbitrary differentiable expressions.
- Scaling computation to many devices, such as clusters of hundreds of GPUs.
- Exporting programs ("graphs") to external runtimes such as servers, browsers, mobile and embedded devices.

Keras is the high-level API of TensorFlow 2: an approachable, highly-productive interface
for solving machine learning problems,
with a focus on modern deep learning. It provides essential abstractions and building blocks for developing
and shipping machine learning solutions with high iteration velocity.

Keras empowers engineers and researchers to take full advantage of the scalability
and cross-platform capabilities of TensorFlow 2: you can run Keras on TPU or on large clusters of GPUs,
and you can export your Keras models to run in the browser or on a mobile device.

---

## First contact with Keras

The core data structures of Keras are __layers__ and __models__.
The simplest type of model is the [`Sequential` model](/guides/sequential_model/), a linear stack of layers.
For more complex architectures, you should use the [Keras functional API](/guides/functional_api/),
which allows to build arbitrary graphs of layers, or [write models entirely from scratch via subclasssing](/guides/making_new_layers_and_models_via_subclassing/).

Here is the `Sequential` model:

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. The Keras philosophy is to keep simple things simple,
while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code via subclassing).

```python
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(
                  learning_rate=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Evaluate your test loss and metrics in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

What you just saw is the most elementary way to use Keras.

However, Keras is also a highly-flexible framework suitable to iterate on state-of-the-art research ideas.
Keras follows the principle of **progressive disclosure of complexity**: it makes it easy to get started,
yet it makes it possible to handle arbitrarily advanced use cases,
only requiring incremental learning at each step.

In much the same way that you were able to train & evaluate a simple neural network above in a few lines,
you can use Keras to quickly develop new training procedures or exotic model architectures.
Here's a low-level training loop example, combining Keras functionality with the TensorFlow `GradientTape`:

```python
import tensorflow as tf

# Prepare an optimizer.
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function.
loss_fn = tf.keras.losses.kl_divergence

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

For more in-depth tutorials about Keras, you can check out:

- [Introduction to Keras for engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [Introduction to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Developer guides](https://keras.io/guides/)

---

## Installation

Keras comes packaged with TensorFlow 2 as `tensorflow.keras`.
To start using Keras, simply [install TensorFlow 2](https://www.tensorflow.org/install).

---

## Support

You can ask questions and join the development discussion:

- In the [TensorFlow forum](https://discuss.tensorflow.org/).
- On the [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).
- On the [Keras Slack channel](https://kerasteam.slack.com). Use [this link](https://keras-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

---

## Opening an issue

You can also post **bug reports and feature requests** (only)
in [GitHub issues](https://github.com/keras-team/keras/issues).


---

## Opening a PR

We welcome contributions! Before opening a PR, please read
[our contributor guide](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md).
