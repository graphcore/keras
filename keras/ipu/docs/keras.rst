.. _keras-with-ipus:

Keras with IPUs
---------------

The Graphcore implementation of Keras includes support for the IPU.
Keras model creation is no different than what you would use if you were
training on other devices. To target the Poplar XLA device, Keras model creation
must be inside the ``strategy.scope`` of an :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy`.

For a more practical walkthrough, see :tutorials-repo:`this tutorial about using Keras on the IPU <tutorials/tensorflow2/keras>` from the Graphcore tutorials repository.

Single IPU models
~~~~~~~~~~~~~~~~~

You can train, evaluate or run inference on single-IPU models through the Keras
APIs as you would with other accelerators, as long as you create the model
inside the scope of an :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy`:

.. literalinclude:: example1.py
  :language: python
  :linenos:
  :emphasize-lines: 2, 7-10, 39-40

.. _using-steps-per-execution:

Using steps_per_execution
~~~~~~~~~~~~~~~~~~~~~~~~~

To reduce Python overhead and maximize the performance of your model, pass the
``steps_per_execution`` argument to the compile method. This argument sets the
number of batches processed sequentially by one replica in a single execution
which can greatly improve performance because any overhead between steps is removed,
thus increasing IPU utilization.

Ideally, ``steps_per_execution`` is equal to the number of steps your model needs
to run per replica in order to complete one epoch. Note that it is not possible
to fetch intermediate results when ``steps_per_execution`` is specified. Model
weights are read on the Python host after all steps are executed on the IPU. If
you need to access model weights during an epoch (for example for saving a
checkpoint), you must set ``steps_per_execution`` accordingly.

.. note::

  In order to achieve best performance, ``steps_per_execution`` needs to be set
  before using ``fit()``, ``evaluate()`` and ``predict()``, even if no training
  is performed.

.. note::

  To achieve best performance when using pipelining, ``steps_per_execution``
  should be set to a significantly larger value than the number of pipeline stages. If
  ``steps_per_execution`` is too small for pipelining to work, your model will fail
  to compile and the minimum value of ``steps_per_execution`` will be reported in the
  error message.

See the documentation for the
`compile method <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`__
for full details.

The example below highlights the usage of ``steps_per_execution``:

.. literalinclude:: example2.py
  :language: python
  :linenos:
  :emphasize-lines: 49-52

.. _gradient-accumulation:

Gradient accumulation
~~~~~~~~~~~~~~~~~~~~~

When training, gradient accumulation allows us to simulate bigger batch sizes.
This is achieved by accumulating the gradients across multiple batches together
then performing the weight update.

For example, if we have a model where each step is of batch size 16 and we set
`gradient_accumulation_steps_per_replica` to 4 then this simulates an input
batch of size 64.

Gradient accumulation can be easily enabled for Keras models created inside of
an :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy` by calling the following methods:

.. table::
  :width: 100%

  +----------------------+----------------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_gradient_accumulation_options` |
  +----------------------+----------------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_gradient_accumulation_options` |
  +----------------------+----------------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_gradient_accumulation_options`      |
  +----------------------+----------------------------------------------------------------------------------------+

This method can be used to configure gradient accumulation parameters in
models that do not use pipelining, particularly `gradient_accumulation_steps_per_replica`
and `gradient_accumulation_reduction_method`. See the respective API documentation for more
details on these arguments.

For pipelined models, these arguments can be passed to the `set_pipelining_options()` methods. See
:numref:`pipelining-options` for more details

.. note::

  A step commonly refers to the forward and backward passes, followed by a
  weight update. On the IPU, when gradient accumulation is used, a step refers
  to the forward and backward passes on a micro batch, but not including the
  corresponding weight update (the weight update is only performed every
  ``gradient_accumulation_steps_per_replica`` steps). The number of weight
  update steps per execution is given by the ``steps_per_execution`` value the
  model was compiled with, divided by
  ``gradient_accumulation_steps_per_replica``. An execution, which is a
  compiled Poplar program, must have an integer number of weight update
  steps, such that all the accumulated gradients are applied.
  Therefore, ``steps_per_execution`` must be an integer multiple of
  ``gradient_accumulation_steps_per_replica``.

.. note::
  The steps per epoch value (``steps_per_epoch``) applies per replica.
  ``steps_per_epoch`` needs to be set when using a dataset of
  infinite cardinality. An epoch consists of one or more executions. Therefore, if set,
  ``steps_per_epoch`` must be an integer multiple of ``steps_per_execution``.

.. note::

  Not all operations are compatible with gradient accumulation.

The example below highlights the usage of ``set_gradient_accumulation_options``:

.. literalinclude:: example3.py
  :language: python
  :linenos:
  :emphasize-lines: 68-69

Model parallelism
~~~~~~~~~~~~~~~~~

The models described so far occupy a single IPU device, however some models
might require the model layers to be split across multiple IPU devices to
achieve high compute efficiency.

One method to achieve model parallelism is called *pipelining*, where the
model layers are assigned to *pipeline stages*. Each pipeline stage can be
assigned to a different device and different devices can execute in parallel.

By default, these pipeline stages will be executed using the grouped schedule
(:numref:`fig-grouped-pipeline`), where the forward and backward stages are
grouped together on each IPU. All IPUs alternate between executing a forward
pass and then a backward pass.

.. figure:: figures/grouped_pipeline.png
   :align: center
   :name: fig-grouped-pipeline

   Grouped pipeline

Two other schedules are available and can be configured as shown in
:numref:`pipelining-options`. When using the interleaved schedule
(:numref:`fig-interleaved-pipeline`) the forward and backward passes are
interleaved (which requires less memory but is likely to be slower). The
sequential schedule (:numref:`fig-sequential-pipeline`) executes one stage at a
time and may be useful when debugging your model.

.. figure:: figures/interleaved_pipeline.png
   :align: center
   :name: fig-interleaved-pipeline

   Interleaved pipeline

.. figure:: figures/sequential_pipeline.png
   :align: center
   :name: fig-sequential-pipeline

   Sequential pipeline

A detailed explanation of pipelining can be found in the technical note on `Model parallelism with
TensorFlow: sharding and pipelining <https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html>`_.

Recomputation of activations can be enabled to reduce memory usage. See
:numref:`recomputation` for further details on when and how to use it.

The method to pipeline your model depends on whether your model is a
``Sequential`` model, a ``Functional`` model, or is subclassed from the ``Model``
class.

Sequential model
________________


Enabling pipelining for a ``Sequential`` model requires assigning each layer in
the model to a pipeline stage, by calling
:py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipeline_stage_assignment`.

If your model does not contain any nested Keras models, a simple overload of
:py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipeline_stage_assignment`
can be used which accepts a list of integers. The integers in the list
correspond to the layers in the model, each specifying which pipeline stage the
corresponding layer is assigned to.

For example, a simple four layer ``Sequential`` model could be assigned to two
different pipeline stages as follows:

.. literalinclude:: example7.py
  :language: python
  :linenos:
  :start-at: model = keras.Sequential([
  :end-at: model.set_pipeline_stage_assignment([0, 0, 1, 1])

You can confirm which layers are assigned to which stages using the
:py:meth:`~keras.ipu.extensions.SequentialExtension.print_pipeline_stage_assignment_summary`
method of the model.

Pipelining a model containing nested models
===========================================

If your ``Sequential`` model contains nested Keras models, you can use
:py:meth:`~keras.ipu.extensions.SequentialExtension.get_pipeline_stage_assignment`.
This returns a list of pipeline stage assignment objects, corresponding to the
layers in the model, which you can inspect and modify.

Each layer in the model has an associated
:py:class:`~keras.ipu.SequentialLayerPipelineStageAssignment`
object, which indicates what pipeline stage that layer is assigned to.

Each nested Keras model in the model has an associated
:py:class:`~keras.ipu.SequentialNestedModelPipelineStageAssignment`
containing a list of pipeline stage assignments which you can inspect and
modify.

Once you are done modifying the stage assignments, you can pass them to
:py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipeline_stage_assignment`
to set them on the model. Pipeline stage assignments for nested Keras models
will **NOT** be recursively set on each nested Keras model. The assignments are
all stored on the root model (the model on which
:py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipeline_stage_assignment`
was called) and they are only used when calling ``fit()``, ``predict()``, or
``evaluate()`` on that model.


Functional model
________________

There are two ways to enable IPU pipelining for a ``Functional`` model (an
instance of `keras.Model`) depending on if you're pipelining a model
you are writing yourself or an existing model.

Pipelining a model you are writing yourself
===========================================

To pipeline a ``Functional`` model you are writing yourself, each layer call
must happen within the scope of an :py:class:`keras.ipu.PipelineStage` context.

For example, a simple four layer ``Functional`` model could be assigned to two
different pipeline stages as follows:

.. literalinclude:: example8.py
  :language: python
  :linenos:
  :start-at: input_layer = keras.layers.Input((28, 28))
  :end-at: model = keras.Model(inputs=input_layer, outputs=x)

.. note::

  Layers *constructed* within a :py:class:`~keras.ipu.PipelineStage` context will have that
  pipeline stage assigned to all invocations of the layer. These assignments are
  overridden if the layer calls happen within a different
  :py:class:`~keras.ipu.PipelineStage` context.

Pipelining an existing functional model
=======================================

To pipeline an existing ``Functional`` model, you can use
:py:meth:`~keras.ipu.extensions.FunctionalExtension.get_pipeline_stage_assignment`.
This returns a list of pipeline stage assignment objects, corresponding to each
invocation in the model which you can inspect and modify. Note that the list is
in post-order, which means the assignments are returned in the order they will
be executed.

Each layer invocation in the model has an associated
:py:class:`~keras.ipu.FunctionalLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.

Each nested Keras model invocation in the model has an associated
:py:class:`~keras.ipu.FunctionalNestedModelPipelineStageAssignment`
containing a list of pipeline stage assignments which you can inspect and modify.

Once you are done modifying the stage assignments, you can pass them to
:py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipeline_stage_assignment`
to set them on the model. Pipeline stage assignments for nested Keras models
will **NOT** be recursively set on each nested Keras model. The assignments are
all stored on the root model (the model on which
:py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipeline_stage_assignment`
was called) and they are only used when calling ``fit()``, ``predict()``, or
``evaluate()`` on that model.

For example, a naive way of pipelining ResNet50 would be to assign everything
up until the "conv4_block2_add" layer invocation to the first stage, then
everything else to the second stage, as follows:

.. literalinclude:: example9.py
  :language: python
  :linenos:
  :start-at: strategy = ipu.ipu_strategy.IPUStrategy()

.. note::

  You can use :py:meth:`~keras.ipu.extensions.FunctionalExtension.print_pipeline_stage_assignment_summary`
  to print the pipeline stage assignments of the model's layer invocations.

.. note::

  This method of assigning pipeline stages can also be used with ``Functional``
  models you are writing yourself, as well as with ``Sequential``
  models and ``Model`` subclasses using the
  :py:class:`~keras.ipu.extensions.SequentialExtension` and
  :py:class:`~keras.ipu.extensions.ModelExtension`
  equivalents.

.. _model-subclass:

Model subclass
______________

``Model`` subclasses are subclasses of `keras.Model`, which override the call
method. There are two ways to enable IPU pipelining for an instance of a
``Model`` subclass, depending on if you're pipelining a model you are writing
yourself or an existing model. These are very similar to the methods available
for ``Functional`` models.

Note that the following ``Model`` methods cannot be overridden:

* `train_step`
* `make_train_function`
* `test_step`
* `make_test_function`
* `predict_step`
* `make_predict_function`

Pipelining a model you are writing yourself
===========================================

To pipeline a ``Model`` subclass you are writing yourself, each layer call
must happen within the scope of an `keras.ipu.PipelineStage` context.

For example, a simple four layer ``Model`` subclass could be assigned to four
different pipeline stages as follows:

.. literalinclude:: example11.py
  :language: python
  :linenos:
  :start-at: class MyModel(keras.Model):
  :end-at:     return x

.. note::

  Layers *constructed* within a :py:class:`~keras.ipu.PipelineStage` context will have that
  pipeline stage assigned to all invocations of the layer. These assignments are
  overridden if the layer calls happen within a different
  :py:class:`~keras.ipu.PipelineStage` context.

Pipelining an existing model
============================

To pipeline an existing ``Model`` subclass, you must use
:py:meth:`~keras.ipu.extensions.ModelExtension.get_pipeline_stage_assignment`.
This returns a list of pipeline stage assignment objects, corresponding to each
invocation in the model which you can inspect and modify. Note that the list is
in post-order, which means the assignments are returned in the order they will
be executed.

Each layer invocation in the model has an associated
:py:class:`~keras.ipu.ModelLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.

Each nested Keras model invocation in the model has an associated
:py:class:`~keras.ipu.NestedModelPipelineStageAssignment` containing a list of
pipeline stage assignments which you can inspect and modify.

Once you are done modifying the stage assignments, you can pass them to
:py:meth:`~keras.ipu.extensions.ModelExtension.set_pipeline_stage_assignment`
to set them on the model. Pipeline stage assignments for nested Keras models
will **NOT** be recursively set on each nested Keras model. The assignments are
all stored on the root model (the model on which
:py:meth:`~keras.ipu.extensions.ModelExtension.set_pipeline_stage_assignment`
was called) and they are only used when calling ``fit()``, ``predict()``, or
``evaluate()`` on that model.

Before you can get or set pipeline stage assignments, you must first call
:py:meth:`keras.Model.build` on your model, specifying the input shapes.
This traces the model's call function using the shapes specified. The resulting
graph is what will be used for pipelined execution. You can update the graph by
calling build again, though this will invalidate existing pipeline stage
assignments if the structure of the updated graph is different.

.. note::

  If you need to specify input dtypes when calling :py:meth:`keras.Model.build`,
  you can pass in :py:class:`keras.Input` objects instead of plain shapes.

For example, an existing ``Model`` subclass with four layers, could be assigned
to four different pipeline stages as follows:

.. literalinclude:: example12.py
  :language: python
  :linenos:
  :start-at:   model = ExistingModel()
  :end-at:   model.set_pipeline_stage_assignment(assignments)

.. note::

  You can use :py:meth:`~keras.ipu.extensions.ModelExtension.print_pipeline_stage_assignment_summary`
  to print the pipeline stage assignments of the model's layer invocations.

.. note::

  This method of assigning pipeline stages can also be used with ``Model``
  subclasses you are writing yourself, as well as with ``Functional`` and
  ``Sequential`` models using the
  :py:class:`~keras.ipu.extensions.SequentialExtension` and
  :py:class:`~keras.ipu.extensions.FunctionalExtension`
  equivalents.


.. _pipelining-options:

Pipelining options
__________________

Pipelining options can be set with the following methods:

.. table::
  :width: 100%

  +----------------------+-----------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipelining_options` |
  +----------------------+-----------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipelining_options` |
  +----------------------+-----------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_pipelining_options`      |
  +----------------------+-----------------------------------------------------------------------------+


See the respective API documentation for more details.

Gradient accumulation is always used when training a pipelined model (unless using the ``Sequential`` schedule). This means
that you must set the option ``gradient_accumulation_steps_per_replica`` using this API when using the ``Grouped`` or
``Interleaved`` schedule. It is optional when using the ``Sequential`` schedule.

The API documentation for ``set_pipelining_options`` explains that the
additional keyword arguments (``pipelining_kwargs``) will be forwarded to the
:py:func:`tensorflow.python.ipu.pipelining_ops.pipeline` operator
(which is used internally - see :numref:`implementation-details`).
Refer to the API documentation for :py:func:`~tensorflow.python.ipu.pipelining_ops.pipeline`
for details about these arguments.

The code sample below illustrates how options can be set with the `set_pipelining_options` API.

.. literalinclude:: example7.py
  :language: python
  :linenos:
  :start-at:   model.set_pipelining_options(
  :end-at:   pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Interleaved)

.. _automatic-data-parallelism:

Automatic data parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~

IPU TensorFlow supports automatic data parallelism when multiple IPU devices are
configured with the system. Automatic data parallelism is achieved by model
replication across available IPU devices. The number of times the model is
replicated is called the replication factor; higher replication factors allow
higher data throughput.

When using replication, gradients are reduced across replicas during training,
which has implications for gradient accumulation. For a non replicated model,
the *effective batch size* is the product of the dataset batch size and the
number of gradient accumulation steps. In the case of a replication factor
greater than one, the *effective batch size* is additionally scaled by the
replication factor according to the following formula:

`effective_batch_size = dataset_batch_size * gradient_accumulation_steps_per_replica * num_replicas`

Metrics can also be reduced across replicas. This behaviour must be configured
by calling the following methods on your model, and specifying a value for
`replicated_metric_reduction_method`:

.. table::
  :width: 100%

  +----------------------+------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_replication_options` |
  +----------------------+------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_replication_options` |
  +----------------------+------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_replication_options`      |
  +----------------------+------------------------------------------------------------------------------+

See the respective API documentation for more details.

Asynchronous callbacks
~~~~~~~~~~~~~~~~~~~~~~

IPU TensorFlow supports the use of ``Callback`` objects with the Keras APIs,
however there is an important difference to note when specifying
`steps_per_execution`. In IPU TensorFlow, if `steps_per_execution` is specified
for your model, then per-batch callback functions will only be invoked every
`steps_per_execution` steps, which can have the effect of delaying access to
results.

However, IPU TensorFlow also supports *asynchronous callbacks* by providing a
polling mechanism which allows results to be accessed at the earliest possible
instance. Asynchronous callbacks can be enabled by passing `True` to the
following methods:

.. table::
  :width: 100%

  +----------------------+---------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_asynchronous_callbacks` |
  +----------------------+---------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_asynchronous_callbacks` |
  +----------------------+---------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_asynchronous_callbacks`      |
  +----------------------+---------------------------------------------------------------------------------+


See the respective API documentation for more details.

Configuring Infeeds and Outfeed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keras models created inside of an :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy` scope
automatically create :py:class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` and
:py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.OutfeedQueue` data queues for efficiently feeding
data to and from the IPU devices when using ``fit()``, ``evaluate()`` and ``predict()``.

Instances of :py:class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` and
:py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.OutfeedQueue` can be created with optional arguments
which can affect performance of the model.

Use the following methods to configure the
:py:class:`~tensorflow.python.ipu.ipu_infeed_queue.IPUInfeedQueue` for your Keras model:

.. table::
  :width: 100%

  +----------------------+-------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_infeed_queue_options` |
  +----------------------+-------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_infeed_queue_options` |
  +----------------------+-------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_infeed_queue_options`      |
  +----------------------+-------------------------------------------------------------------------------+


Use the following methods to configure the :py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.OutfeedQueue` for your Keras model:

.. table::
  :width: 100%

  +--------------------+--------------------------------------------------------------------------------+
  | ``Functional``     | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_outfeed_queue_options` |
  +--------------------+--------------------------------------------------------------------------------+
  | ``Sequential``     | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_outfeed_queue_options` |
  +--------------------+--------------------------------------------------------------------------------+
  | ``Model`` subclass | :py:meth:`~keras.ipu.extensions.ModelExtension.set_outfeed_queue_options`      |
  +--------------------+--------------------------------------------------------------------------------+


For example the ``prefetch_depth`` parameter of the :py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.OutfeedQueue` and the
``buffer_depth`` parameter of the :py:class:`~tensorflow.python.ipu.ipu_outfeed_queue.OutfeedQueue` can be configured as
follows:

.. literalinclude:: example10.py
  :language: python
  :linenos:
  :emphasize-lines: 28-29


Saving and loading Keras models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saving and loading a Keras model must be done within the IPUStrategy scope in
order to save/load IPU-specific information.

When saving and loading ``Model`` subclasses, make sure to save and restore
class members, such as layers, via the config. This can be done by overriding
the ``get_config`` and ``from_config`` methods. Re-creating members from scratch
can cause errors, as the original members may be restored as part of the
IPU-specific internal state.

.. note::
  The arguments `pipelining_kwargs` from :py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipelining_options` and
  `gradient_accumulation_optimizer_kwargs` from :py:meth:`~keras.ipu.extensions.SequentialExtension.set_gradient_accumulation_options`
  are not serializable, which means that when the model
  is being saved, their values are not saved. When restoring/loading a model,
  call ``set_pipelining_options()`` or ``set_gradient_accumulation_options()``
  again.


Exporting precompiled Keras models for TensorFlow Serving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways of exporting Keras models for TensorFlow Serving, independent of whether they're pipelined or not.
Keras models can be exported using the :py:func:`tensorflow.python.ipu.serving.export_keras` function.
This takes only three arguments: the model to export, a directory where the SavedModel will be stored and, optionally, a batch size value.
The other way uses the model's :py:func:`export_for_ipu_serving` method which takes only the path to the SavedModel directory and, optionally, a batch size value.

It's important to note that before exporting the model you must build it, providing the input shapes to the model's :py:func:`build` method.
Similarly to exporting non-Keras models, you can set the iteration
parameter by calling the model's :py:func:`compile` method with `steps_per_execution` argument. The meaning of that parameter is analogous to that of non-Keras models, both non-pipelined and pipelined ones.
In both cases you can use it to tweak the inference latency.

The :py:func:`export_for_ipu_serving` method adds the possibility of passing the `preprocessing_step` and `postprocessing_step`
functions which will be included into the SavedModel graph and executed on the CPU on the server. If all preprocessing
and postprocessing operations are available on the IPU, `preprocessing_step` and `postprocessing_step` functions should
be called inside the Keras model. Then function bodies will be compiled together with the inference model.

Exported models contain Poplar programs compiled for specific batch size value. Because of that, you must always provide the batch size value to be used by the exported model.
You can achieve it in two ways:

* passing the `batch_size` argument explicitly to the export function, or
* setting the batch size value during model creation and leaving the default value of the `batch_size` argument.


Non-pipelined Keras model example
_________________________________

This example creates a simple non-pipelined Keras model that adds two inputs together.
After that, the model is exported for TensorFlow Serving.

.. literalinclude:: exporting_model_example.py
  :language: python
  :linenos:


Non-pipelined Keras model example with additional preprocessing and postprocessing steps
________________________________________________________________________________________

This example exports a very simple Keras model with an embedded IPU program that adds two inputs together. The model also
performs a preprocessing step (on the IPU) to compute the absolute value of the input tensors and a postprocessing step
(on the IPU) to reduce the output.

.. literalinclude:: exporting_model_preprocessing_postprocessing_example.py
  :language: python
  :linenos:

This example exports a very simple Keras model with an embedded IPU program, which doubles the input tensor. The model
also performs a preprocessing step (on the CPU) to convert string tensors to floats and a postprocessing step
(on the CPU) to compute the absolute value of the outputs.

.. literalinclude:: exporting_model_preprocessing_postprocessing_cpu_example.py
  :language: python
  :linenos:

Pipelined Keras model example
_____________________________

This example creates a simple pipelined Keras model that adds two inputs together in the first pipeline stage
and later multiplies the result of the addition operation with the second input in the second pipeline stage.
After that, the model is exported for TensorFlow Serving.

Note that building, compiling and exporting look exactly the same for pipelined and non-pipelined models.

.. literalinclude:: exporting_pipelined_model_example.py
  :language: python
  :linenos:

Pipelined Keras model example with additional preprocessing and postprocessing steps
____________________________________________________________________________________


This example creates a simple pipelined Keras model that adds two inputs together in the first computational
pipeline stage of the model and later multiplies the result of the addition operation with the second input
in the next pipeline stage. The model also performs a preprocessing stage (on the IPU) to compute the
absolute value of the input and a postprocessing stage (on the IPU) to reduce the output.

.. literalinclude:: exporting_pipelined_model_preprocessing_postprocessing_example.py
  :language: python
  :linenos:

This example creates a simple pipelined Keras model that adds two inputs together in the first pipeline stage
and later multiplies the result of the addition operation with the second input in the second pipeline stage.
The model also performs a preprocessing step (on the CPU) to convert string tensors to floats and a postprocessing step
(on the CPU) to compute the absolute value of the outputs.

.. literalinclude:: exporting_pipelined_model_preprocessing_postprocessing_cpu_example.py
  :language: python
  :linenos:

IPU-specific Keras layers and optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:mod:`ipu_tensorflow_addons.keras.layers` namespace contains IPU-specific
implementations of standard Keras layers and optimizers. More information,
including details of every layer and optimizer in this namespace and a code
example showing how to use it can be found in :numref:`ipu-tensorflow-addons`.

.. _implementation-details:

Implementation details
~~~~~~~~~~~~~~~~~~~~~~

When instantiating a standard TensorFlow Keras model inside the scope of
an :py:class:`~tensorflow.python.ipu.ipu_strategy.IPUStrategy` instance, it is dynamically injected with additional,
IPU-specific, functions.
This is done through the relevant *IPU Keras extension classes*:

.. table::
  :width: 100%

  +----------------------+------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension` |
  +----------------------+------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension` |
  +----------------------+------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension`      |
  +----------------------+------------------------------------------------------+

.. _automatic-loss-scaling:

Automatic loss scaling
~~~~~~~~~~~~~~~~~~~~~~

When training deep learning models, the magnitude of computed gradients is typically significantly smaller than
their corresponding weights and activations. Whilst rarely causing training issues for models using float32
precision weights, models using reduced precision formats like float16 are left vulnerable to numerical underflow and vanishing
gradients.

Loss scaling aims to combat vanishing gradients by multiplying the loss value at the end of the forward
pass by some loss scaling factor :math:`{\alpha}`. Consequently, gradients obtained through back-propagation are also
scaled by that constant. The aim is to shift the gradient distribution across the dynamic range,
so that underflow and overflow are prevented (as much as possible) in float16. Note that the gradients need to be scaled down by the inverse
of :math:`{\alpha}` before being consumed by the optimizer for the weight update.

Automatic loss scaling (ALS), as included with IPU Keras, eliminates the need for the manual selection of an appropriate
value for :math:`{\alpha}`. The ALS approach observes the gradients with respect to both weights and activations.
These observations are used to generate histograms that inform the adjustment of the loss scaling factor, with the aim of
preventing excessive underflow or overflow in gradient distribution.
The proportion of FP16 gradients whose absolute values exceed `histogram_bin_edge` is recorded.
If this proportion is below `ratio_threshold` then  :math:`{\alpha}` is scaled up by `increase_factor`.
Otherwise, it is scaled down by `1 / increase_factor`. These updates to :math:`{\alpha}`
occur with a user-specified `update_frequency`. For more details and visual examples, see
`our blogpost about ALS <https://www.graphcore.ai/posts/training-large-models-more-stably-with-automatic-loss-scaling>`_.

.. warning::

  The automatic loss scaling feature in the IPU version of Keras is experimental and may lead to unexpected results.
  So far it has been validated for the SGD optimizer without momentum and with constant learning rate. You can see an example
  of it in `our CNNs application <https://github.com/graphcore/examples/tree/master/vision/cnns/tensorflow2>`_.

.. note::

  `update_frequency` uses the number of calls to `apply_gradients` to determine how often to update :math:`{\alpha}`.
  Features affecting the frequency of these calls, like replication, pipelining or gradient accumulation, might therefore
  require corresponding modifications to `update_frequency` for ALS to exhibit the desired behavior.

The continual updating of :math:`{\alpha}` has the added benefit, as compared to a static scaling factor, of
allowing the optimizer to adapt to changes in the distribution of magnitudes of gradients during training.

In IPU Keras, ALS can be added to any `OptimizerV2`-derived Keras optimizer through the
:py:class:`~keras.ipu.optimizers.ALSOptimizer` wrapper. The example below illustrates how the :py:class:`~keras.ipu.optimizers.ALSOptimizer`
wrapper can be used to add ALS functionality to a standard SGD optimizer and train a model.

.. literalinclude:: als_example1.py
  :language: python
  :linenos:
  :emphasize-lines: 24

While the example above uses a `keras.Model` for simplicity, the :py:class:`~keras.ipu.optimizers.ALSOptimizer` wrapper can also be used within
custom TensorFlow training loops, as shown in the example below.

.. literalinclude:: als_example1_non_keras.py
  :language: python
  :linenos:
  :emphasize-lines: 20

The following example shows how ALS can be combined with gradient accumulation
(outlined in :numref:`gradient-accumulation`) in order to simulate larger batch sizes.

.. literalinclude:: als_example2.py
  :language: python
  :linenos:
  :emphasize-lines: 24, 27-28

The combination of ALS and gradient accumulation does not require the use of `keras.Model`, however. The example
below illustrates how :py:class:`~keras.ipu.optimizers.ALSGradientAccumulationOptimizer` can be used to provide
any `OptimizerV2`-derived Keras optimizer with both ALS and gradient accumulation.

.. note::

  In order to use :py:class:`~keras.ipu.optimizers.ALSGradientAccumulationOptimizer`, the target optimizer must first be wrapped with
  :py:class:`~keras.ipu.optimizers.ALSOptimizer`.

.. literalinclude:: als_example2_non_keras.py
  :language: python
  :linenos:
  :emphasize-lines: 22-23

While it is ultimately the gradients with respect to weights and biases which are used to update the parameters of a model,
gradients with respect to layer activations are also computed during each backward pass. It is possible that these gradients
might saturate, thus losing information, without this being detected in upstream parameter gradients.

By wrapping a Keras
layer in the `CaptureActivationGradients` wrapper, these intermediate activation gradients can be recorded and added to the
statistics used to determine whether to increase or decrease :math:`{\alpha}`. The example below shows how the activation
gradients of select layers can be incorporated into an :py:class:`~keras.ipu.optimizers.ALSOptimizer`. In order to reduce the computational cost of ALS,
the `update_frequency` is increased to 5 (from its default value of 1), and the `accumulate_statistics_over_update_period`
flag is set to `False`. With this setting, the ratio of gradients exceeding `histogram_bin_edge` are recorded only every
fifth batch, rather than being based on the average ratio over the last 5 batches.

.. literalinclude:: als_example3.py
  :language: python
  :linenos:
  :emphasize-lines: 25, 31, 41, 67-69
