.. _keras:

Keras with IPUs
---------------

The Graphcore implementation of TensorFlow includes Keras support for IPUs.
Keras model creation is no different than what you would use if you were
training on other devices. To target the Poplar XLA device, Keras model creation
must be inside the ``strategy.scope`` of an ``IPUStrategy``.

For a more practical walkthrough, see `this tutorial about using Keras on the IPU <https://github.com/graphcore/tutorials/tree/sdk-release-2.5/tutorials/tensorflow2/keras>`_
from the Graphcore tutorials repository.

Single IPU models
~~~~~~~~~~~~~~~~~

You can train, evaluate or run inference on single-IPU models through the Keras
APIs as you would with other accelerators, as long as you create the model
inside the scope of an ``IPUStrategy``:

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

See the documentation for the
`compile method <https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile>`__
for full details.

The example below highlights the usage of ``steps_per_execution``:

.. literalinclude:: example2.py
  :language: python
  :linenos:
  :emphasize-lines: 49-52

Gradient accumulation
~~~~~~~~~~~~~~~~~~~~~

When training, gradient accumulation allows us to simulate bigger batch sizes.
This is achieved by accumulating the gradients across multiple batches together
then performing the weight update.

For example, if we have a model where each step is of batch size 16 and we set
`gradient_accumulation_steps_per_replica` to 4 then this simulates an input
batch of size 64.

Gradient accumulation can be easily enabled for Keras models created inside of
an ``IPUStrategy`` by calling the following methods:

.. table::
  :width: 100%

  +----------------------+----------------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_gradient_accumulation_options` |
  +----------------------+----------------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_gradient_accumulation_options` |
  +----------------------+----------------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_gradient_accumulation_options`      |
  +----------------------+----------------------------------------------------------------------------------------+


See the respective API documentation for more details.

.. note::

  When using data-parallelism, the ``steps_per_execution`` value the model was
  compiled with must be an integer multiple of
  ``gradient_accumulation_steps_per_replica``. Data parallelism is discussed in
  :numref:`automatic-data-parallelism`.


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
(:numref:`fig-grouped-pipeline`), where the forward and backward stages are grouped
together on each IPU. All IPUs alternate between executing a forward pass and then a
backward pass.

Two other schedules are available and can be configured as shown in
:numref:`pipelining-options`. When using the interleaved schedule
(:numref:`fig-interleaved-pipeline`) the forward and backward passes are
interleaved (which requires less memory but is likely to be slower). The
sequential schedule (:numref:`fig-sequential-pipeline`) executes one stage at a
time and may be useful when debugging your model.

A detailed explanation of pipelining can be found in the technical note on `Model parallelism with
TensorFlow: sharding and pipelining <https://docs.graphcore.ai/projects/tf-model-parallelism/en/latest/pipelining.html>`_.

The method to pipeline your model depends on whether your model is a
``Sequential`` model, a ``Functional`` model, or is subclassed from the ``Model``
class.

Sequential model
________________

To enable IPU pipelining for a ``Sequential`` model (an instance of
`keras.Sequential`), a list of per-layer pipeline stage
assignments should be passed to the
:py:meth:`~keras.ipu.extensions.SequentialExtension.set_pipeline_stage_assignment`
method of the model.

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

Functional model
________________

There are two ways to enable IPU pipelining for a ``Functional`` model (an
instance of `keras.Model`) depending on if you're pipelining a model
you are writing yourself or an existing model.

Pipelining a model you are writing yourself
===========================================

To pipeline a ``Functional`` model you are writing yourself, each layer call
must happen within the scope of an `keras.ipu.PipelineStage` context.

For example, a simple four layer ``Functional`` model could be assigned to two
different pipeline stages as follows:

.. literalinclude:: example8.py
  :language: python
  :linenos:
  :start-at: input_layer = keras.layers.Input((28, 28))
  :end-at: model = keras.Model(inputs=input_layer, outputs=x)

.. note::
Layers *constructed* within an `keras.ipu.PipelineStage` context will have that
pipeline stage assigned to all invocations of the layer. These assignments are
overridden if the layer calls happen within a different
`keras.ipu.PipelineStage` context.

Pipelining an existing functional model
=======================================

To pipeline an existing ``Functional`` model, you can use
:py:meth:`~keras.ipu.extensions.FunctionalExtension.get_pipeline_stage_assignment`.
Each layer invocation in the model has an associated
:py:class:`~keras.ipu.extensions.FunctionalLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.
`get_pipeline_stage_assignment` returns a list of these stage assignments,
which you can inspect and modify. Note that the list is in post-order, which
means the assignments are returned in the order they will be executed.

Once you are done modifying the stage assignments, you should use
:py:meth:`~keras.ipu.extensions.FunctionalExtension.set_pipeline_stage_assignment`
to set them on the model.

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
Layers *constructed* within an `keras.ipu.PipelineStage` context will have that
pipeline stage assigned to all invocations of the layer. These assignments are
overridden if the layer calls happen within a different
`keras.ipu.PipelineStage` context.

Pipelining an existing model
============================

To pipeline an existing ``Model`` subclass, you must use
:py:meth:`~keras.ipu.extensions.ModelExtension.get_pipeline_stage_assignment`.
Each layer invocation in the model has an associated
:py:class:`~keras.ipu.extensions.ModelLayerPipelineStageAssignment`
object, which indicates what pipeline stage that invocation is assigned to.
`get_pipeline_stage_assignment` returns a list of these stage assignments,
which you can inspect and modify. Note that the list is in post-order, which
means the assignments are returned in the order they will be executed.

Once you are done modifying the stage assignments, you should use
:py:meth:`~keras.ipu.extensions.ModelExtension.set_pipeline_stage_assignment`
to set them on the model.

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

When replicating, gradients are reduced across replicas during training, which
has implications for gradient accumulation. For a non replicated model, the
*effective batch size* is the product of the dataset batch size and the number
of gradient accumulation steps. In the case of a replication factor greater than
one, the *effective batch size* is additionally scaled by the replication
factor according to the following formula:

`effective_batch_size = dataset_batch_size * gradient_accumulation_steps_per_replica * num_replicas`

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

Keras models created inside of an ``IPUStrategy`` scope automatically create
``IPUInfeedQueue`` and ``IPUOutfeedQueue`` data queues for efficiently feeding
data to and from the IPU devices when using ``fit()``, ``evaluate()`` and
``predict()``.

Instances of ``IPUInfeedQueue`` and ``IPUOutfeedQueue`` can be created with
optional arguments which can affect performance of the model.

Use the following methods to configure the ``IPUInfeedQueue`` for your Keras model:

.. table::
  :width: 100%

  +----------------------+-------------------------------------------------------------------------------+
  | ``Functional`` model | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_infeed_queue_options` |
  +----------------------+-------------------------------------------------------------------------------+
  | ``Sequential`` model | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_infeed_queue_options` |
  +----------------------+-------------------------------------------------------------------------------+
  | ``Model`` subclass   | :py:meth:`~keras.ipu.extensions.ModelExtension.set_infeed_queue_options`      |
  +----------------------+-------------------------------------------------------------------------------+


Use the following methods to configure the ``IPUOutfeedQueue`` for your Keras model:

.. table::
  :width: 100%

  +--------------------+--------------------------------------------------------------------------------+
  | ``Functional``     | :py:meth:`~keras.ipu.extensions.FunctionalExtension.set_outfeed_queue_options` |
  +--------------------+--------------------------------------------------------------------------------+
  | ``Sequential``     | :py:meth:`~keras.ipu.extensions.SequentialExtension.set_outfeed_queue_options` |
  +--------------------+--------------------------------------------------------------------------------+
  | ``Model`` subclass | :py:meth:`~keras.ipu.extensions.ModelExtension.set_outfeed_queue_options`      |
  +--------------------+--------------------------------------------------------------------------------+


For example the ``prefetch_depth`` parameter of the ``IPUInfeedQueue`` and the
``buffer_depth`` parameter of the ``IPUOutfeedQueue`` can be configured as
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


Pipelined Keras model example
_____________________________

This example creates a simple pipelined Keras model that adds two inputs together in the first pipeline stage
and later multiplies the result of the addition operation with the second input in the second pipeline stage.
After that, the model is exported for TensorFlow Serving.

Note that building, compiling and exporting look exactly the same for pipelined and non-pipelined models.

.. literalinclude:: exporting_pipelined_model_example.py
  :language: python
  :linenos:


.. _implementation-details:

Implementation details
~~~~~~~~~~~~~~~~~~~~~~

When instantiating a standard TensorFlow Keras model inside the scope of
an `IPUStrategy` instance, it is dynamically injected with additional,
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
