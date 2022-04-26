from tensorflow.python import ipu
from keras.applications import resnet

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():

  model = resnet.ResNet50(weights='imagenet')

  # Get the individual assignments - note that they are returned in post-order.
  assignments = model.get_pipeline_stage_assignment()

  # Iterate over them and set their pipeline stages.
  stage_id = 0
  for assignment in assignments:
    assignment.pipeline_stage = stage_id
    # Split the model on the `conv4_block2_add` layer.
    if assignment.layer.name.startswith("conv4_block2_add"):
      stage_id = 1

  # Set the assignments to the model.
  model.set_pipeline_stage_assignment(assignments)

  model.print_pipeline_stage_assignment_summary()
