import time
import example_robot_data
import numpy as np
import pinocchio
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer


# Load the model
robot = example_robot_data.load("talos")
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
#print(model.frames[0].__dir__())
#print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
# print(list(config for config in model.referenceConfigurations)) # Print all the names of all the configurations
POSITION_SITTING = model.referenceConfigurations["half_sitting"]
FOOT_TAG_LEFT = "left_sole_link"
FOOT_TAG_RIGHT = "right_sole_link"
ROOT_TAG = "root_joint"
DELAY_BEFORE_LOADED = 1 # seconds to wait before broser is loaded


# Determine what to keep still
STARTING_POSITION = POSITION_SITTING
# TAGS_TO_KEEP_STILL = [ROOT_TAG]
TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT, FOOT_TAG_RIGHT]
FRAMES_TO_KEEP_STILL = [model.getFrameId(tag) for tag in TAGS_TO_KEEP_STILL]


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


# Create constraints
constraint_models = [
    pinocchio.RigidConstraintModel(
        pinocchio.ContactType.CONTACT_6D,
        model,
        model.frames[frame_id].parent,
        model.frames[frame_id].placement,
        0,
        viz.data.oMf[frame_id],
    ) for frame_id in FRAMES_TO_KEEP_STILL
]


# Put the robot into a balerine pose for fun
constraint_models[0].joint2_placement = pinocchio.SE3(pinocchio.rpy.rpyToMatrix(np.array([0.0, 0.0, 0.8])), np.array([0.2, 0.1, 0.0]))
constraint_models[1].joint2_placement = pinocchio.SE3(pinocchio.rpy.rpyToMatrix(np.array([0.0, 0.0, -1.4])), np.array([00.0, -0.2, 0.0]))


time.sleep(DELAY_BEFORE_LOADED)
viz.display(STARTING_POSITION)

