import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm

from utils import play, record, get_out_video_name
MeshcatVisualizer.play = play


# Load the URDF model
pinocchio_model_dir = Path(__file__).parent / "models"
model_path = pinocchio_model_dir / "example-robot-data/robots"
mesh_dir = pinocchio_model_dir
urdf_filename = "solo.urdf"
urdf_model_path = model_path / "solo_description/robots" / urdf_filename

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)


# Constants
POSITION_READY = pin.neutral(model)
POSITION_STAND = np.array([0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6])
VELOCITY_RANDOM = np.random.randn(model.nv) ** 2 * 0.5 + 0.3
VELOCITY_RANDOM_UP = VELOCITY_RANDOM
VELOCITY_RANDOM_UP[2] = 4

START_POSITION = POSITION_READY
START_VELOCITY = VELOCITY_RANDOM
DTIME = 0.0002
NSTEPS = 4000
OUT_VIDEO_NAME = get_out_video_name(__file__)


# Create a coordinate frame
frame_id = model.getFrameId("HR_FOOT")

# Start the visualizer | Opens blank webpage
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


def sim_loop(viz, model, frame_id: int, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, nsteps: int):
    tau0 = np.zeros(model.nv) # only torque, no velocity
    qs = [start_position]
    vs = [start_velocity]
    for i in tqdm(range(nsteps)):
        # Take current q,v
        # Find acceleration
        # Calculate next q,v
        # Add them
        # Display positions and draw velocities
        q = qs[-1]
        v = vs[-1]
        a1 = pin.aba(model, viz.data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        viz.drawFrameVelocities(frame_id=frame_id)
        # print(q)
    return qs, vs


# Run the simulation online
qs, vs = sim_loop(viz, model, frame_id, START_POSITION, START_VELOCITY, DTIME, NSTEPS)


# Record a video based on already simulated data
record(viz, qs, OUT_VIDEO_NAME, DTIME, model.getFrameId("FL_FOOT"))
