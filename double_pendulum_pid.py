import time
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import play, record, get_out_video_name, get_out_plot_name
MeshcatVisualizer.play = play


# Load the model
robot = example_robot_data.load('double_pendulum')
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
START_POSITION = np.array([np.pi / 2, np.pi]) # rotation of 1 from OY clockwise, rotation of 2 in respect to 1 clockwise
START_VELOCITY = np.array([5, 3])
GRAVITY = 0.1  # Strength of the gravity
LENGTH1 = 0.1  # Length of pendulum's first hand
LENGTH2 = 0.2  # Length of pendulum's second hand
DTIME = 0.02  # Simulation delta time step
POWER = 10  # Power of motor to control the pendulum
SIMULATION_FRAMERATE = 60
INTEGRATION_USE_RUNGE_KUTTA = True  # Whether to use Runge-Kutta method for integration, or just simple dt * a
TARGET_TIP_POS = [0, LENGTH1 + LENGTH2]  # Target position: tip should be at maximum possible height
NSTEPS = 4000
OUT_VIDEO_NAME = get_out_video_name(__file__)
OUT_PLOT_NAME = get_out_plot_name(__file__)


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


viz.display(START_POSITION)
time.sleep(1)


class PID:
    pass

    def __call__(self, qs, vs, tau0):
        return tau0


def sim_loop(viz, model, pid: PID, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, nsteps: int):
    tau0 = np.zeros(model.nv)
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
        xdot = lambda qc, vc, tauc: (vc, pin.aba(model, viz.data, qc, vc, tauc))
        if INTEGRATION_USE_RUNGE_KUTTA:
            # Runge-Kutta 4 integration
            k1 = xdot(q,                     v,                     tau0)
            k2 = xdot(q + DTIME / 2 * k1[0], v + DTIME / 2 * k1[1], tau0)
            k3 = xdot(q + DTIME / 2 * k2[0], v + DTIME / 2 * k2[1], tau0)
            k4 = xdot(q + DTIME * k3[0],     v + DTIME * k3[1],     tau0)
            qdelta = DTIME / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            vdelta = DTIME / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        else:
            qdelta, vdelta = DTIME * xdot(q, v, tau0)
        qnext = q + qdelta
        vnext = v + vdelta
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)

        tau0 = pid(qs, vs, tau0)
        time.sleep(1 / SIMULATION_FRAMERATE)
    return qs, vs


pid = PID() # TODO


# Run the simulation online
qs, vs = sim_loop(viz, model, pid, START_POSITION, START_VELOCITY, DTIME, NSTEPS)


# Record a video based on already simulated data
record(viz, qs, OUT_VIDEO_NAME, DTIME, model.getFrameId("base_link"))
