import time
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from utils import play, record, get_out_video_name, get_out_plot_name
MeshcatVisualizer.play = play


# Load the model
robot = example_robot_data.load('double_pendulum')
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
START_POSITION = np.array([np.pi, 0]) # rotation of 1 from OY clockwise, rotation of 2 in respect to 1 clockwise
START_POSITION = np.array([0, 0])
#START_VELOCITY = np.array([5.0, 5.0])
START_VELOCITY = np.array([1, 0])
#START_VELOCITY /= 20
LENGTH1 = 0.1  # Length of pendulum's first hand
LENGTH2 = 0.2  # Length of pendulum's second hand
DTIME = 0.0003  # Simulation delta time step
SIMULATION_FRAMERATE = 60
INTEGRATION_USE_RUNGE_KUTTA = True  # Whether to use Runge-Kutta method for integration, or just simple dt * a
TARGET_MOTORS = [0.0, 0.0]  # Target position
NSTEPS = 1000
OUT_VIDEO_NAME = get_out_video_name(__file__)
OUT_PLOT_NAME = get_out_plot_name(__file__)


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


viz.display(START_POSITION)
time.sleep(1)


class RotationPos:
    def __init__(self, rotations: tuple[int, int]):
        self.a, self.b = rotations
        self.pos1 = np.array([-math.sin(self.a) * LENGTH1, math.cos(self.a) * LENGTH1])
        self.dpos2 = np.array([-math.sin(self.a + self.b) * LENGTH2, math.cos(self.a + self.b) * LENGTH2])
        self.pos2 = self.pos1 + self.dpos2

    def __sub__(self, oth):
        return np.array([np.linalg.norm(self.pos1 - oth.pos1)**0.5, np.linalg.norm(self.pos2 - oth.pos2)**0.5])


class RotationBoth:
    def __init__(self, rotations: tuple[int, int]):
        self.a, self.b = rotations
        self.rot1 = self.a % (2 * np.pi)
        if self.rot1 > np.pi:
            self.rot1 -= 2 * np.pi
        self.rot2 = (self.a + self.b) % (2 * np.pi)
        if self.rot2 > np.pi:
            self.rot2 -= 2 * np.pi

    def modify(self, current: tuple[int, int]):
        ROT = 0.2
        value = min(ROT, abs(current[0] + current[1]) * 0.2)
        self.rot2 -= (value / ROT) * ROT * np.sign(current[0])
        # self.rot2 += min(0.1, max(-0.1, (current[0] + current[1]) * 10))
        # self.rot2 -= current[1] / 10
        #print(self.rot2)
        #print(RotationPos(current).pos2[0] * 2, "\n\n")
        #self.rot2 += np.clip(RotationPos(current).pos2[0] * 2, -0.2, 0.2)
        #self.rot1 -= current[0] * 0.5
        return self

    def __sub__(self, oth):
        result = np.array([self.rot1 - oth.rot1, self.rot2 - oth.rot2])
        for i in range(2):
            result[i] = result[i] % (2 * np.pi)
            if result[i] >= np.pi:
                result[i] = result[i] - 2 * np.pi
        result[0] = (result[1] / np.pi) ** 3 * np.pi
        return result

    def __repr__(self):
        return f"({self.rot1}, {self.rot2})"


PID_MAX_POWER = 10
PID_KP = np.array([100.0, 100.0])
PID_KI = np.array([-0.01, 0.01])
PID_KD = np.array([0.08, 0.8])
PID_KD_MAX_MOD = 500
PID_VELOCITY_STEP = DTIME * 5
PID_K = 0.1
PID_FIRST_WEIGHT = 0.2

class PID:
    def __init__(self, target: np.ndarray):
        self.target = target
        # Internal parameters
        self.prev_error = 0.0
        self.integral = 0.0
        # State
        # self.is_pumping = False

    def compute(self, current: np.ndarray, velocity: np.ndarray, dt: float):
        # # Pump energy into system if needed
        # Ek1 = velocity[0]**2 * LENGTH1**2 / 2
        # Ek2 = velocity[1]**2 * LENGTH2**2 / 2
        # Ep1 = LENGTH1 * math.cos(current[0]) * 9.81
        # Ep2 = (LENGTH1 * math.cos(current[0]) + LENGTH2 * math.cos(current[0] + current[1])) * 9.81
        # Ek = Ek1 + Ek2
        # Ep = Ep1 + Ep2
        # if Ek < 0.5 and Ep < 0:
        #     self.is_pumping = True
        # if Ek >= 2 and Ep > 0:
        #     self.is_pumping = False
        # if False and self.is_pumping:
        #     #print("PUMP" * 10)
        #     result = np.array([
        #         PID_MAX_POWER / 2, #np.sign(np.sin(current[0])) * PID_MAX_POWER,
        #         0
        #     ])
        #     print(f"PUMP {result=}")
        #     return result

        # if Ek > 50:
        #     MAX_POWER = PID_MAX_POWER / 10
        # else:
        #     MAX_POWER = PID_MAX_POWER
        MAX_POWER = PID_MAX_POWER
        # Basic PID
        current = current + velocity * PID_VELOCITY_STEP
        error = RotationBoth(self.target).modify(current) - RotationBoth(current)
        error -= velocity * 0.02
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        #print(RotationBoth(self.target))
        #print(RotationBoth(self.target).modify(current))
        #print(self.target)
        #print(current)
        #print(RotationBoth(current))
        #print("---compute---")
        #print(f"{error=} {self.integral=} {derivative=} {self.prev_error=} {velocity=}")
        self.prev_error = error
        result = PID_KP * error + PID_KI * self.integral + np.clip(-PID_KD * derivative, -PID_KD_MAX_MOD, PID_KD_MAX_MOD) #- 1000 * velocity * dt #* np.sign(velocity)
        result *= PID_K
        #print(f"{result=}")
        # Clip
        #print(-result[0] * PID_FIRST_WEIGHT, result[1])
        result[0] = min(MAX_POWER, max(-MAX_POWER, -result[0] * PID_FIRST_WEIGHT + result[1])) * -1 #* -np.sign(np.cos(current[0]))
        result[1] = 0
        #print(f"{result=}")
        #return np.sqrt(np.abs(result)) * np.sign(result)
        return result

    def __call__(self, qs: np.ndarray, vs: np.ndarray, torque0: np.ndarray, dt: float):
        return self.compute(qs[-1], vs[-1], dt)


def sim_loop(viz, model, pid: PID, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, nsteps: int):
    torque0 = np.zeros(model.nv)
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
        #torque0 = pid(qs, vs, torque0, dt)
        xdot = lambda qc, vc, torquec: (vc, pin.aba(model, viz.data, qc, vc, torquec))
        if INTEGRATION_USE_RUNGE_KUTTA:
            # Runge-Kutta 4 integration
            k1 = xdot(q,                  v,                  torque0)
            k2 = xdot(q + dt / 2 * k1[0], v + dt / 2 * k1[1], torque0)
            k3 = xdot(q + dt / 2 * k2[0], v + dt / 2 * k2[1], torque0)
            k4 = xdot(q + dt * k3[0],     v + dt * k3[1],     torque0)
            qdelta = dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            vdelta = dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        else:
            qdelta, vdelta = xdot(q, v, torque0)
            qdelta *= dt
            vdelta *= dt
        #print(f"{q=} {v=}")
        #print(f"{torque0=}")
        #print(f"{qdelta=} {vdelta=}")
        qnext = q + qdelta
        vnext = v + vdelta
        pin.computeAllTerms(model, viz.data, qnext, vnext)
        torque0 = pid(qs, vs, torque0, dt)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        time.sleep(1 / SIMULATION_FRAMERATE)
    return qs, vs


pid = PID(target=TARGET_MOTORS)


# Run the simulation online
qs, vs = sim_loop(viz, model, pid, START_POSITION, START_VELOCITY, DTIME, NSTEPS)


# Record a video based on already simulated data
record(viz, qs, OUT_VIDEO_NAME, DTIME, model.getFrameId("base_link"))
