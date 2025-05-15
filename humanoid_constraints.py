import time
import example_robot_data
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer
from scipy.integrate import odeint
from tqdm import tqdm


from utils import play, record, get_out_video_name
MeshcatVisualizer.play = play


# Load the model
robot = example_robot_data.load("talos")
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
#print(model.frames[0].__dir__())
print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
# print(list(config for config in model.referenceConfigurations)) # Print all the names of all the configurations
POSITION_SITTING = model.referenceConfigurations["half_sitting"]
VELOCITY_RANDOM = np.random.randn(model.nv) ** 2 / 200
VELOCITY_ZERO = np.zeros(model.nv)
FOOT_TAG_LEFT = "left_sole_link"
FOOT_TAG_RIGHT = "right_sole_link"
HAND_TAG_LEFT = "gripper_left_fingertip_3_link"
HAND_TAG_RIGHT = "gripper_right_fingertip_3_link"
ROOT_TAG = "root_joint"
DELAY_BEFORE_LOADED = 1 # seconds to wait before broser is loaded
OUT_VIDEO_NAME = get_out_video_name(__file__)





# vvvvvvvvvvvvvvvvvvvvvvvvvvv
# CHANGEABLE

# 1
# What to keep still, options: 1) only one leg, 2) both legs, 3) one leg and one arm
# TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT]
# TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT, FOOT_TAG_RIGHT]
TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT, FOOT_TAG_RIGHT, HAND_TAG_RIGHT]

# 2
# How exactly to move main part of the body, options: 1) squats, 2) free fall under gravity
#BODY_DISPLACEMENT = lambda k: np.array(
#    [
#        0.0,
#        0.0,
#        np.abs(0.2 * np.sin(2.0 * np.pi * k / 200)),
#    ]
#)
BODY_DISPLACEMENT = lambda k: 0.1 * np.array(
    [
        0.0,
        10 * k / 3000,
        -10 * k / 3000 + 9.81 * (k / 3000)**2 * 10,
    ]
)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^




START_VELOCITY = VELOCITY_ZERO
FRAMES_TO_KEEP_STILL = [model.getFrameId(tag) for tag in TAGS_TO_KEEP_STILL]
START_POSITION = POSITION_SITTING
DTIME = 0.001
NSTEPS = 1000
SLEEP_BETWEEN = 1 / 600



# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


pin.forwardKinematics(model, viz.data, START_POSITION)
pin.framesForwardKinematics(model, viz.data, START_POSITION)


# Create constraints
constraint_models = [
    pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        model,
        model.frames[frame_id].parent,
        model.frames[frame_id].placement,
        0,
        viz.data.oMf[frame_id],
    ) for frame_id in FRAMES_TO_KEEP_STILL
]


# Put the robot into a balerine pose for fun
constraint_models[0].joint2_placement = pin.SE3(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, 0.8])), np.array([0.2, 0.1, 0.0]))
if len(constraint_models) > 1:
    constraint_models[1].joint2_placement = pin.SE3(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, -1.4])), np.array([0.0, -0.2, 0.0]))


time.sleep(DELAY_BEFORE_LOADED)
viz.display(START_POSITION)






# create helper variables for constraints
constraint_datas = [cm.createData() for cm in constraint_models]
pin.computeAllTerms(model, viz.data, START_POSITION.copy(), np.zeros(model.nv))
kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_models)
constraint_dim = sum([cm.size() for cm in constraint_models])


def sim_loop(viz, model, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, sleep_between: float, nsteps: int):
    qs = [START_POSITION]
    vs = [START_VELOCITY]

    y = np.ones(constraint_dim)
    pin.computeAllTerms(model, viz.data, qs[-1], np.zeros(model.nv))
    com_base = viz.data.com[0].copy()
    kp = 1.0
    speed = 1.0

    for k in tqdm(range(NSTEPS)):
        q = qs[-1]
        v = vs[-1]
        # Update positions
        pin.computeAllTerms(model, viz.data, q, v)
        pin.computeJointJacobians(model, viz.data, q)
        # Update body's position using force - gravity + initial velocity
        com_act = viz.data.com[0].copy()
        com_err = com_act - com_base + BODY_DISPLACEMENT(k)
        kkt_constraint.compute(model, viz.data, constraint_models, constraint_datas, 1e-8)
        constraint_value = np.concatenate([pin.log6(cd.c1Mc2) for cd in constraint_datas])
        J = np.vstack([pin.getFrameJacobian(model, viz.data, cm.joint1_id, cm.joint1_placement, cm.reference_frame) for cm in constraint_models])
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        rhs = np.concatenate([-constraint_value - y * 1e-8, kp * viz.data.mass[0] * com_err, np.zeros(model.nv - 3)])
        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]
        alpha = 1.0
        q = pin.integrate(model, q, -alpha * dq)
        y -= alpha * (-dy + y)
        qnext = q
        vnext = v

        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        for frame_id in FRAMES_TO_KEEP_STILL:
            viz.drawFrameVelocities(frame_id=frame_id)
        # time.sleep(sleep_between)
    return qs, vs


qs, vs = sim_loop(viz, model, START_POSITION, START_VELOCITY, DTIME, DELAY_BEFORE_LOADED, NSTEPS)

# Record a video based on already simulated data
record(viz, qs, OUT_VIDEO_NAME, DTIME, model.getFrameId(FOOT_TAG_LEFT))
