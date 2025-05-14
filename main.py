import time
import example_robot_data
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer, MeshcatVisualizer


# Load the model
robot = example_robot_data.load("talos")
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
#print(model.frames[0].__dir__())
#print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
# print(list(config for config in model.referenceConfigurations)) # Print all the names of all the configurations
POSITION_SITTING = model.referenceConfigurations["half_sitting"]
VELOCITY_RANDOM = np.random.randn(model.nv) ** 2
FOOT_TAG_LEFT = "left_sole_link"
FOOT_TAG_RIGHT = "right_sole_link"
ROOT_TAG = "root_joint"
DELAY_BEFORE_LOADED = 1 # seconds to wait before broser is loaded


# Determine what to keep still
START_POSITION = POSITION_SITTING
START_VELOCITY = VELOCITY_RANDOM
# TAGS_TO_KEEP_STILL = [ROOT_TAG]
TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT]
# TAGS_TO_KEEP_STILL = [FOOT_TAG_LEFT, FOOT_TAG_RIGHT]
FRAMES_TO_KEEP_STILL = [model.getFrameId(tag) for tag in TAGS_TO_KEEP_STILL]
DTIME = 0.001
NSTEPS = 2000
SLEEP_BETWEEN = 1 / 60


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


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




## create  # TODO
#constraint_datas = [cm.createData() for cm in constraint_models]
#pin.computeAllTerms(model, viz.data, START_POSITION.copy(), np.zeros(model.nv))
#kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_models)
#constraint_dim = sum([cm.size() for cm in constraint_models])


def sim_loop(viz, model, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, sleep_between: float, nsteps: int):
    tau0 = np.zeros(model.nv) # only torque, no velocity # TODO
    qs = [START_POSITION]
    vs = [START_VELOCITY]

    ## TODO v
    #y = np.ones(constraint_dim)

    ## Decrease CoMz by 0.2
    #pin.computeAllTerms(model, viz.data, q, np.zeros(model.nv))
    #com_base = viz.data.com[0].copy()
    #kp = 1.0
    #speed = 1.0

    for k in range(NSTEPS):
        q = qs[-1]
        v = vs[-1]
        # Update positions
        pin.computeAllTerms(model, viz.data, q, v)
        pin.computeJointJacobians(model, viz.data, q)
        # Update body's position using force - gravity + initial velocity
        print(len(viz.data.com))
        print(list(viz.data.com))
        print(len(q))
        print(q)
        a1 = pin.aba(model, viz.data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        q_move_delta = viz.data.com[0] - 1
        #com_act = viz.data.com[0].copy()
        #com_err = com_act - com_des(k)
        #kkt_constraint.compute(model, viz.data, constraint_models, constraint_datas, 1e-7)
        #constraint_value = np.concatenate([pin.log6(cd.c1Mc2) for cd in constraint_datas])
        #J = np.vstack([pin.getFrameJacobian(model, viz.data, cm.joint1_id, cm.joint1_placement, cm.reference_frame) for cm in constraint_models])
        #primal_feas = np.linalg.norm(constraint_value, np.inf)
        #dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        #rhs = np.concatenate([-constraint_value - y * 1e-7, kp * viz.data.mass[0] * com_err, np.zeros(model.nv - 3)])
        #dz = kkt_constraint.solve(rhs)
        #dy = dz[:constraint_dim]
        #dq = dz[constraint_dim:]
        #alpha = 1.0
        #q = pin.integrate(model, q, -alpha * dq)
        #y -= alpha * (-dy + y)
        # TODO ^
        qnext = qs[-1]
        vnext = vs[-1]

        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        for frame_id in FRAMES_TO_KEEP_STILL:
            viz.drawFrameVelocities(frame_id=frame_id)
        time.sleep(sleep_between)
    return qs, vs


qs, vs = sim_loop(viz, model, START_POSITION, START_VELOCITY, DTIME, SLEEP_BETWEEN, NSTEPS)
