import numpy as np
import pinocchio as pin
from pinocchio import Model, JointModelRX, JointModelRY, Inertia, SE3, StdVec_Force, Force, Motion
from pinocchio.visualize import MeshcatVisualizer
from scipy.integrate import odeint

model = Model()
model.gravity = Motion(np.zeros(3), np.array([0, 0, -9.81]))

joint1 = JointModelRX()
joint2 = JointModelRY()
inertia1 = Inertia.FromSphere(1.0, 0.5)
inertia2 = Inertia.FromSphere(1.0, 0.5)

model.addJoint(0, joint1, SE3.Identity(), "joint1")
model.appendBodyToJoint(1, inertia1, SE3.Identity())
model.addJoint(1, joint2, SE3.Identity(), "joint2")
model.appendBodyToJoint(2, inertia2, SE3.Identity())

data = model.createData()

q_current = np.array([0.0, 0.0])
q_target = np.array([1.0, 0.5])
dt = 0.01
k_spring = 100.0
damping = 2.0

def dynamics(y, t):
    q, qdot = y[:model.nq], y[model.nq:]

    pin.forwardKinematics(model, data, q, qdot)
    com1 = data.oMi[1].translation
    com2 = data.oMi[2].translation

    pin.forwardKinematics(model, data, q_target)
    com1_target = data.oMi[1].translation
    com2_target = data.oMi[2].translation

    force1 = k_spring * (com1_target - com1) - damping * qdot[0] * np.array([1, 0, 0])
    force2 = k_spring * (com2_target - com2) - damping * qdot[1] * np.array([0, 1, 0])

    tau = np.zeros(model.nv)
    fext = StdVec_Force()
    fext.extend([Force.Zero() for _ in range(model.njoints)])
    fext[1] = Force(force1, np.zeros(3))
    fext[2] = Force(force2, np.zeros(3))

    ddq = pin.aba(model, data, q, qdot, tau, fext)
    return np.concatenate([qdot, ddq])

t_span = np.arange(0, 2.0, dt)
y0 = np.concatenate([q_current, np.zeros(model.nv)])
result = odeint(dynamics, y0, t_span)

visual_model = pin.GeometryModel()
for frame in model.frames:
    if frame.parent != 0:
        visual_geometry = pin.GeometryObject(
            "visual_" + frame.name,
            frame.parent,
            frame.placement,
            SE3.Identity(),
            Inertia.FromSphere(1.0, 0.1).toVisual()
        )
        visual_model.addGeometryObject(visual_geometry)

viz = MeshcatVisualizer(model, visual_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

for q in result[:, :model.nq]:
    viz.display(q)
