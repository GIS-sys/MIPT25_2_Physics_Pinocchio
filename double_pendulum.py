import time
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import matplotlib.pyplot as plt


# Load the model
robot = example_robot_data.load('double_pendulum')
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
START_POSITION = np.array([np.pi / 2, np.pi]) # rotation of 1 from OY clockwise, rotation of 2 in respect to 1 clockwise
START_VELOCITY = np.array([5, 3])
GRAVITY = 0.2  # Strength of the gravity
LENGTH1 = 0.1  # Length of pendulum's first hand
LENGTH2 = 0.2  # Length of pendulum's second hand
DT = 0.02  # Simulation delta time step
POWER = 10  # Power of motor to control the pendulum
INTEGRATION_USE_RUNGE_KUTTA = True  # Whether to use Runge-Kutta method for integration, or just simple dt * a
TARGET_TIP_POS = [0, LENGTH1 + LENGTH2]  # Target position: tip should be at maximum possible height
RUNNING_MODELS_AMOUNT = 100  # Amount of steps to achieve goal


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


viz.display(START_POSITION)
time.sleep(1)


# We will solve the problem by finding a chain of states, which satisfy certain conditions
# States are described via pair (xt, ut), where:
#   xt is (angle1, angle2, rotspeed1, rotspeed2)
#   ut is (force1, force2) - how we control the pendulum
# We define method calc(), which will take state (x, u) and calculate both next state, as well as loss - which we will minimize
# Also we define methods xdot (to produce deriviative / accelerations) and tip (to calculate position of pendulum's tip)
class CasadiActionModelDoublePendulum:
    def __init__(self, model):
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        nq, nv = cmodel.nq, cmodel.nv
        self.nx, self.nu = cmodel.nq + cmodel.nv, cmodel.nv

        # Casadi function: [state, control] -> [velocity, acceleration]
        cx = casadi.SX.sym("x", self.nx, 1)
        cu = casadi.SX.sym("u", self.nu, 1)
        self.xdot = casadi.Function(
            'xdot',
            [cx, cu],
            [ casadi.vertcat(cx[nq:], cpin.aba(cmodel, cdata, cx[:nq], cx[nq:], cu)) ]
        )

        # Casadi function mapping: state -> X-Y position of the pendulum tip
        cpin.framesForwardKinematics(cmodel, cdata, cx[:nq])
        self.tip = casadi.Function(
            'tip',
            [cx],
            [ casadi.vertcat(
                LENGTH1 * -casadi.sin(cx[0:1]) - LENGTH2 * casadi.sin(cx[0:1] + cx[1:2]),
                LENGTH1 * casadi.cos(cx[0:1]) - LENGTH2 * casadi.cos(cx[0:1] + cx[1:2])
            ) ]
        )

    def calc(self, x, u):
        # Clip control - in particular disable control over second handle, because duh
        u[0] = casadi.fmin(u[0], POWER)
        u[0] = casadi.fmax(u[0], -POWER)
        u[1] = 0
        # Add gravity
        # g = (sin(a1), sin(a1 + b1))
        g = GRAVITY * casadi.vertcat(casadi.sin(x[0:1]) / LENGTH1, casadi.sin(x[0:1] + x[1:2]) / LENGTH2)
        u += g
        if INTEGRATION_USE_RUNGE_KUTTA:
            # Runge-Kutta 4 integration
            k1 = self.xdot(x,                    u)
            k2 = self.xdot(x + DT / 2 * k1, u)
            k3 = self.xdot(x + DT / 2 * k2, u)
            k4 = self.xdot(x + DT * k3,     u)
            xnext = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            xnext = x + DT * self.xdot(x, u)

        cost = u.T @ u
        return xnext,cost


# Create chain of models
runningModels = [CasadiActionModelDoublePendulum(model) for t in range(RUNNING_MODELS_AMOUNT)]
terminalModel = CasadiActionModelDoublePendulum(model)


# Create equation for constraints
opti = casadi.Opti()
# xs = [(angle1, angle2, rotspeed1, rotspeed2), ...]
xs = [opti.variable(model.nx) for model in runningModels + [terminalModel]]
# us = [(engine1, engine2)]
us = [opti.variable(model.nu) for model in runningModels]


# Start position and velocity
opti.subject_to(xs[0][0:2] == START_POSITION)
opti.subject_to(xs[0][2:4] == START_VELOCITY)

# Link all the models, and calculate total cost to minimize
totalcost = 0
for t in range(RUNNING_MODELS_AMOUNT):
    xnext, rcost = runningModels[t].calc(xs[t], us[t])
    totalcost += rcost
    opti.subject_to(xs[t + 1] == xnext) # states are linked
    opti.subject_to(opti.bounded(-POWER, us[t][0], POWER)) # control is limited
    
# End position and velocity
opti.subject_to(terminalModel.tip(xs[-1]) == TARGET_TIP_POS) # tip of pendulum at max altitude
opti.subject_to(xs[-1][model.nq:] == 0)


# Solve
opti.minimize(totalcost)
opti.solver("ipopt")
try:
    sol = opti.solve_limited()
except:
    pass
finally:
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.value(u) for u in us ])


print("Solution:")
print(f"{xs_sol[0]=}")
print(f"{us_sol[0]=}")
print(f"{xs_sol[-1]=}")
print(f"{us_sol[-1]=}")


# Plot
plt.ion()
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
ax0.plot(xs_sol[:,:model.nq])
ax0.set_ylabel('q')
ax0.legend(['1','2'])
ax1.plot(xs_sol[:,model.nq:])
ax1.set_ylabel('v')
ax1.legend(['1','2'])
ax2.plot(us_sol)
ax2.set_ylabel('u')
ax2.legend(['1','2'])
ax0.set_title("Positions, velocities and losses for joints")
plt.show()


# Visualize
viz.setCameraZoom(5)
viz.play(xs_sol[:,:model.nq], DT, callback=lambda _: time.sleep(0.1))

input("Press ENTER to exit...")


# Save
#np.save(open("solution.txt",'wb'),[xs_sol,us_sol])
