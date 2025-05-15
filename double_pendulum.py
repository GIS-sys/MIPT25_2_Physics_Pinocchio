import time
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import numpy as np


# Load the model
robot = example_robot_data.load('double_pendulum')
model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
print(list(frame.name for frame in model.frames)) # Print all the names of all the points


# Constants
# TODO
RUNNING_MODELS_AMOUNT = 100
START_POSITION = np.array([-np.pi, 1])
START_VELOCITY = np.array([-0.5, 0.1])


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


viz.display(START_POSITION)
time.sleep(1)


# Set camera and robot positions
# TODO
#viz.setCameraPosition(np.array([1, 1, 1]))
#viz.setCameraTarget([0, 0, 0])
#viz.setCameraZoom(5)
#constraint_models[0].joint2_placement = pin.SE3(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, 0.8])), np.array([0.2, 0.1, 0.0]))
#fr = robot.model.getFrameId("base_link")
#fr = robot.model.getFrameId("universe")
#robot.model.frames[fr].placement = pin.SE3(pin.rpy.rpyToMatrix(np.array([0.0, 0.0, 0.8])), np.array([20.2, 0.1, 0.0]))


# We will solve the problem by finding a chain of states, which satisfy certain conditions
# States are described via pair (xt, ut), where:
#   xt is (angle1, angle2, rotspeed1, rotspeed2)
#   ut is (force1, force2) - how we control the pendulum
# TODO
# We define method calc(), which will take state (x, u) and calculate both next state, as well as loss - which we will minimize
class CasadiActionModelDoublePendulum:
    dt = 0.02
    length = .3
    def __init__(self,model):
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        nq,nv = cmodel.nq,cmodel.nv
        self.nx = nq+nv
        self.nu = nv

        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        cx = casadi.SX.sym("x",self.nx,1)
        cu = casadi.SX.sym("u",self.nu,1)
        self.xdot = casadi.Function('xdot', [cx,cu], [ casadi.vertcat(cx[nq:], cpin.aba(cmodel,cdata,cx[:nq],cx[nq:],cu)) ])
        print(self.xdot)

        # The self.tip will be a casadi function mapping: state -> X-Z position of the pendulum tip
        cpin.framesForwardKinematics(self.cmodel,self.cdata,cx[:nq])
        self.tip = casadi.Function('tip', [cx], [ self.cdata.oMf[-1].translation[[0,2]] ])

    def calc(self, x, u):
        if True:
            # Runge-Kutta 4 integration
            k1 = self.xdot(x,                u)
            k2 = self.xdot(x + self.dt/2*k1, u)
            k3 = self.xdot(x + self.dt/2*k2, u)
            k4 = self.xdot(x + self.dt*k3,   u)
            xnext = x + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            xnext = x + self.dt * self.xdot(x, u)

        cost = u.T@u
        return xnext,cost


















'''
Solve a pinocchio-based double-pendulum problem, formulated as multiple shooting with RK4 integration.
min_xs,us    sum_0^T-1  l(x,u)
       s.t.     x_0 = x0
                x_t+1 = f(x_t,u_t)  for all t=0..T-1
                c(x_T) = 0

where xs = [ x_0 ... x_T ] and us = [ u_0 ... u_T-1 ]
      l(x,u) = l(u) = u**2  is the running cost.
      f is the integrated dynamics, writen as xnext = x + RK4(ABA(q,v,tau)) with q=x[:NQ], v=x[NQ:] and TAU=U
      c is a terminal (hard) constraints asking the pendulum to be standing up with 0 velocity.

The model is stored in a so-called action-model, mostly defining the [xnext,cost] = calc(x,u) concatenating
l and f functions.

As a results, it plots the state and control trajectories and display the movement in gepetto viewer.
'''

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import matplotlib.pyplot as plt; plt.ion()


    
### PROBLEM
opti = casadi.Opti()
# The control models are stored as a collection of shooting nodes called running models,
# with an additional terminal model.
runningModels = [ CasadiActionModelDoublePendulum(model) for t in range(RUNNING_MODELS_AMOUNT) ]
terminalModel = CasadiActionModelDoublePendulum(model)

# Decision variables
xs = [ opti.variable(model.nx) for model in runningModels+[terminalModel] ]     # state variable
us = [ opti.variable(model.nu) for model in runningModels ]                     # control variable

# Roll out loop, summing the integral cost and defining the shooting constraints.
totalcost = 0
opti.subject_to(xs[0] == np.concatenate((START_POSITION, START_VELOCITY)))
for t in range(RUNNING_MODELS_AMOUNT):
    xnext,rcost = runningModels[t].calc(xs[t], us[t])
    opti.subject_to(xs[t + 1] == xnext )
    totalcost += rcost
    opti.subject_to(opti.bounded(-.00005, us[t][0], .00005)) # control is limited
    opti.subject_to(opti.bounded(-.00005, us[t][1], .00005)) # control is limited
    
# Additional terminal constraint
opti.subject_to(xs[-1][model.nq:] == 0)  # 0 terminal value
opti.subject_to(terminalModel.tip(xs[RUNNING_MODELS_AMOUNT])==[0, terminalModel.length]) # tip of pendulum at max altitude

### SOLVE
opti.minimize(totalcost)
opti.solver("ipopt") # set numerical backend
# Caution: in case the solver does not converge, we are picking the candidate values
# at the last iteration in opti.debug, and they are NO guarantee of what they mean.
try:
    sol = opti.solve_limited()
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.value(u) for u in us ])
except:
    print('ERROR in convergence, plotting debug info.')
    xs_sol = np.array([ opti.value(x) for x in xs ])
    us_sol = np.array([ opti.value(u) for u in us ])
    #xs_sol = np.array([ opti.debug.value(x) for x in xs ])
    #us_sol = np.array([ opti.debug.value(u) for u in us ])

print(xs_sol[0])
print(us_sol[0])
print(xs_sol[-1])
print(us_sol[-1])
#[-3.13884811  2.36685492 -0.49989182  0.10006967]
#[ 1.06416938e-03 -3.72472978e+01]
#[-3.13884811  2.36685492 -0.49989182  0.10006967]
#[ 1.06416938e-03 -3.72472978e+01]


### PLOT AND VIZ
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3) #, constrained_layout=True)
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
if viz:
    print(len(xs_sol))
    print(len(xs_sol[0]))
    print(len(xs_sol[1]))
    viz.setCameraZoom(5)
    viz.play(xs_sol[:,:model.nq], CasadiActionModelDoublePendulum.dt, callback=lambda _: time.sleep(0.1))
    #viz.play(qs, DTIME, callback=my_callback, capture_only_step=cos)

input("Press ENTER to exit...")
#np.save(open("solution.txt",'wb'),[xs_sol,us_sol])
