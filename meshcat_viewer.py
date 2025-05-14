import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


# Redefine method in capturing video to increase video capturing speed
import time
from tqdm import tqdm
def play(self, q_trajectory, dt=None, callback=None, capture=False, capture_only_step=1, **kwargs):
    """
    Play a trajectory with given time step. Optionally capture RGB images and
    returns them.
    """
    nsteps = len(q_trajectory)
    if not capture:
        capture = self.has_video_writer()

    imgs = []
    for i in tqdm(range(nsteps)):
        t0 = time.time()
        self.display(q_trajectory[i])
        if callback is not None:
            callback(i, **kwargs)
        if capture and i % capture_only_step == 0:
            img_arr = self.captureImage()
            if not self.has_video_writer():
                imgs.append(img_arr)
            else:
                self._video_writer.append_data(img_arr)
        t1 = time.time()
        elapsed_time = t1 - t0
        if dt is not None and elapsed_time < dt:
            self.sleep(dt - elapsed_time)
    if capture and not self.has_video_writer():
        return imgs

MeshcatVisualizer.play = play




# Load the URDF model.
pinocchio_model_dir = Path(__file__).parent / "models"
model_path = pinocchio_model_dir / "example-robot-data/robots"
mesh_dir = pinocchio_model_dir
urdf_filename = "solo.urdf"
urdf_model_path = model_path / "solo_description/robots" / urdf_filename

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)


# Start the visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
# Load the robot in the viewer.
viz.loadViewerModel()


# Display a robot configuration # ???
q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)

# Create a convex shape from solo main body # ???
mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex

# Place the convex object on the scene and display it # ???
if convex is not None:
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    geometry = pin.GeometryObject("convex", 0, placement, convex)
    geometry.meshColor = np.ones(4)
    # Add a PhongMaterial to the convex object
    geometry.overrideMaterial = True
    geometry.meshMaterial = pin.GeometryPhongMaterial()
    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
    geometry.meshMaterial.meshShininess = 0.8
    visual_model.addGeometryObject(geometry)
    # After modifying the visual_model we must rebuild
    # associated data inside the visualizer
    viz.rebuildData()

## Display another robot # ???
#viz2 = MeshcatVisualizer(model, collision_model, visual_model)
#viz2.initViewer(viz.viewer)
#viz2.loadViewerModel(rootNodeName="pinocchio2")
#q = q0.copy()
#q[1] = 1.0
#viz2.display(q)

# standing config # ???
q1 = np.array(
    [0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
)

v0 = np.random.randn(model.nv) ** 2 / 2
data = viz.data
pin.forwardKinematics(model, data, q1, v0)
frame_id = model.getFrameId("HR_FOOT")
viz.display()
viz.drawFrameVelocities(frame_id=frame_id)

model.gravity.linear[:] = 0.0
dt = 0.0001


def sim_loop(): # ???
    tau0 = np.zeros(model.nv)
    qs = [q1]
    vs = [v0]
    nsteps = 2000
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        viz.drawFrameVelocities(frame_id=frame_id)
    return qs, vs


qs, vs = sim_loop() # ???

fid2 = model.getFrameId("FL_FOOT") # ???


def my_callback(i, *args): # ???
    viz.drawFrameVelocities(frame_id)
    viz.drawFrameVelocities(fid2)


with viz.create_video_ctx("out/leap.mp4"): # ???
    viz.play(qs, dt, callback=my_callback, capture_only_step=16)
