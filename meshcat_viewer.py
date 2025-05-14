# Imports
import sys
from pathlib import Path
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


# Redefine method in capturing video to increase video capturing speed (add capture_only_step kwarg)
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
VELOCITY_RANDOM = np.random.randn(model.nv) ** 2
VELOCITY_RANDOM_UP = VELOCITY_RANDOM
VELOCITY_RANDOM_UP[2] = 4

START_POSITION = POSITION_READY
START_VELOCITY = VELOCITY_RANDOM
DTIME = 0.0001
NSTEPS = 2000

OUT_VIDEO_NAME = "leap.mp4"


# Create a coordinate frame
frame_id = model.getFrameId("HR_FOOT")

# Start the visualizer | Opens blank webpage
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()


## Create a convex shape from solo main body # ???
#mesh = visual_model.geometryObjects[0].geometry
#mesh.buildConvexRepresentation(True)
#convex = mesh.convex
## Place the convex object on the scene and display it # ???
#if convex is not None:
#    placement = pin.SE3.Identity()
#    placement.translation[0] = 2.0
#    geometry = pin.GeometryObject("convex", 0, placement, convex)
#    geometry.meshColor = np.ones(4)
#    # Add a PhongMaterial to the convex object
#    geometry.overrideMaterial = True
#    geometry.meshMaterial = pin.GeometryPhongMaterial()
#    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
#    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])
#    geometry.meshMaterial.meshShininess = 0.8
#    visual_model.addGeometryObject(geometry)
#    # After modifying the visual_model we must rebuild
#    # associated data inside the visualizer
#    viz.rebuildData()
#
## Display another robot # ???
#viz2 = MeshcatVisualizer(model, collision_model, visual_model)
#viz2.initViewer(viz.viewer)
#viz2.loadViewerModel()
#q = q0.copy()
#q = np.array(
#    [0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
#)
#viz2.display(q0)
#
## standing config # ???
#q1 = POSITION_READY
#
#v0 = np.random.randn(model.nv) ** 2
#v0[2] = 3
#data = viz.data
#pin.forwardKinematics(model, data, q1, v0)
#frame_id = model.getFrameId("HR_FOOT")
#viz.display()
#viz.drawFrameVelocities(frame_id=frame_id)
#
#model.gravity.linear[:] = [0.0, 0.0, -9.81]
#dt = 0.0002
#
#
def sim_loop(viz, model, frame_id: int, start_position: np.ndarray, start_velocity: np.ndarray, dt: float, nsteps: int): # ???
    tau0 = np.zeros(model.nv)
    qs = [START_POSITION]
    vs = [START_VELOCITY]
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
    return qs, vs


# Run the simulation online
qs, vs = sim_loop(viz, model, frame_id, START_POSITION, START_VELOCITY, DTIME, NSTEPS)


# Record a video based on already simulated data
cos = input("Want to record a video? Enter to skip, number to create video with capture_only_step=<input>: ")
if cos:
    try:
        cos = int(cos)
    except:
        print("You should input an integer! To avoid error, assuming integer 32")
        cos = 32

    # Create a frame for video
    frame_id_video = model.getFrameId("FL_FOOT")

    # Callback for writing both into output AND video
    def my_callback(i, *args):
        viz.drawFrameVelocities(frame_id)
        viz.drawFrameVelocities(frame_id_video)

    # Write video
    with viz.create_video_ctx(f"out/{OUT_VIDEO_NAME}"):
        viz.play(qs, DTIME, callback=my_callback, capture_only_step=cos)
