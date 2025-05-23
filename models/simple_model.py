#
# Copyright (c) 2016-2023 CNRS INRIA
#

from math import pi
from typing import ClassVar

import pinocchio as pin
from pinocchio.utils import np, rotate

DENSITY = 1


def placement(x=0, y=0, z=0, rx=0, ry=0, rz=0):
    m = pin.SE3.Identity()
    m.translation = np.array([x, y, z])
    m.rotation *= rotate("x", rx)
    m.rotation *= rotate("y", ry)
    m.rotation *= rotate("z", rz)
    return m


def color(body_number=1):
    return [int(i) for i in f"{int(bin(body_number % 8)[2:]):03d}"] + [1]


class ModelWrapper:
    def __init__(self, name=None, display=False):
        self.visuals = [
            ("universe", pin.SE3.Identity(), pin.SE3.Identity().translation)
        ]
        self.name = self.__class__.__name__ if name is None else name
        self.model = pin.Model()
        self.display = display
        self.add_joints()

    def add_joints(self):
        for joint in self.joints:
            self.add_joint(**joint)

    def add_joint(
        self,
        joint_name,
        joint_model=None,
        joint_placement=None,
        lever=None,
        shape="box",
        dimensions=1,
        mass=None,
        body_color=1,
        parent=0,
    ):
        if joint_model is None:
            joint_model = pin.JointModelFreeFlyer()
        elif isinstance(joint_model, str):
            joint_model = pin.__dict__["JointModel" + joint_model]()
        if joint_placement is None:
            joint_placement = pin.SE3.Identity()
        elif isinstance(joint_placement, dict):
            joint_placement = placement(**joint_placement)
        if lever is None:
            lever = pin.SE3.Identity()
        elif isinstance(lever, dict):
            lever = placement(**lever)
        joint_name, body_name = (
            f"world/{self.name}_{joint_name}_{i}" for i in ("joint", "body")
        )
        body_inertia = pin.Inertia.Random()
        if shape == "box":
            w, h, d = (
                (float(i) for i in dimensions)
                if isinstance(dimensions, tuple)
                else [float(dimensions)] * 3
            )
            if mass is None:
                mass = w * h * d * DENSITY
            body_inertia = pin.Inertia.FromBox(mass, w, h, d)
            if self.display:
                self.display.viewer.gui.addBox(body_name, w, h, d, color(body_color))
        elif shape == "cylinder":
            r, h = dimensions
            if mass is None:
                mass = pi * r**2 * h * DENSITY
            body_inertia = pin.Inertia.FromCylinder(mass, r, h)
            if self.display:
                self.display.viewer.gui.addCylinder(body_name, r, h, color(body_color))
        elif shape == "sphere":
            w, h, d = (
                (float(i) for i in dimensions)
                if isinstance(dimensions, tuple)
                else [float(dimensions)] * 3
            )
            if mass is None:
                mass = 4.0 / 3.0 * pi * w * h * d * DENSITY
            body_inertia = pin.Inertia.FromEllipsoid(mass, w, h, d)
            if self.display:
                self.display.viewer.gui.addSphere(
                    body_name, dimensions, color(body_color)
                )
        body_inertia.lever = lever.translation
        joint_id = self.model.addJoint(parent, joint_model, joint_placement, joint_name)
        self.model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())
        self.model.addJointFrame(joint_id, -1)
        self.model.addBodyFrame(body_name, joint_id, pin.SE3.Identity(), -1)
        self.visuals.append((body_name, joint_placement, lever))
        self.data = self.model.createData()
        if self.display:
            self.place()

    def place(self):
        for i, (name, placement, lever) in enumerate(self.visuals):
            if i == 0:
                continue
            self.display.place(name, self.data.oMi[i] * placement * lever)
        self.display.viewer.gui.refresh()


class RobotDisplay:
    def __init__(self, window_name="pinocchio"):
        from gepetto import corbaserver

        self.viewer = corbaserver.Client()
        try:
            window_id = self.viewer.gui.getWindowID(window_name)
        except:  # noqa: E722
            window_id = self.viewer.gui.createWindow(window_name)
            self.viewer.gui.createSceneWithFloor("world")
            self.viewer.gui.addSceneToWindow("world", window_id)
        self.viewer.gui.setLightingMode("world", "OFF")
        self.viewer.gui.setVisibility("world/floor", "OFF")
        self.viewer.gui.refresh()

    def place(self, obj_name, m):
        self.viewer.gui.applyConfiguration(obj_name, pin.se3ToXYZQUATtuple(m))


class SimplestWalker(ModelWrapper):
    joints: ClassVar = [
        {
            "joint_name": "pelvis",
            "dimensions": (0.1, 0.2, 0.1),
            "mass": 0.5,
        },
        {
            "joint_name": "left_leg",
            "joint_model": "RY",
            "joint_placement": {"y": -0.15},
            "lever": {"z": -0.45},
            "shape": "cylinder",
            "dimensions": (0.1, 0.9),
            "mass": 20,
            "body_color": 2,
            "parent": 1,
        },
        {
            "joint_name": "right_leg",
            "joint_model": "RY",
            "joint_placement": {"y": 0.15},
            "lever": {"z": -0.45},
            "shape": "cylinder",
            "dimensions": (0.1, 0.9),
            "mass": 20,
            "body_color": 3,
            "parent": 1,
        },
    ]


walker = SimplestWalker()
model = walker.model
