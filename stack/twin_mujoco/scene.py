"""The MuJoCo twin's scene, built from robot.yaml (docs/ARCHITECTURE.md).

The MuJoCo world frame IS the rig frame: work surface at z = 0, origin midway between the shoulder-pan axes,
+x forward, +y to the robot's left. (The Digital Twin (Newton) uses a shifted frame; this one does not
need to.) Contents:

- the work surface (a plane at z = 0) and the 18 mm board inside its measured outline;
- the fixed obstacles (camera stand, camera head) as static boxes;
- both SO-101 arms from the upstream MJCF (meshes from ~/.cache/robot-use/so101_mujoco, tools/fetch_so101_assets.py),
  attached with prefixes "left_" / "right_" at the mounts robot.yaml derives;
- objects with a free joint (boxes, or open cups built from a base and a ring of wall boxes) given by the caller:
  the program's own beliefs about the scene (previews), or a scene file (`scenes/*.yaml`, `load_scene_file`) with
  the objects placed at random within ranges, from a seed.

Cameras (`cameras=`, render.py) are placed from robot.yaml, for the simulated rig (robotd --sim);
previews and robotd's hull checks build the scene without them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
DEFAULT_ASSETS = Path("~/.cache/robot-use/so101_mujoco").expanduser()
OLD_ASSETS = Path("~/.cache/so2x/so101_mujoco").expanduser()      # where it was fetched before the rename (D15)
BOARD_BACK_X = -0.30                     # the board extends behind the arms; its back edge is not measured
SHAPES = ("box", "cup")
CUP_SEGMENTS = 24                        # wall boxes around a cup (MuJoCo collides convex pieces only)
SCENES_DIR = Path(__file__).resolve().parent / "scenes"


def assets_dir() -> Path:
    env = os.environ.get("ROBOT_USE_SO101_ASSETS")
    d = Path(env).expanduser() if env else DEFAULT_ASSETS
    if not env and not (d / "so101_new_calib_camera.xml").exists() and (OLD_ASSETS / "so101_new_calib_camera.xml").exists():
        d = OLD_ASSETS                                    # a machine that fetched it before: no new download needed
    if not (d / "so101_new_calib_camera.xml").exists():
        raise FileNotFoundError(f"SO-101 MuJoCo model not found in {d}: run tools/fetch_so101_assets.py")
    return d


@dataclass
class SceneObject:
    name: str
    size: tuple                          # full extents x, y, z (m)
    xyz: tuple                           # centre; z defaults to resting on the table
    yaw_deg: float = 0.0
    mass: float = 0.05
    rgba: tuple = (0.85, 0.15, 0.15, 1.0)
    friction: float = 1.2
    shape: str = "box"                   # "box", or "cup": open at the top, size = (diameter, diameter, height)
    wall_m: float = 0.004                # a cup's wall and base thickness

    @classmethod
    def from_dict(cls, d: dict) -> "SceneObject":
        size = tuple(float(v) for v in d.get("size", (0.04, 0.04, 0.04)))
        xyz = [float(v) for v in d["xyz"]] if "xyz" in d else [float(v) for v in d["xy"]] + [size[2] / 2]
        if len(xyz) == 2:
            xyz.append(size[2] / 2)
        shape = str(d.get("shape", "box"))
        if shape not in SHAPES:
            raise ValueError(f"object {d['name']!r}: shape {shape!r} is not one of {SHAPES}")
        o = cls(str(d["name"]), size, tuple(xyz), float(d.get("yaw_deg", 0.0)), float(d.get("mass", 0.05)),
                tuple(d.get("rgba", (0.85, 0.15, 0.15, 1.0))), float(d.get("friction", 1.2)), shape,
                float(d.get("wall_m", 0.004)))
        if shape == "cup" and size[0] != size[1]:
            raise ValueError(f"cup {o.name!r}: a cup is round: size [diameter, diameter, height], got {size}")
        if shape == "cup" and not (0 < 2 * o.wall_m < min(size[0], size[2])):
            raise ValueError(f"cup {o.name!r}: wall {o.wall_m} m does not fit its size {size}")
        return o


@dataclass
class Scene:
    model: object
    arms: dict = field(default_factory=dict)          # arm -> {"qpos": [...], "dof": [...], "act": [...], "tip_site": id}
    objects: dict = field(default_factory=dict)       # name -> {"body": id, "qpos": addr}
    mounts: dict = field(default_factory=dict)


def _yaw_quat(deg: float) -> list:
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _swap_in_jaw_pieces(child, jaw_hulls: dict) -> None:
    """Replace each jaw's single collision mesh by its convex pieces (stack/assets/gripper_collision), placed where
    the MJCF places the mesh. The original geom moves to group 4 (kept for reference, not collided)."""
    import mujoco

    for g in list(child.geoms):
        if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.meshname not in jaw_hulls or g.group != 3:
            continue
        body = g.parent
        for i, H in enumerate(jaw_hulls[g.meshname]):
            name = f"{g.meshname}_piece{i:02d}"
            child.add_mesh(name=name, uservert=[float(v) for v in np.asarray(H, float).ravel()])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=list(g.pos), quat=list(g.quat),
                          group=3, contype=g.contype, conaffinity=g.conaffinity, rgba=[0.9, 0.5, 0.1, 0.4])
        g.group, g.contype, g.conaffinity = 4, 0, 0


def load_jaw_hulls(path=None) -> dict | None:
    """{mesh name: [piece vertex arrays]} from the jaw decomposition (tools/build_jaw_collision.py), or None."""
    import json

    p = Path(path) if path else Path(__file__).resolve().parents[1] / "assets" / "gripper_collision" / "so101_jaws.json"
    if not p.exists():
        return None
    return {name: part["hulls"] for name, part in json.loads(p.read_text())["parts"].items()}


def _add_cup(body, o: SceneObject) -> None:
    """An open cup about the body's origin (the centre of its bounding box): a disc base and CUP_SEGMENTS wall
    boxes on a ring, overlapping slightly so nothing slips between them. The mass is spread over the pieces."""
    import mujoco

    r, h, t = o.size[0] / 2, o.size[2], o.wall_m
    n = CUP_SEGMENTS
    kw = dict(mass=o.mass / (n + 1), rgba=list(o.rgba), friction=[o.friction, 0.01, 0.0001], condim=4)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, t / 2, 0], pos=[0, 0, -h / 2 + t / 2], **kw)
    rc = r - t / 2                                  # the wall's mid-line radius
    half_len = r * math.tan(math.pi / n)            # tangential half-length: closes the ring at the outer face
    for k in range(n):
        a = 2 * math.pi * k / n
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[t / 2, half_len, (h - t) / 2],
                      pos=[rc * math.cos(a), rc * math.sin(a), t / 2], quat=_yaw_quat(math.degrees(a)), **kw)


def build_scene(rig, objects=(), jaw_hulls: dict | None = None, cameras=()) -> Scene:
    """The scene; `cameras`: render.SimCamera list to place in it (the simulated rig's views)."""
    import mujoco

    adir = assets_dir()
    spec = mujoco.MjSpec()
    spec.modelname = "robot_use_twin"
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1])
    wb.add_geom(name="work_surface", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01], pos=[0, 0, 0],
                rgba=[0.6, 0.55, 0.5, 1], friction=[1.0, 0.005, 0.0001])
    b = rig.board
    bx = (BOARD_BACK_X + b.front_x_m) / 2
    wb.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[(b.front_x_m - BOARD_BACK_X) / 2, b.half_width_y_m, b.thickness_m / 2],
                pos=[bx, 0, b.thickness_m / 2], rgba=[0.75, 0.7, 0.6, 1])
    for ob in rig.obstacles():
        lo, hi = np.asarray(ob["min"], float), np.asarray(ob["max"], float)
        wb.add_geom(name=ob["name"].replace(" ", "_"), type=mujoco.mjtGeom.mjGEOM_BOX, size=list((hi - lo) / 2),
                    pos=list((hi + lo) / 2), rgba=[0.3, 0.3, 0.35, 1])

    mounts = {}
    for arm in ("left", "right"):
        m = rig.arm_mount(arm)
        mounts[arm] = m
        child = mujoco.MjSpec.from_file(str(adir / "so101_new_calib_camera.xml"))
        child.meshdir = str(adir / "assets")
        if jaw_hulls:
            _swap_in_jaw_pieces(child, jaw_hulls)
        frame = wb.add_frame(pos=list(m.xyz), quat=_yaw_quat(m.yaw_deg))
        spec.attach(child, prefix=f"{arm}_", frame=frame)

    for o in (SceneObject.from_dict(o) if isinstance(o, dict) else o for o in objects):
        body = wb.add_body(name=f"obj_{o.name}", pos=list(o.xyz), quat=_yaw_quat(o.yaw_deg))
        body.add_freejoint(name=f"obj_{o.name}_free")
        if o.shape == "cup":
            _add_cup(body, o)
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in o.size], mass=o.mass,
                          rgba=list(o.rgba), friction=[o.friction, 0.01, 0.0001], condim=4)

    if cameras:
        from .render import add_cameras
        add_cameras(spec, cameras)
    model = spec.compile()
    if cameras:
        from .render import NEAR_CLIP_M
        model.vis.map.znear = NEAR_CLIP_M / model.stat.extent      # MuJoCo's near plane is relative to the extent
    scene = Scene(model, mounts=mounts)
    for arm in ("left", "right"):
        info = {"qpos": [], "dof": [], "act": []}
        for j in ARM_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}")
            if jid < 0 or aid < 0:
                raise RuntimeError(f"twin: no joint/actuator {arm}_{j}")
            info["qpos"].append(int(model.jnt_qposadr[jid]))
            info["dof"].append(int(model.jnt_dofadr[jid]))
            info["act"].append(aid)
        info["tip_site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
        scene.arms[arm] = info
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.startswith("obj_"):
            jid = model.body_jntadr[i]
            scene.objects[name[4:]] = {"body": i, "qpos": int(model.jnt_qposadr[jid])}
    return scene
