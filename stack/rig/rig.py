"""Load ``robot.yaml`` (the single measured description of the rig) and derive what consumers need.

Derived here, never stored twice:
- each arm's URDF base pose in the rig frame (pan axis at (0, +/-sep/2), base plane on the board top,
  minus the URDF's own base-origin -> pan-axis offset read from the URDF);
- the board outline (side edges from the pan axes plus the measured margin);
- obstacle boxes (the camera stand from its measured face, size and centre);
- the twin's frame offset (sim = rig + (pan-axis x in the URDF, 0, -board thickness)).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .kinematics import JOINTS, ArmMount

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RIG = REPO / "robot.yaml"


class RigError(ValueError):
    pass


@dataclass(frozen=True)
class Board:
    thickness_m: float
    front_x_m: float
    half_width_y_m: float


def _rpy(r, p, y):
    import numpy as np
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


@lru_cache(maxsize=16)
def urdf_fixed_pose(urdf_path: Path, parent: str, child: str):
    """4x4 pose of `child` in `parent` through fixed joints only (e.g. a camera link on gripper_link)."""
    import numpy as np
    root = ET.parse(urdf_path).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    T, link = np.eye(4), child
    while link != parent:
        j = by_child.get(link)
        if j is None or j.get("type") != "fixed":
            raise RigError(f"{urdf_path}: no fixed-joint chain from {parent} to {child}")
        o = j.find("origin")
        M = np.eye(4)
        M[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T, link = M @ T, j.find("parent").get("link")
    return T


def _quat_xyzw(R) -> list[float]:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = max(range(3), key=lambda k: R[k, k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    n = math.sqrt(sum(v * v for v in q))
    return [round(float(v) / n, 6) for v in q]


def urdf_pan_offset(urdf_path: Path) -> tuple[float, float, float]:
    """The shoulder_pan joint origin in the URDF base frame: where the pan axis sits relative to base_link."""
    root = ET.parse(urdf_path).getroot()
    for j in root.findall("joint"):
        if j.get("name") == "shoulder_pan":
            return tuple(float(v) for v in j.find("origin").get("xyz").split())
    raise RigError(f"{urdf_path}: no shoulder_pan joint")


class Rig:
    def __init__(self, raw: dict, path: Path):
        self.raw, self.path = raw, Path(path)
        if raw.get("version") != 1:
            raise RigError(f"{path}: unsupported robot.yaml version {raw.get('version')!r}")
        self._validate()

    # ------------------------------------------------------------------ arms
    @property
    def urdf_path(self) -> Path:
        p = Path(self.raw["arms"]["urdf"])
        return p if p.is_absolute() else REPO / p

    @property
    def pan_axis_y(self) -> float:
        return float(self.raw["arms"]["pan_axis_separation_m"]) / 2.0

    def arm(self, name: str) -> dict:
        if name not in ("left", "right"):
            raise RigError(f"unknown arm {name!r}")
        return self.raw["arms"][name]

    def arm_mount(self, name: str) -> ArmMount:
        a = self.arm(name)
        ox, oy, _ = urdf_pan_offset(self.urdf_path)
        yaw = math.radians(float(a.get("yaw_deg", 0.0)))
        c, s = math.cos(yaw), math.sin(yaw)
        pan = (0.0, self.pan_axis_y if name == "left" else -self.pan_axis_y)
        base_z = self.board.thickness_m if self.raw["arms"].get("base_plane", "board_top") == "board_top" else 0.0
        base_z += float(a.get("base_z_offset_m", 0.0))
        xyz = (pan[0] - (c * ox - s * oy), pan[1] - (s * ox + c * oy), base_z)
        return ArmMount(xyz=xyz, yaw_deg=float(a.get("yaw_deg", 0.0)),
                        signs=tuple(float(v) for v in a["joint_signs"]),
                        offsets_deg=tuple(float(v) for v in a["joint_offsets_deg"]))

    # ------------------------------------------------------------------ fixed geometry
    @property
    def board(self) -> Board:
        b = self.raw["board"]
        return Board(float(b["thickness_m"]), float(b["front_edge_x_m"]), self.pan_axis_y + float(b["side_margin_m"]))

    def obstacles(self) -> list[dict]:
        """Axis-aligned boxes in the rig frame: [{"name", "min", "max"}]."""
        out = []
        for name, ob in self.raw.get("obstacles", {}).items():
            if name == "sources":
                continue
            if "min" in ob:
                out.append({"name": name.replace("_", " "), "min": [float(v) for v in ob["min"]],
                            "max": [float(v) for v in ob["max"]]})
            elif name == "camera_stand":
                f, d, w, cy = (float(ob[k]) for k in ("front_face_x_m", "depth_x_m", "width_y_m", "centre_y_m"))
                out.append({"name": "camera stand", "min": [f - d, cy - w / 2, self.board.thickness_m],
                            "max": [f, cy + w / 2, float(ob["top_z_m"])]})
            else:
                raise RigError(f"obstacle {name!r}: give min/max, or a known parametric shape")
        return out

    def camera_file(self, name: str, key: str) -> Path:
        """A camera's calibration file (`intrinsics_file` / `extrinsics_file`): relative paths are relative to the
        repository, like the URDF."""
        c = self.camera(name)
        if key not in c:
            raise RigError(f"camera {name!r} has no {key}")
        p = Path(c[key]).expanduser()
        return p if p.is_absolute() else REPO / p

    def camera_model(self, name: str):
        """The camera's model in the rig frame, and where it came from: (CameraModel, {"lens", "pose"}), each
        "calibrated" (from its file: `intrinsics_file`, `extrinsics_file`) or "estimated" (from the entry's own
        numbers: `focal_px`/`principal_point_px`/`coeffs` under `distortion`, or `fov_h_deg`; `eye_m`,
        `tilt_down_deg`, `yaw_deg`, `roll_deg`). A file always wins over the numbers beside it."""
        from .camera_model import CameraModel

        c = self.camera(name)
        if "urdf_link" in c:
            raise RigError(f"camera {name!r} rides on an arm (urdf_link): it has no fixed pose in the rig frame")
        if "extrinsics_file" in c and "intrinsics_file" not in c:
            raise RigError(f"camera {name!r}: an extrinsics_file needs its intrinsics_file (the pose was solved "
                           f"with that lens)")
        src = {}
        if "intrinsics_file" in c:
            K, dist, size = CameraModel.intrinsics_from_file(self.camera_file(name, "intrinsics_file"))
            src["lens"] = "calibrated"
        else:
            mode = c.get("mode") or {}
            if "width" not in mode or "height" not in mode:
                raise RigError(f"camera {name!r}: no intrinsics_file, and no mode width/height to estimate it from")
            size = (int(mode["width"]), int(mode["height"]))
            d = c.get("distortion") or {}
            try:
                K, dist = CameraModel.intrinsics_estimate(size, fov_h_deg=c.get("fov_h_deg"),
                                                          focal_px=d.get("focal_px"),
                                                          principal_point_px=d.get("principal_point_px"),
                                                          coeffs=d.get("coeffs"))
            except ValueError:
                raise RigError(f"camera {name!r}: no intrinsics_file, and neither distortion.focal_px nor "
                               f"fov_h_deg to estimate the lens from") from None
            src["lens"] = "estimated"
        if "extrinsics_file" in c:
            m = CameraModel.from_files(self.camera_file(name, "intrinsics_file"),
                                       self.camera_file(name, "extrinsics_file"))
            src["pose"] = "calibrated"
            m.source = src
            return m, src
        if "eye_m" not in c or "tilt_down_deg" not in c:
            raise RigError(f"camera {name!r}: no extrinsics_file, and no eye_m and tilt_down_deg to estimate its "
                           f"pose from")
        R_cw, t_cw = CameraModel.pose_estimate(c["eye_m"], c["tilt_down_deg"], c.get("yaw_deg", 0.0),
                                               c.get("roll_deg", 0.0))
        src["pose"] = "estimated"
        m = CameraModel(K, dist, R_cw, t_cw, size)
        m.source = src
        return m, src

    def camera(self, name: str) -> dict:
        """The camera's entry. A camera given by `urdf_link` gets its mount from the URDF: `parent` gripper_link and
        `offset_m` / `rot_xyzw` (the link's pose in gripper_link, through the URDF's fixed joints)."""
        c = self.raw["cameras"].get(name)
        if c is None or name == "sources":
            raise RigError(f"unknown camera {name!r}")
        if "urdf_link" in c:
            T = urdf_fixed_pose(self.urdf_path, "gripper_link", c["urdf_link"])
            c = {**c, "parent": "gripper_link", "offset_m": [round(float(v), 6) for v in T[:3, 3]],
                 "rot_xyzw": _quat_xyzw(T[:3, :3])}
        return c

    @property
    def camera_names(self) -> list[str]:
        return [k for k in self.raw["cameras"] if k != "sources"]

    def sim_offset(self) -> tuple[float, float, float]:
        """The twin's frame: sim = rig + this (x from the arm roots, z from the arm-base plane)."""
        ox, _, _ = urdf_pan_offset(self.urdf_path)
        return (ox, 0.0, -self.board.thickness_m)

    @property
    def description(self) -> str:
        f = self.raw.get("description_file")
        return (REPO / f).read_text() if f else ""

    # ------------------------------------------------------------------ checks
    def _validate(self):
        r = self.raw
        for section in ("arms", "board", "cameras"):
            if section not in r:
                raise RigError(f"{self.path}: missing section {section!r}")
            if "sources" not in r[section]:
                raise RigError(f"{self.path}: section {section!r} has no sources")
        for arm in ("left", "right"):
            a = r["arms"].get(arm) or {}
            signs = a.get("joint_signs", [])
            if len(signs) != len(JOINTS) or any(float(s) not in (1.0, -1.0) for s in signs):
                raise RigError(f"arms.{arm}.joint_signs must be {len(JOINTS)} values of 1 or -1")
            if len(a.get("joint_offsets_deg", [])) != len(JOINTS):
                raise RigError(f"arms.{arm}.joint_offsets_deg must have {len(JOINTS)} values")
            if abs(float(a.get("base_z_offset_m", 0.0))) > 0.03:
                raise RigError(f"arms.{arm}.base_z_offset_m is a small correction (|v| <= 0.03 m)")
        if not self.urdf_path.is_file():
            raise RigError(f"URDF not found: {self.urdf_path}")


@lru_cache(maxsize=8)
def _load(path_str: str) -> Rig:
    p = Path(path_str).expanduser()
    old = p.with_name("rig.yaml")
    if p.name == "robot.yaml" and old.exists():      # the file's name until 2026-10-14 (public-repo sprint, D16)
        raise RigError(f"{old} is there beside {p}: rig.yaml was renamed robot.yaml. Keep one description "
                       f"(merge your changes into robot.yaml), then remove rig.yaml")
    return Rig(yaml.safe_load(p.read_text()), p)


def load_rig(path: Path | str | None = None) -> Rig:
    return _load(str(Path(path or DEFAULT_RIG).expanduser().resolve()))
