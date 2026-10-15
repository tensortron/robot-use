"""The ego camera in the rig frame, and the model-arm overlay drawn on its frames.

Intrinsics: ``calib/ego_intrinsics.json`` (pinhole K + 5 distortion coefficients).
Extrinsics: ``calib/ego_extrinsics.json`` from ``tools/solve_extrinsics.py``: the board's
pose in the camera (rvec, tvec) plus where the board lay in the rig frame. The rig-to-board mapping
is the one ``solve_extrinsics.place()`` uses, so the camera centre recovered here equals the
``eye_m`` that tool reported (checked in the tests).

The overlay draws each arm's skeleton from FK. If a joint's sign, the mount or the calibration is
wrong, the drawn arm does not lie on the real one: that is the Check step of docs/CALIBRATION.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ARM_COLORS = {"left": (80, 200, 255), "right": (255, 160, 60)}     # BGR


class CameraModel:
    def __init__(self, K, dist, R_cw, t_cw, image_size):
        self.K = np.asarray(K, float)
        self.dist = np.asarray(dist, float).ravel()
        self.R_cw = np.asarray(R_cw, float)              # rig -> camera rotation
        self.t_cw = np.asarray(t_cw, float).ravel()
        self.image_size = tuple(image_size)              # (w, h)
        self.source = {"lens": "calibrated", "pose": "calibrated"}   # or "estimated" (Rig.camera_model)

    @classmethod
    def from_files(cls, intrinsics: Path, extrinsics: Path) -> "CameraModel":
        import cv2

        intr = json.loads(Path(intrinsics).read_text())
        ext = json.loads(Path(extrinsics).read_text())
        K, dist = intr["pinhole"]["K"], intr["pinhole"]["dist"]
        R_cb, _ = cv2.Rodrigues(np.asarray(ext["rvec"], float))        # board -> camera
        t_cb = np.asarray(ext["tvec"], float).ravel()
        b = ext["board"]
        psi = math.radians(b["yaw_deg"])
        Xw = np.array([math.cos(psi), math.sin(psi), 0.0])
        Zw = np.array([0.0, 0.0, b["board_z_sign"]])
        R_wb = np.column_stack([Xw, np.cross(Zw, Xw), Zw])              # board -> rig (solve_extrinsics.place)
        c0 = np.array([b["corner_xy"][0], b["corner_xy"][1], 0.0])
        R_cw = R_cb @ R_wb.T
        t_cw = t_cb - R_cw @ c0
        return cls(K, dist, R_cw, t_cw, ext["image_size"])

    @staticmethod
    def intrinsics_from_file(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """(K, dist, (w, h)) from an intrinsics file (`tools/calibrate_camera.py`)."""
        intr = json.loads(Path(path).read_text())
        return (np.asarray(intr["pinhole"]["K"], float), np.asarray(intr["pinhole"]["dist"], float).ravel(),
                tuple(intr["image_size"]))

    @staticmethod
    def intrinsics_estimate(image_size, fov_h_deg=None, focal_px=None, principal_point_px=None,
                            coeffs=None) -> tuple[np.ndarray, np.ndarray]:
        """(K, dist) from best estimates: the focal length in pixels (and the principal point and distortion when
        known), or only the horizontal field of view (a pinhole centred on the image, no distortion)."""
        w, h = image_size
        if focal_px is not None:
            fx, fy = (float(focal_px), float(focal_px)) if np.isscalar(focal_px) else map(float, focal_px)
        elif fov_h_deg is not None:
            fx = fy = w / 2 / math.tan(math.radians(float(fov_h_deg)) / 2)
        else:
            raise ValueError("give focal_px or fov_h_deg")
        cx, cy = principal_point_px if principal_point_px is not None else ((w - 1) / 2, (h - 1) / 2)
        K = np.array([[fx, 0.0, float(cx)], [0.0, fy, float(cy)], [0.0, 0.0, 1.0]])
        return K, np.asarray(coeffs if coeffs is not None else [0.0] * 5, float).ravel()

    @staticmethod
    def pose_estimate(eye_m, tilt_down_deg, yaw_deg=0.0, roll_deg=0.0) -> tuple[np.ndarray, np.ndarray]:
        """(R_cw, t_cw) from a camera's position and its angles in the rig frame: `tilt_down_deg` from straight
        down (0 looks at the table below it, 90 looks level), `yaw_deg` about +z (positive toward the robot's
        left), `roll_deg` about the optical axis (positive turns the camera's image-right axis toward its
        image-down axis). With yaw and roll 0, it looks along +x and the image's right is the robot's right."""
        t, ps, r = (math.radians(float(a)) for a in (tilt_down_deg, yaw_deg, roll_deg))
        fwd = np.array([math.sin(t) * math.cos(ps), math.sin(t) * math.sin(ps), -math.cos(t)])
        right0 = np.array([math.sin(ps), -math.cos(ps), 0.0])
        right = right0 * math.cos(r) + np.cross(fwd, right0) * math.sin(r)
        down = np.cross(fwd, right)
        R_cw = np.vstack([right, down, fwd])
        return R_cw, -R_cw @ np.asarray(eye_m, float).ravel()

    def pose_angles(self) -> dict:
        """The inverse of `pose_estimate`: {eye_m, tilt_down_deg, yaw_deg, roll_deg} of this camera."""
        right, _, fwd = self.R_cw
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -fwd[2]))))
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        ps = math.radians(yaw)
        r0 = np.array([math.sin(ps), -math.cos(ps), 0.0])
        roll = math.degrees(math.atan2(float(np.dot(np.cross(r0, right), fwd)), float(np.dot(r0, right))))
        return {"eye_m": [float(v) for v in self.eye], "tilt_down_deg": tilt, "yaw_deg": yaw, "roll_deg": roll}

    @property
    def eye(self) -> np.ndarray:
        """Camera centre in the rig frame."""
        return -self.R_cw.T @ self.t_cw

    def project(self, pts_rig) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) pixels and (N,) in-front mask for rig-frame points."""
        import cv2

        pts = np.asarray(pts_rig, float).reshape(-1, 3)
        cam = pts @ self.R_cw.T + self.t_cw
        rvec, _ = cv2.Rodrigues(self.R_cw)
        px, _ = cv2.projectPoints(pts, rvec, self.t_cw, self.K, self.dist)
        return px.reshape(-1, 2), cam[:, 2] > 0.01


def _box_edges(lo, hi):
    c = [[(hi if (i >> k) & 1 else lo)[k] for k in range(3)] for i in range(8)]
    return [(c[a], c[b]) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def draw_static(frame, cam: CameraModel, obstacles=(), board=None):
    """Fixed geometry robotd believes in: obstacle boxes (red) and the board outline (yellow).
    If these do not sit on the real stand and board edge, the rig description is wrong."""
    import cv2

    for ob in obstacles:
        for p, q in _box_edges(np.asarray(ob["min"], float), np.asarray(ob["max"], float)):
            depth = (np.array([p, q]) @ cam.R_cw.T + cam.t_cw)[:, 2]
            if depth.min() < 0.10:            # at or behind the lens (the camera-head box holds the camera)
                continue
            (a, b), front = cam.project([p, q])
            if front.all():
                cv2.line(frame, tuple(int(v) for v in a), tuple(int(v) for v in b), (40, 40, 230), 1, cv2.LINE_AA)
    if board is not None:
        fx, hy, z = board
        corners = [(fx, -hy, z), (fx, hy, z), (-0.25, hy, z), (-0.25, -hy, z)]
        px, front = cam.project(corners)
        for i in range(4):
            if front[i] and front[(i + 1) % 4]:
                cv2.line(frame, tuple(int(v) for v in px[i]), tuple(int(v) for v in px[(i + 1) % 4]),
                         (0, 220, 255), 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, cam: CameraModel, arms: dict[str, dict]):
    """Draw each arm's FK skeleton. ``arms``: {name: {"keypoints": (6,3), "tip": xyz, "pitch": deg}}.

    The frame must be at the calibrated resolution; anything else is refused rather than drawn
    wrong.
    """
    import cv2

    h, w = frame.shape[:2]
    if (w, h) != cam.image_size:
        cv2.putText(frame, f"overlay off: frame {w}x{h} != calibrated {cam.image_size[0]}x{cam.image_size[1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return frame
    for name, a in arms.items():
        color = ARM_COLORS.get(name, (0, 255, 0))
        px, front = cam.project(a["keypoints"])
        p = [tuple(int(round(v)) for v in xy) for xy in px]
        for i in range(len(p) - 1):
            if front[i] and front[i + 1]:
                cv2.line(frame, p[i], p[i + 1], (0, 0, 0), 5, cv2.LINE_AA)
                cv2.line(frame, p[i], p[i + 1], color, 2, cv2.LINE_AA)
        for i, xy in enumerate(p):
            if front[i]:
                cv2.circle(frame, xy, 6 if i == len(p) - 1 else 4, color, -1, cv2.LINE_AA)
        if front[-1]:
            tip = a["tip"]
            label = f"{name} tip ({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}) m  pitch {a['pitch']:+.0f}"
            org = (p[-1][0] + 10, p[-1][1] - 10)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "MODEL overlay: the drawn arm should lie on the real one", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
