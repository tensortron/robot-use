"""Geometry in the rig frame (docs/ARCHITECTURE.md): camera pixels <-> table positions.

The rig frame: origin on the work surface (the table) midway between the two shoulder-pan axes; +x forward
over the table, +y to the robot's left, +z up; metres. Only the ego camera is calibrated (intrinsics +
pose), so these functions use it unless told otherwise.

    pixel_to_table(u, v, z=0.0)      where the ego pixel (u, v) sees a horizontal plane at height z: the
                                     table (z = 0) or the top of an object of known height
    table_to_pixel(xyz)              where a rig-frame point appears in the ego image
    ray(u, v)                        the camera centre and the unit direction through pixel (u, v)

Pixel coordinates are in the full-resolution ego image (1280 x 720): scale them up if you measured them in
a downscaled frame (see frame_scale).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

EGO = "ego"


@lru_cache(maxsize=4)
def _camera(name: str = EGO):
    from stack.rig.rig import RigError, load_rig

    try:
        return load_rig().camera_model(name)[0]       # calibrated files, or robot.yaml's estimates
    except RigError as e:
        raise ValueError(f"camera {name!r} has no model for geometry: {e}") from None


def image_size(camera: str = EGO) -> tuple[int, int]:
    """(width, height) of the calibrated image."""
    return tuple(_camera(camera).image_size)


def frame_scale(width: int, camera: str = EGO) -> float:
    """Factor to turn pixel coordinates read in a frame `width` pixels wide into full-resolution ones."""
    return image_size(camera)[0] / float(width)


def ray(u: float, v: float, camera: str = EGO) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit direction) in the rig frame of the camera ray through pixel (u, v), lens distortion removed."""
    import cv2

    cam = _camera(camera)
    n = cv2.undistortPoints(np.array([[[float(u), float(v)]]]), cam.K, cam.dist).reshape(2)
    d_cam = np.array([n[0], n[1], 1.0])
    d = cam.R_cw.T @ d_cam
    return cam.eye.copy(), d / np.linalg.norm(d)


def pixel_to_table(u: float, v: float, z: float = 0.0, camera: str = EGO) -> list[float]:
    """The rig-frame point [x, y, z] where pixel (u, v) of the camera sees the horizontal plane at height z.

    z = 0 is the table. For a point on top of an object, pass the object's height: the same pixel lands
    elsewhere on the table plane (for a 4 cm block in the reachable area, about 2 cm from where its pixel
    meets the table)."""
    o, d = ray(u, v, camera)
    if d[2] >= -1e-9:
        raise ValueError(f"pixel ({u}, {v}) looks at or above the horizon; it never meets the plane z={z}")
    s = (float(z) - o[2]) / d[2]
    if s <= 0:
        raise ValueError(f"the plane z={z} is above the camera")
    p = o + s * d
    return [float(p[0]), float(p[1]), float(z)]


def table_to_pixel(xyz, camera: str = EGO) -> list[float]:
    """Where a rig-frame point appears in the camera image, [u, v] in full-resolution pixels."""
    cam = _camera(camera)
    px, front = cam.project(np.asarray(xyz, float).reshape(1, 3))
    if not front[0]:
        raise ValueError(f"{list(xyz)} is behind the camera")
    return [float(px[0, 0]), float(px[0, 1])]


def distance(a, b) -> float:
    """Euclidean distance between two points (2-D or 3-D)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    return float(np.linalg.norm(a[:n] - b[:n]))
