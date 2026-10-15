"""Rendered cameras for the simulated rig: the ego and wrist views robotd serves, made by MuJoCo instead of a lens.

Each camera is placed where robot.yaml says the real one is, with the real one's projection, so the stack's geometry
(`pixel_to_table`, `table_to_pixel`, the overlay) holds on the rendered frames as it does on the real ones:

- **ego:** the calibrated model (calib/ego_*.json): pose, focal lengths, principal point AND lens distortion. Real
  ego frames are served as captured, distorted, and the camera model projects through the distortion; so the render
  is distorted the same way: a pinhole image is rendered on a canvas wide enough for the lens's field, then remapped
  through the calibrated distortion. Where the calibrated polynomial folds back (its distorted radius peaks at
  about 0.84, short of the frame's right corners at 0.94), it describes no ray, and those pixels are black: the
  same model's `table_to_pixel` cannot reach them either.
- **wrists:** the URDF's camera link on the gripper (the same pose robotd's wrist-view code uses) and robot.yaml's
  `fov_h_deg`; a pinhole with the principal point at the centre (the wrist lenses are not calibrated).

Rendering needs an OpenGL context. MuJoCo chooses the backend from MUJOCO_GL, which must be set before mujoco is
first imported: `cgl` on macOS, `egl` (or `osmesa`) on Linux. `find_gl_env()` probes which works here, each in a
child process (`python -m stack.twin_mujoco.render` prints it). On the Jetson `egl` renders on the GPU; the EGL error
printed when a renderer is freed at exit is teardown only. One thread owns the context and renders on demand: a
frame is rendered when one is asked for and the newest is older than one frame period, so idle cameras cost
nothing. If rendering is impossible, each camera serves a frame saying so, and its
`error` names the reason.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RENDER_FPS = 10.0                       # at most this often per camera, and only while frames are asked for
FIRST_FRAME_WAIT_S = 15.0               # latest() waits this long for a fresh frame before serving an older one
                                        # (the first ego frame takes ~1.4 s on a GPU, longer in software GL)
NEAR_CLIP_M = 0.03                      # a wrist camera sits inside its own module's mesh, which reaches 26 mm
                                        # ahead of the URDF optical frame: clip what is nearer (the jaws: ~7 cm)
CANVAS_MARGIN_PX = 4                    # extra pinhole canvas round the distorted field, for interpolation
_MJ_OPTICAL = np.diag([1.0, -1.0, -1.0])    # OpenCV optical (x right, y down, z forward) -> MuJoCo camera axes


@dataclass
class SimCamera:
    """A camera to render: its pose (in the world, or on a body), the pinhole it renders, and the output it serves.

    `K` and `dist` are the served image's model (OpenCV conventions: pixel centres at integers). For a distorted
    camera the pinhole is rendered on a larger canvas (`render_K`, `render_size`) and remapped (`maps`)."""
    name: str
    body: str | None                      # MuJoCo body the camera rides on, or None for the world
    pos: tuple                            # in that body's frame
    R_optical: np.ndarray                 # 3x3: the OpenCV optical axes in that body's frame
    size: tuple                           # served (w, h)
    K: np.ndarray
    dist: np.ndarray | None = None
    render_K: np.ndarray = field(init=False)
    render_size: tuple = field(init=False)
    maps: tuple | None = field(init=False, default=None)

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        if self.dist is None or not np.any(self.dist):
            self.dist, self.render_K, self.render_size = None, self.K, tuple(self.size)
            return
        self.dist = np.asarray(self.dist, float).ravel()
        self._build_distortion_maps()

    def _build_distortion_maps(self):
        """For every served (distorted) pixel, where it lies on an undistorted pinhole canvas with the same focal
        lengths. Pixels the distortion polynomial cannot invert (it folds back beyond the calibrated field) map to
        nothing and come out black, rather than wrong."""
        import cv2

        w, h = self.size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        norm = cv2.undistortPointsIter(pts, self.K, self.dist, None, None, crit).reshape(-1, 2)
        back, _ = cv2.projectPoints(np.column_stack([norm, np.ones(len(norm))]), np.zeros(3), np.zeros(3),
                                    self.K, self.dist)
        good = np.linalg.norm(back.reshape(-1, 2) - pts.reshape(-1, 2), axis=1) < 0.05
        fx, fy = self.K[0, 0], self.K[1, 1]
        nx, ny = norm[good, 0], norm[good, 1]
        x0 = math.floor(fx * nx.min()) - CANVAS_MARGIN_PX
        y0 = math.floor(fy * ny.min()) - CANVAS_MARGIN_PX
        x1 = math.ceil(fx * nx.max()) + CANVAS_MARGIN_PX
        y1 = math.ceil(fy * ny.max()) + CANVAS_MARGIN_PX
        self.render_size = (int(x1 - x0) + 1, int(y1 - y0) + 1)
        self.render_K = np.array([[fx, 0, -x0], [0, fy, -y0], [0, 0, 1]], float)
        mx = np.where(good, fx * norm[:, 0] - x0, -1).reshape(h, w).astype(np.float32)
        my = np.where(good, fy * norm[:, 1] - y0, -1).reshape(h, w).astype(np.float32)
        self.maps = (mx, my)

    def mj_quat(self) -> list:
        """The MuJoCo camera's orientation (w, x, y, z) in its body."""
        R = np.asarray(self.R_optical, float) @ _MJ_OPTICAL
        q = np.empty(4)
        import mujoco
        mujoco.mju_mat2Quat(q, R.ravel())
        return [float(v) for v in q]


def _quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sim_cameras(rig, sizes: dict, arms=("left", "right")) -> list[SimCamera]:
    """The rig's cameras as SimCameras. `sizes`: {camera name: (w, h)} served, from robotd's camera config.
    The ego is served at its calibrated size only: `table_to_pixel` and the overlay use the calibrated model
    unscaled, so another size would render frames the stack's geometry does not fit. A wrist camera is made only
    for an arm in `arms`."""

    out = []
    for name, (w, h) in sizes.items():
        c = rig.camera(name)
        if "urdf_link" not in c:
            cm = rig.camera_model(name)[0]                  # calibrated files, or robot.yaml's estimates
            if (w, h) != tuple(cm.image_size):
                raise ValueError(f"camera {name!r}: served at {w}x{h}, calibrated at "
                                 f"{cm.image_size[0]}x{cm.image_size[1]}; the sim renders the calibrated size only")
            out.append(SimCamera(name, None, tuple(cm.eye), cm.R_cw.T, (w, h), cm.K, cm.dist))
        else:                                               # a camera on an arm's URDF link
            arm = name.removeprefix("wrist_")
            if arm not in arms:
                continue
            fx = w / 2 / math.tan(math.radians(float(c["fov_h_deg"])) / 2)
            K = np.array([[fx, 0, (w - 1) / 2], [0, fx, (h - 1) / 2], [0, 0, 1]])
            out.append(SimCamera(name, f"{arm}_gripper", tuple(c["offset_m"]), _quat_xyzw_to_R(c["rot_xyzw"]),
                                 (w, h), K))
    return out


def add_cameras(spec, cameras) -> None:
    """Add the cameras to a MuJoCo spec (before compiling), with an offscreen buffer big enough for every canvas."""
    for cam in cameras:
        parent = spec.worldbody if cam.body is None else spec.body(cam.body)
        if parent is None:
            raise ValueError(f"camera {cam.name!r}: no body {cam.body!r} in the scene")
        w, h = cam.render_size
        K = cam.render_K
        mc = parent.add_camera(name=f"cam_{cam.name}", pos=[float(v) for v in cam.pos], quat=cam.mj_quat())
        p = 1e-6                                     # sensor units per pixel: any scale, MuJoCo uses the ratios
        mc.resolution = [w, h]
        mc.sensor_size = [w * p, h * p]
        mc.focal_length = [K[0, 0] * p, K[1, 1] * p]
        # MuJoCo's principal offset runs opposite to OpenCV's, on both axes, and its raster puts pixel centres at
        # i + 0.5 where OpenCV puts them at i (checked by rendering a point: test_render)
        mc.principal_length = [(w / 2 - (K[0, 2] + 0.5)) * p, (h / 2 - (K[1, 2] + 0.5)) * p]
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, w)
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, h)


def unavailable_frame(size, name: str, reason: str, stopped: bool = False) -> np.ndarray:
    import cv2

    w, h = size
    f = np.full((h, w, 3), 40, np.uint8)
    head = f"sim camera {name}: " + ("rendering stopped" if stopped else "no rendering here")
    hint = "see robotd's log" if stopped else "python -m stack.twin_mujoco.render finds a GL setting that works"
    for i, line in enumerate((head, reason[:90], hint)):
        cv2.putText(f, line, (20, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return f


class SimRenderer:
    """Owns the GL context (on its own thread) and renders the cameras of a SimWorld on demand.

    Started and stopped by reference count, so each SimCameraIO can start and stop it as robotd does a camera: the
    thread runs from the first start() to the last stop(), and does not restart after that."""

    def __init__(self, world, cameras, fps: float = RENDER_FPS):
        self.world = world
        self.cameras = {c.name: c for c in cameras}
        self.period = 1.0 / fps
        self.error: str | None = None
        self._cv = threading.Condition()
        # freshness runs on the monotonic clock (a wall-clock step must not leave a camera due forever);
        # frames carry the wall time of the physics state they show, which is what robotd reports
        self._frames: dict[str, tuple] = {}            # name -> (BGR frame, unix time, monotonic time) of the state
        self._wanted: dict[str, float] = {}            # name -> monotonic time a fresh frame was last asked for
        self._times: dict[str, list] = {n: [] for n in self.cameras}
        self._stop = threading.Event()
        self._thread = None
        self._users = 0
        self._stopped = False

    def start(self):
        with self._cv:
            if self._stopped:
                return
            self._users += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="sim-render", daemon=True)
                self._thread.start()

    def stop(self):
        with self._cv:
            self._users = max(0, self._users - 1)
            if self._users or self._thread is None or self._stopped:
                return
            self._stopped = True
            self._stop.set()
            self._cv.notify_all()
        self._thread.join(timeout=5)

    def latest(self, name: str):
        """(BGR frame, unix time of the physics state it shows). Asks for a fresh render if the newest shows a state
        older than one period, and waits for it (up to FIRST_FRAME_WAIT_S): a frame asked for after a motion shows
        the scene at most one period before the request, as a real camera's newest frame does."""
        cam = self.cameras[name]
        deadline = time.monotonic() + FIRST_FRAME_WAIT_S
        with self._cv:
            asked = time.monotonic()
            self._wanted[name] = asked
            self._cv.notify_all()
            while self.error is None and not self._stop.is_set():
                f = self._frames.get(name)
                if f is not None and f[2] >= asked - self.period:
                    return f[0], f[1]
                left = deadline - time.monotonic()
                if left <= 0:
                    return (f[0], f[1]) if f is not None else (None, 0.0)
                self._cv.wait(left)
            if self.error is not None:
                return (unavailable_frame(cam.size, name, self.error, stopped=self.error.startswith("render failed")),
                        time.time())
            f = self._frames.get(name)
        return unavailable_frame(cam.size, name, "the renderer was stopped", stopped=True), (f[1] if f else 0.0)

    def fps_measured(self, name: str) -> float:
        with self._cv:
            ts = list(self._times[name])
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def _run(self):
        import mujoco

        renderers = {}
        try:
            try:
                model = self.world.model
                data = mujoco.MjData(model)
                for name, cam in self.cameras.items():
                    w, h = cam.render_size
                    renderers[name] = mujoco.Renderer(model, height=h, width=w)
                log.info("sim cameras rendering (%s): %s", os.environ.get("MUJOCO_GL", "default GL"),
                         ", ".join(f"{n} {c.size[0]}x{c.size[1]}" for n, c in self.cameras.items()))
            except Exception as e:                                  # no GL here: say so, once, and serve that
                with self._cv:
                    self.error = f"{type(e).__name__}: {e}"
                    self._cv.notify_all()
                log.error("sim cameras cannot render (%s); MUJOCO_GL=%s. Frames will say so. "
                          "`python -m stack.twin_mujoco.render` finds a setting that works",
                          self.error, os.environ.get("MUJOCO_GL", "(unset)"))
                return
            while not self._stop.is_set():
                with self._cv:
                    # stale for the newest request: the rule latest() accepts a frame by
                    due = [n for n, t in self._wanted.items()
                           if self._frames.get(n, (None, 0.0, -math.inf))[2] < t - self.period]
                    if not due:
                        self._cv.wait(self.period)
                        continue
                with self.world.lock:
                    data.qpos[:] = self.world.data.qpos
                    data.time = self.world.data.time
                    wall, mono = time.time(), time.monotonic()         # the frames show this moment
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                for name in due:
                    frame = self._render(renderers[name], data, self.cameras[name])
                    with self._cv:
                        self._frames[name] = (frame, wall, mono)
                        ts = self._times[name]
                        ts.append(wall)
                        self._times[name] = [t for t in ts if wall - t < 2.0]
                        self._cv.notify_all()
        except Exception as e:
            with self._cv:
                self.error = f"render failed: {type(e).__name__}: {e}"
                self._cv.notify_all()
            log.exception("sim render thread stopped")
        finally:
            for r in renderers.values():
                try:
                    r.close()
                except Exception:                                   # EGL teardown can fail at exit: harmless
                    pass

    @staticmethod
    def _render(renderer, data, cam: SimCamera) -> np.ndarray:
        import cv2

        renderer.update_scene(data, camera=f"cam_{cam.name}")
        rgb = renderer.render()
        if cam.maps is not None:
            rgb = cv2.remap(rgb, cam.maps[0], cam.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class SimCameraIO:
    """One rendered camera with CameraIO's interface (start, latest, fps_measured, stop, error). The cameras share
    one SimRenderer, which runs while any of them is started."""

    def __init__(self, renderer: SimRenderer, name: str):
        self.renderer, self.name = renderer, name
        cam = renderer.cameras[name]
        self.width, self.height = cam.size
        self._started = False

    @property
    def error(self) -> str | None:
        if self.renderer.error is None and self.renderer._stopped:
            return "the renderer was stopped"
        return self.renderer.error

    def start(self):
        if not self._started:
            self._started = True
            self.renderer.start()

    def latest(self):
        return self.renderer.latest(self.name)

    def fps_measured(self) -> float:
        return self.renderer.fps_measured(self.name)

    def stop(self):
        if self._started:
            self._started = False
            self.renderer.stop()


# ---------------------------------------------------------------------------------------------- choosing a backend
_PROBE = """
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 1"/><geom type="box" size=".1 .1 .1"/>'
                                   '</worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 32, 32)
r.update_scene(d)
assert r.render().max() > 0
print("RENDER_OK", flush=True)
import os
os._exit(0)                  # skip GL teardown, which some EGL stacks get wrong at exit
"""
MESA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def gl_env_candidates() -> list[dict]:
    """GL settings to try, best first. An explicit MUJOCO_GL is tried alone."""
    if os.environ.get("MUJOCO_GL"):
        return [{}]
    if sys.platform == "darwin":
        return [{"MUJOCO_GL": "cgl"}]
    out = [{"MUJOCO_GL": "egl"}]
    if Path(MESA_EGL_VENDOR).exists():               # Mesa's software EGL devices (llvmpipe), if the GPU's fails
        out += [{"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": MESA_EGL_VENDOR, "MUJOCO_EGL_DEVICE_ID": str(i)}
                for i in range(4)]
    out.append({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    return out


def probe_gl(env: dict, timeout_s: float = 20.0) -> bool:
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env={**os.environ, **env}, capture_output=True,
                           text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return "RENDER_OK" in r.stdout


def find_gl_env() -> dict | None:
    """The first GL setting that renders here ({} if the current environment already does), or None."""
    for env in gl_env_candidates():
        if probe_gl(env):
            return env
    return None


if __name__ == "__main__":                      # python -m stack.twin_mujoco.render: print a working setting
    env = find_gl_env()
    print(json.dumps(env) if env is not None else "no working GL setting found")
    sys.exit(0 if env is not None else 1)
