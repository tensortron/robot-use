#!/usr/bin/env python3
"""Guided calibration of a rig: one page in the browser walks through every step, in order, and writes the results.

    .venv/bin/python tools/calibrate.py                      # then open http://127.0.0.1:8098/
    # from a laptop:  ssh -L 8098:localhost:8098 <user>@<robot-host>   then http://localhost:8098/
    .venv/bin/python tools/calibrate.py --trial        # into $ROBOT_USE_HOME/calibration/trial-<time>; servos unchanged
    .venv/bin/python tools/calibrate.py --camera-only  # the measurements and the camera; the arms are never touched

The steps (docs/CALIBRATION.md says what each one does and how to do it by hand):

  1. Arms: lerobot's calibration of each follower (torque off, the arm to the middle of its range, then every joint
     through its whole range). Writes the file lerobot and robotd read, under robot.yaml's `calibration_id`.
  2. Measurements: the distance between the arms' pan axes, the board and the camera stand, into robot.yaml.
  3. Camera estimate: the ego camera's position, tilt, yaw and field of view, into robot.yaml, with the board drawn on
     the live view from those numbers to line them up. From here the stack can run (positions approximate).
  4. Lens: the guided checkerboard capture (hold the board where the page asks), then the solve.
  5. Camera pose: one photo of the board lying flat in front of the arms, two tape measurements, the solve.
  6. Check: start the rig read-only and look at the overlay on the operator page.

A step whose result already exists is shown done; redo it from the page. Progress is kept in
$ROBOT_USE_HOME/calibration/ (the captures and state.json). robotd must be stopped: it holds the arms and cameras.

Safety: the arm step turns torque OFF (the arm falls: hold it first, as with `lerobot-calibrate`) and only reads
positions; nothing here turns torque on or moves an arm. The page is served on 127.0.0.1 unless --host says
otherwise: it can turn torque off, and it has no authentication.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import calib_steps as cs  # noqa: E402

PAGE = Path(__file__).with_name("calibrate.html")
BOUNDARY = "calibframe"
ROBOTD_PORTS = (8770, 8771, 8772)

# (form name, robot.yaml keys, label, unit, plausible range): a value outside the range is refused, never clipped -
# these feed robotd's collision checks, and a unit slip (mm typed for m) must not reach them.
MEASUREMENTS = [
    ("pan_axis_separation_m", ("arms", "pan_axis_separation_m"), "Pan axis to pan axis", "m", (0.1, 1.0)),
    ("board_thickness_m", ("board", "thickness_m"), "Board thickness", "m", (0.0, 0.1)),
    ("board_front_edge_x_m", ("board", "front_edge_x_m"), "Board front edge, ahead of the pan-axis line", "m",
     (-0.3, 0.5)),
    ("board_side_margin_m", ("board", "side_margin_m"), "Board side edge, outboard of each pan axis", "m", (0.0, 0.5)),
    ("stand_front_face_x_m", ("obstacles", "camera_stand", "front_face_x_m"),
     "Camera stand front face (negative: behind the pan-axis line)", "m", (-0.6, 0.6)),
    ("stand_depth_x_m", ("obstacles", "camera_stand", "depth_x_m"), "Camera stand depth (front to back)", "m",
     (0.005, 0.5)),
    ("stand_width_y_m", ("obstacles", "camera_stand", "width_y_m"), "Camera stand width (side to side)", "m",
     (0.005, 0.5)),
    ("stand_centre_y_m", ("obstacles", "camera_stand", "centre_y_m"),
     "Camera stand centre (0: midway between the arms; + to the robot's left)", "m", (-0.6, 0.6)),
    ("stand_top_z_m", ("obstacles", "camera_stand", "top_z_m"), "Camera stand top, above the table", "m", (0.1, 2.5)),
]
ESTIMATE = [("eye_x", "Camera x (+ forward of the pan-axis line)", "m", (-1.0, 1.0)),
            ("eye_y", "Camera y (+ to the robot's left)", "m", (-1.0, 1.0)),
            ("eye_z", "Camera height above the table", "m", (0.1, 2.5)),
            ("tilt_down_deg", "Tilt from straight down (0 = looking straight down)", "deg", (0.0, 89.0)),
            ("yaw_deg", "Yaw (+ toward the robot's left)", "deg", (-90.0, 90.0)),
            ("fov_h_deg", "Horizontal field of view", "deg", (20.0, 170.0))]
CORNER_RANGE = (-1.0, 1.0)


def number(values: dict, key: str, rng: tuple, label: str | None = None) -> float:
    """A posted number, finite and inside its plausible range, or ValueError naming the field."""
    try:
        v = float(values[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{label or key}: give a number") from None
    if not math.isfinite(v) or not rng[0] <= v <= rng[1]:
        raise ValueError(f"{label or key}: {values[key]!r} is outside {rng[0]:g}..{rng[1]:g} (metres and degrees)")
    return v


class Calibrator:
    def __init__(self, a):
        from stack import paths
        from stack.robotd.calibration import calibration_path
        from stack.robotd.config import load_config

        self.a = a
        if a.demo and not a.trial:
            a.trial = "auto"                     # a demo never writes the real robot.yaml, calibration files or servos
        if a.trial == "auto":
            a.trial = str(paths.calibration() / f"trial-{time.strftime('%Y%m%d-%H%M%S')}")
        self.trial = Path(a.trial).expanduser().resolve() if a.trial else None
        self.home = self.trial or paths.calibration()
        self.backups = self.home / "backups"
        self.home.mkdir(parents=True, exist_ok=True)
        self.cfg = load_config(a.config)
        if self.trial:
            self.rig_path = self.trial / "robot.yaml"
            if not self.rig_path.exists():
                shutil.copy2(Path(a.rig).expanduser(), self.rig_path)
            self.calib_dir = self.trial / "calib"
            self.arm_dir = self.trial / "arms"
        else:
            self.rig_path = Path(a.rig).expanduser().resolve()
            self.calib_dir = Path(a.calib_dir).expanduser().resolve()
            self.arm_dir = calibration_path("x").parent       # lerobot's robots/so_follower/, where robotd reads
        self.arm_names = list(self.cfg.arms)                   # ports from robotd's config; ids from robot.yaml
        if not self.trial and Path(self.cfg.rig.path).resolve() != self.rig_path:
            raise SystemExit(f"--rig {self.rig_path} is not the rig robotd's config reads ({self.cfg.rig.path}): the "
                             f"arms' calibration ids and robotd would disagree. Use --config with that rig, or --trial")
        self.state = cs.State(self.home / "state.json")
        self.camera = a.camera
        cam = self.cfg.cameras.get(self.camera)
        self.cam_dev, self.cam_size = (cam.path, (cam.width, cam.height)) if cam else (None, None)
        if a.demo:
            from calib_demo import DemoBus
            self.arm_bus_factory = DemoBus
        else:
            self.arm_bus_factory = cs.feetech_bus
        self.arms: dict[str, cs.ArmCalibration] = {}
        self.mode = "view"                      # what the camera view shows: view | lens | pose
        self.preview = None                     # an estimate being typed: drawn on the view
        self.guided = None
        self.pose_frame = None                  # the still the pose is solved from
        self.pose_pass1 = None
        self.latest_jpeg = None
        self.latest_raw = None
        self.cam_error = None
        self.messages = deque(maxlen=60)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._cam_thread = None
        self._preflight = (0.0, [])
        self.closed = False
        self._existing = (None, [])
        self.camera_only = bool(getattr(a, "camera_only", False))     # never an arm's serial port (task 2d)
        for name in self.arm_names:                # a run that ended with servos changed: offer the restore at once
            if self.pending_path(name).exists():
                if self.camera_only:
                    self.say(f"the {name} arm's servos may still hold a half-finished calibration from an earlier "
                             f"run: run this tool without --camera-only to restore them", False)
                else:
                    self._arm(name)
        known = {self.pending_path(n).name for n in self.arm_names}
        for f in self._pending_dir().glob("servos_before_*.json"):
            if f.name not in known:
                self.say(f"{f} is from an arm no longer in robotd's config: its servos may still hold a changed "
                         f"calibration (it records what they held before)", False)

    # ------------------------------------------------------------------ helpers
    def say(self, text, ok=True):
        self.messages.append({"t": time.strftime("%H:%M:%S"), "text": text, "ok": ok})
        print(("  " if ok else "! ") + text, flush=True)

    def rig(self):
        """robot.yaml as it is now (not load_rig's cache: this tool changes the file)."""
        import yaml

        from stack.rig.rig import Rig
        return Rig(yaml.safe_load(self.rig_path.read_text()), self.rig_path)

    def preflight(self, cached: bool = False, camera_only: bool = False) -> list[str]:
        """What blocks the hardware steps: robotd running, a device missing or held by another process. The camera
        steps need only the camera (`camera_only`): the arms may be off."""
        import os

        from launch_common import port_free

        from stack.robotd.hardware import device_holders
        out = []
        if self.a.demo:
            return out
        if cached and not camera_only and time.monotonic() - self._preflight[0] < 3.0:
            return self._preflight[1]
        busy = [p for p in ROBOTD_PORTS if not port_free(p)]
        if busy:
            out.append(f"ports {busy} are in use: is robotd (or tools/run_rig.py) running? Stop it first: it holds "
                       f"the arms and the cameras")
        devs = ([] if camera_only else [c.port for c in self.cfg.arms.values()]) + \
            ([self.cam_dev] if self.cam_dev else [])
        for d in devs:
            if not os.path.exists(d):
                out.append(f"{d} does not exist (12 V on, then the USB hub; docs/HARDWARE_SETUP.md)")
        for d, pids in device_holders(devs).items():         # this process's own camera is not counted
            out.append(f"{d} is held by pid {', '.join(map(str, pids))}")
        if not camera_only:
            self._preflight = (time.monotonic(), out)
        return out

    # ------------------------------------------------------------------ the camera
    def start_camera(self):
        if self._cam_thread is not None or self.cam_dev is None:
            return
        self._cam_thread = threading.Thread(target=self._camera_loop, name="calib-camera", daemon=True)
        self._cam_thread.start()

    def _camera_loop(self):
        import cv2
        import numpy as np
        if self.a.demo:
            cap = None
        else:
            cap = cv2.VideoCapture(self.cam_dev, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])
            if not cap.isOpened():
                self.cam_error = f"cannot open {self.cam_dev}: is another process holding it?"
                self.say(self.cam_error, False)
                return
        while not self._stop.is_set():
            if cap is None:
                frame = np.full((self.cam_size[1], self.cam_size[0], 3), 60, np.uint8)
                cv2.putText(frame, "demo: no camera", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (200, 200, 200), 2)
                time.sleep(0.1)
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                if (frame.shape[1], frame.shape[0]) != tuple(self.cam_size):
                    self.cam_error = (f"the camera gives {frame.shape[1]}x{frame.shape[0]}, robotd's config asks for "
                                      f"{self.cam_size[0]}x{self.cam_size[1]}")
            raw = frame.copy()
            with self._lock:
                self.latest_raw = raw
                mode, guided, preview = self.mode, self.guided, self.preview
            try:
                if mode == "lens" and guided is not None:
                    frame = guided.step(frame)
                elif mode == "pose":
                    frame = self._draw_pose(frame)
                else:
                    frame = self._draw_model(frame, preview)
            except Exception as e:                   # a drawing problem must not stop the view
                cv2.putText(frame, f"overlay: {e}"[:90], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ok:
                self.latest_jpeg = buf.tobytes()
        if cap is not None:
            cap.release()

    def _draw_model(self, frame, preview):
        """The rig as the camera model sees it: the board outline, the stand, the pan axes and a 5 cm grid on the
        board. Lined up with the real ones, the model is right."""
        import cv2
        import numpy as np

        from stack.rig.camera_model import draw_static
        rig = self.rig()
        model = preview
        label = "estimate being edited"
        if model is None:
            try:
                model, src = rig.camera_model(self.camera)
                label = "robot.yaml: lens " + src["lens"] + ", pose " + src["pose"]
            except Exception as e:
                cv2.putText(frame, f"no camera model yet: {e}"[:100], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (60, 180, 240), 2)
                return frame
        b = rig.board
        z = b.thickness_m
        hy = b.half_width_y_m
        for x in np.arange(0.05, b.front_x_m + 0.4, 0.05):            # grid lines across the work area
            pts = [(x, -hy, 0.0), (x, hy, 0.0)]
            (p, q), front = model.project(pts)
            if front.all():
                cv2.line(frame, tuple(int(v) for v in p), tuple(int(v) for v in q), (90, 90, 90), 1, cv2.LINE_AA)
        sep = float(rig.raw["arms"]["pan_axis_separation_m"])
        for y in (sep / 2, -sep / 2):                                   # the pan axes
            (p,), front = model.project([(0.0, y, z)])
            if front.all():
                cv2.drawMarker(frame, tuple(int(v) for v in p), (0, 140, 255), cv2.MARKER_CROSS, 18, 2)   # orange (BGR)
        draw_static(frame, model, rig.obstacles(), (b.front_x_m, hy, z))
        cv2.putText(frame, label, (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        return frame

    def _draw_pose(self, frame):
        import cv2
        from board_detect import find_board
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, _ = find_board(gray, self.a.cols, self.a.rows)
        if found:
            cv2.drawChessboardCorners(frame, (self.a.cols, self.a.rows), corners, True)
            c0 = tuple(int(v) for v in corners[0, 0])
            cx = tuple(int(v) for v in corners[self.a.cols - 1, 0])
            cv2.arrowedLine(frame, c0, cx, (0, 140, 255), 4, cv2.LINE_AA, tipLength=0.08)
            cv2.circle(frame, c0, 24, (0, 0, 255), 3)
        cv2.putText(frame, "board found: take the photo" if found else "lay the board flat, all of it in view",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 120) if found else (60, 180, 240), 2)
        return frame

    # ------------------------------------------------------------------ steps and their status
    def steps(self) -> list[dict]:
        out = []
        existing = [] if self.camera_only else self.existing_calibrations()
        ids = {n: self.arm_id(n) for n in self.arm_names}
        for name in ([] if self.camera_only else self.arm_names):
            path = self.arm_out(name)
            arm = self.arms.get(name)
            st = "restore" if arm is not None and arm.touched and arm.phase not in ("limp", "ranging") else \
                ("done" if path.exists() else "todo")
            out.append({"id": f"arm_{name}", "title": f"Arm: {name}", "status": st,
                        "detail": (f"{cs.portable(path)}, {time.strftime('%Y-%m-%d', time.localtime(path.stat().st_mtime))}"
                                   if path.exists() else f"will write {cs.portable(path)}"),
                        "arm": arm.snapshot() if arm else None, "calibration_id": self.arm_id(name),
                        "existing": [e for e in existing if Path(e["path"]) != path and
                                     e["id"] not in [i for n, i in ids.items() if n != name]]})
        out.append({"id": "measurements", "title": "Measurements", "status":
                    "done" if "measurements" in self.state.data else "check",
                    "detail": "confirmed " + self.state.data["measurements"]["done"]
                    if "measurements" in self.state.data else "robot.yaml's values shown: check each, then save"})
        try:
            c = self.rig().camera(self.camera)
        except Exception:
            c = {}
        files = "intrinsics_file" in c and "extrinsics_file" in c
        has_est = "eye_m" in c and "tilt_down_deg" in c and ("fov_h_deg" in c or "distortion" in c)
        out.append({"id": "estimate", "title": "Camera estimate", "status":
                    "done" if files else ("estimated" if has_est else "todo"),
                    "detail": "the lens and pose are calibrated (the Lens and Camera pose steps)" if files else
                    ("robot.yaml has an estimate" if has_est else "no camera numbers yet")})
        out.append({"id": "lens", "title": "Lens", "status": "done" if "intrinsics_file" in c else "todo",
                    "detail": c.get("intrinsics_file", "not calibrated"),
                    "capture": {"kept": self.guided.kept, "want": self.guided.want, "guide": self.guided.guide,
                                "status": self.guided.status} if self.guided else None})
        out.append({"id": "pose", "title": "Camera pose", "status": "done" if "extrinsics_file" in c else "todo",
                    "detail": c.get("extrinsics_file", "not calibrated"), "pass1": self.pose_pass1})
        out.append({"id": "check", "title": "Check", "status": "info",
                    "detail": "tools/run_rig.py (read-only), then the operator page's overlay"})
        return out

    def values(self) -> dict:
        rig = self.rig()
        ed = cs.RigEditor(self.rig_path)
        meas = {k: ed.get(*keys) for k, keys, *_ in MEASUREMENTS}
        c = rig.camera(self.camera) if self.camera in rig.camera_names else {}
        eye = c.get("eye_m") or [None, None, None]
        est = {"eye_x": eye[0], "eye_y": eye[1], "eye_z": eye[2], "tilt_down_deg": c.get("tilt_down_deg"),
               "yaw_deg": c.get("yaw_deg", 0.0), "fov_h_deg": c.get("fov_h_deg")}
        lens = "distortion" in c
        offsets = {n: {"joint_offsets_deg": ed.get("arms", n, "joint_offsets_deg"),
                       "base_z_offset_m": ed.get("arms", n, "base_z_offset_m")} for n in self.arm_names}
        return {"offsets": json.loads(json.dumps(offsets, default=list)),
                "measurements": {k: (float(v) if v is not None else None) for k, v in meas.items()},
                "estimate": {k: (float(v) if v is not None else None) for k, v in est.items()},
                "lens_numbers": lens}

    def status(self) -> dict:
        pf = self.preflight(cached=True, camera_only=self.camera_only)
        return {"trial": str(self.trial) if self.trial else None, "rig": cs.portable(self.rig_path),
                "camera": self.camera, "mode": self.mode, "camera_error": self.cam_error,
                "preflight": pf, "preflight_camera": pf if self.camera_only else self.preflight(camera_only=True),
                "camera_only": self.camera_only,
                "steps": self.steps(),
                "messages": list(self.messages)[-12:], "demo": bool(self.a.demo),
                "fields": {"measurements": [[k, label, u] for k, _, label, u, _ in MEASUREMENTS],
                           "estimate": [[k, label, u] for k, label, u, _ in ESTIMATE]}}

    # ------------------------------------------------------------------ actions
    def pending_path(self, name) -> Path:
        """What an arm's servos held before this tool changed them, while they may differ from it. The servos are
        physical: one place for trials and real runs alike."""
        return self._pending_dir() / f"servos_before_{name}.json"      # the physical arm on that port

    def arm_id(self, name) -> str:
        return str(self.rig().arm(name)["calibration_id"])

    def arm_out(self, name) -> Path:
        return self.arm_dir / f"{self.arm_id(name)}.json"

    def _pending_dir(self) -> Path:
        from stack import paths
        return self.home / "demo" if self.a.demo else paths.calibration()   # a demo's pretend servos stay apart

    def existing_calibrations(self) -> list[dict]:
        """lerobot calibration files of an SO-101 arm found on this machine: under lerobot's calibration folder (any
        robot type: a bimanual follower, a single follower, an older version's folder name) and, in a trial, its own
        arms folder. Each: {"path", "id", "where", "date"}."""
        from stack.robotd.calibration import calibration_dir
        roots = [r for r in [calibration_dir()] + ([self.arm_dir] if self.trial else [])
                 if str(r) not in ("", ".") and r.is_absolute()]       # an empty variable means "." in lerobot
        files = []
        for root in roots:
            if root.exists():
                try:
                    files += [f for f in sorted(root.rglob("*.json")) if "teleoperators" not in f.parts]
                except OSError:
                    continue
        key = []
        for f in dict.fromkeys(files):
            try:
                st = f.stat()
            except OSError:
                continue
            key.append((str(f), st.st_mtime, st.st_size))
        if key == self._existing[0]:             # parsed again only when a file changes
            return self._existing[1]
        out = []
        for path, mtime, size in key:
            if size > 64_000:
                continue
            try:
                vals = cs._servo_values(json.loads(Path(path).read_text()))
            except Exception:                    # not an SO-101 arm calibration (or not even JSON): skip it
                continue
            if not all(0 <= v["range_min"] < v["range_max"] <= 4095 for v in vals.values()):
                continue
            out.append({"path": path, "id": Path(path).stem, "where": cs.portable(path),
                        "date": time.strftime("%Y-%m-%d", time.localtime(mtime))})
        self._existing = (key, out)
        return out

    def _use_existing(self, name, path) -> dict:
        """An arm calibrated before (with lerobot): point robot.yaml's calibration_id at that file, copying it to where
        robotd reads when it is elsewhere. Nothing is written to the servos; robotd checks at its start that they
        hold the same limits."""
        if name not in self.arm_names:
            raise ValueError(f"unknown arm {name!r}")
        arm = self.arms.get(name)
        if arm is not None and (arm.touched or arm.phase in ("limp", "ranging")):
            raise ValueError(f"{name}: finish, cancel or restore this arm's calibration first")
        found = {e["path"]: e for e in self.existing_calibrations()}
        if path not in found:
            raise ValueError("choose one of the calibration files listed")
        src, new_id = Path(path), Path(path).stem
        for other in self.arm_names:
            if other != name and self.arm_id(other) == new_id:
                raise ValueError(f"the {other} arm already uses {new_id!r}: each arm needs its own calibration")
        target = self.arm_dir / f"{new_id}.json"
        if src.resolve() != target.resolve():
            if target.exists() and target.read_bytes() != src.read_bytes():
                raise ValueError(f"{cs.portable(target)} already exists with a different calibration: rename one "
                                 f"of them first")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        ed = cs.RigEditor(self.rig_path)
        if ed.get("arms", name, "calibration_id") != new_id:
            ed.set(("arms", name, "calibration_id"), new_id,
                   f"lerobot calibration {new_id!r} (chosen {cs.today()} in tools/calibrate.py)", f"{name}_calibration_id")
            ed.save(backup_dir=self.backups)
        self.state.mark(f"arm_{name}", used=new_id)
        return {"message": f"{name}: uses {cs.portable(target)}. robotd checks at its next start that the servos "
                           f"hold these limits; then check the overlay (the Check step)"}

    def _arm(self, name) -> cs.ArmCalibration:
        if self.camera_only:
            raise ValueError("--camera-only: this run does not use the arms")
        if name not in self.arm_names:
            raise ValueError(f"unknown arm {name!r}")
        arm = self.arms.get(name)
        if arm is not None and arm.touched:
            return arm                          # its servos still need their old calibration back: same object
        if arm is None or arm.phase in ("done", "cancelled", "error") or \
                (arm.phase == "idle" and arm.calibration_id != self.arm_id(name)):
            if arm is not None:
                arm.close()
            c = self.cfg.arms[name]
            arm = cs.ArmCalibration(name, c.port, self.arm_id(name), self.arm_out(name),
                                    restore_after=self.trial is not None, bus_factory=self.arm_bus_factory,
                                    pending_path=self.pending_path(name))
            self.arms[name] = arm
        return arm

    def act(self, d: dict) -> dict:
        do = d.get("do")
        with self._lock:
            if self.closed:
                raise ValueError("the tool is stopping")
            if self.camera_only and str(do).startswith("arm_"):
                raise ValueError("--camera-only: this run does not use the arms")
            if do in ("arm_limp", "lens_start", "pose_capture"):          # what starts using a device
                problems = self.preflight(camera_only=do != "arm_limp")
                if problems:
                    raise ValueError("; ".join(problems))
            if do == "arm_use":
                return self._use_existing(d.get("arm"), d.get("path"))
            if do == "arm_limp":
                for other, arm in self.arms.items():
                    if other != d["arm"] and arm.phase in ("limp", "ranging"):
                        raise ValueError(f"finish or cancel the {other} arm first")
                self._arm(d["arm"]).go_limp()
                return {"message": f"{d['arm']}: torque off. Move the arm to the middle of its range, then Next"}
            if do == "arm_middle":
                self._arm(d["arm"]).set_middle()
                return {"message": f"{d['arm']}: recording. Move every joint except the wrist roll through its whole "
                                   f"range, then Finish"}
            if do == "arm_finish":
                arm = self._arm(d["arm"])
                arm.finish()
                arm.close()
                self.say(f"{d['arm']}: {arm.message}")
                if not self.trial:
                    self.say("the arm's zero may have moved: check the overlay (the Check step), and robot.yaml's joint offsets "
                             "if the drawn arm sits off the real one", True)
                return {"message": arm.message}
            if do == "arm_cancel":
                arm = self.arms.get(d["arm"])
                if arm is not None:
                    arm.cancel()
                    arm.close()
                    return {"message": f"{d['arm']}: {arm.message}"}
                return {"message": "nothing to cancel"}
            if do == "measurements":
                return self._save_measurements(d["values"])
            if do == "estimate_preview":
                self.preview = self._estimate_model(d["values"])
                self.mode = "view"
                return {"message": "drawn on the view"}
            if do == "estimate_save":
                return self._save_estimate(d["values"])
            if do == "view":
                self.mode, self.preview = "view", None
                return {"message": "showing robot.yaml's camera model"}
            if do == "lens_start":
                from guided_capture import GuidedCapture
                folder = self.home / f"{self.camera}_lens_{time.strftime('%Y%m%d-%H%M%S')}"
                self.guided = GuidedCapture(self.a.cols, self.a.rows, self.a.squares_mm, str(folder), want=27,
                                            log=lambda m: self.say(m.strip()))
                self.mode = "lens"
                return {"message": f"capturing into {cs.portable(folder)}: hold the board where the view asks"}
            if do == "lens_solve":
                return self._solve_lens()
            if do == "pose_start":
                self.mode, self.pose_pass1, self.pose_frame = "pose", None, None
                return {"message": "lay the board flat in front of the arms, all of it in view"}
            if do == "pose_capture":
                return self._pose_capture()
            if do == "pose_solve":
                yaw = number(d, "yaw", (0, 359), "the arrow")
                if yaw not in (0, 90, 180, 270):
                    raise ValueError("the arrow: 0, 90, 180 or 270")
                return self._solve_pose(number(d, "corner_x", CORNER_RANGE, "Corner x"),
                                        number(d, "corner_y", CORNER_RANGE, "Corner y"), yaw)
            if do == "redo":
                self.state.clear(d["step"])
                return {"message": f"{d['step']}: cleared"}
        raise ValueError(f"unknown action {do!r}")

    def _save_measurements(self, values: dict) -> dict:
        ed = cs.RigEditor(self.rig_path)
        src = f"measured {cs.today()} (tools/calibrate.py, tape)"
        nums = {k: number(values, k, rng, label) for k, _, label, _, rng in MEASUREMENTS}   # all checked first
        half = nums["pan_axis_separation_m"] / 2
        comments = {   # the lines whose comments repeat a number: written afresh with it
            "pan_axis_separation_m": f"left pan axis at y = +{half:.5f}, right at y = -{half:.5f}, both at x = 0",
            "board_side_margin_m": f"outboard of each pan axis -> side edges at y = +/-({half:.5f} + "
                                   f"{round(nums['board_side_margin_m'], 4):g})"}
        for k, keys, _, _, _ in MEASUREMENTS:
            v = nums[k]
            old = ed.get(*keys)
            if old is not None and abs(float(old) - v) < 1e-9 and k not in comments:
                continue
            ed.set(keys, round(v, 4), src if old is None or abs(float(old) - v) >= 1e-9 else None,
                   keys[1] if keys[0] == "obstacles" else keys[-1], comment=comments.get(k))
        if values.get("new_rig"):               # this repository's offsets are this rig's arms', not a new rig's
            for name in self.arm_names:
                ed.set(("arms", name, "joint_offsets_deg"), [0.0] * 5)
                ed.set(("arms", name, "base_z_offset_m"), 0.0)
            reset = (f"reset to 0 {cs.today()} for a new rig (tools/calibrate.py); measure them on the overlay "
                     f"(docs/CALIBRATION.md, Check)")
            ed.set(("arms", "sources", "joint_offsets_deg"), reset)
            ed.set(("arms", "sources", "base_z_offset_m"), reset)
        ed.save(backup_dir=self.backups)
        self.state.mark("measurements")
        return {"message": f"saved into {cs.portable(self.rig_path)}"}

    def _estimate_model(self, values: dict):
        from stack.rig.camera_model import CameraModel
        v = {k: number(values, k, rng, label) for k, label, _, rng in ESTIMATE}
        v["use_lens_numbers"] = bool(values.get("use_lens_numbers"))
        c = self.rig().camera(self.camera)
        size = self.cam_size or (int(c["mode"]["width"]), int(c["mode"]["height"]))
        d = c.get("distortion") if v.get("use_lens_numbers") else None
        if "intrinsics_file" in c and v.get("use_lens_numbers"):
            K, dist, _ = CameraModel.intrinsics_from_file(self.rig().camera_file(self.camera, "intrinsics_file"))
        else:
            K, dist = CameraModel.intrinsics_estimate(size, fov_h_deg=float(v["fov_h_deg"]),
                                                      focal_px=(d or {}).get("focal_px"),
                                                      principal_point_px=(d or {}).get("principal_point_px"),
                                                      coeffs=(d or {}).get("coeffs"))
        R, t = CameraModel.pose_estimate([float(v["eye_x"]), float(v["eye_y"]), float(v["eye_z"])],
                                         float(v["tilt_down_deg"]), float(v.get("yaw_deg") or 0.0))
        return CameraModel(K, dist, R, t, size)

    def _save_estimate(self, values: dict) -> dict:
        self._estimate_model(values)                  # refuses numbers that make no camera
        v = {k: number(values, k, rng, label) for k, label, _, rng in ESTIMATE}
        v["use_lens_numbers"] = bool(values.get("use_lens_numbers"))
        ed = cs.RigEditor(self.rig_path)
        cam = ("cameras", self.camera)
        src = f"estimated {cs.today()} (tools/calibrate.py, tape and the live view)"
        ed.set(cam + ("eye_m",), [round(float(v[k]), 3) for k in ("eye_x", "eye_y", "eye_z")])
        t = round(float(v["tilt_down_deg"]), 1)
        ed.set(cam + ("tilt_down_deg",), t, comment=f"from straight down ({90 - t:.1f} below horizontal)")
        ed.set(cam + ("yaw_deg",), round(float(v.get("yaw_deg") or 0.0), 1))
        ed.set(cam + ("roll_deg",), 0.0)
        ed.set(cam + ("fov_h_deg",), round(float(v["fov_h_deg"]), 1))
        if not v.get("use_lens_numbers"):
            ed.delete(cam + ("distortion",))
            ed.delete(cam + ("intrinsics_file",))
        ed.delete(cam + ("extrinsics_file",))         # a file would win over the estimate
        ed.set(("cameras", "sources", self.camera), src)
        ed.save(backup_dir=self.backups)
        self.preview = None
        self.state.mark("estimate")
        return {"message": f"saved into {cs.portable(self.rig_path)}; robotd uses it at its next start"}

    def _solve_lens(self) -> dict:
        import calibrate_camera
        if self.guided is None or self.guided.kept < 8:
            raise ValueError("capture first: at least 8 frames (27 for a complete set)")
        self.mode = "view"                            # stop banking frames into the folder being solved
        time.sleep(0.3)
        folder = Path(self.guided.capture_to)
        try:
            res = calibrate_camera.run(SimpleNamespace(frames=str(folder), squares_mm=self.a.squares_mm,
                                                       cols=self.a.cols, rows=self.a.rows, out=str(folder)))
        except SystemExit as e:
            raise ValueError(str(e)) from None
        ph = res["pinhole"]
        problems = []
        if ph["rms"] > 1.0:
            problems.append(f"reprojection error {ph['rms']:.2f} px (want under 1)")
        if ph["tilt_max_deg"] < 20.0:
            problems.append(f"the board was never tilted beyond {ph['tilt_max_deg']:.0f} deg: the focal length "
                            f"cannot be trusted; recapture with the board angled 20-45 deg")
        if problems:
            self.say("lens: " + "; ".join(problems), False)
            return {"message": "not saved: " + "; ".join(problems), "result": ph}
        intr, _, intr_name, _ = cs.camera_entry_paths(self.calib_dir, self.camera)
        res["frames_dir"] = cs.portable(folder)
        intr.parent.mkdir(parents=True, exist_ok=True)
        tmp = intr.with_name(intr.name + ".tmp")        # into place only once robot.yaml is saved
        try:
            return self._save_lens(res, ph, intr, intr_name, tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def _save_lens(self, res, ph, intr, intr_name, tmp) -> dict:
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        ed = cs.RigEditor(self.rig_path)
        cam = ("cameras", self.camera)
        ed.set(cam + ("intrinsics_file",), intr_name)
        ed.set(cam + ("fov_h_deg",), round(ph["fov_h_deg"], 1))
        ed.set(cam + ("distortion",), {"model": "opencv_pinhole"})
        ed.set(cam + ("distortion", "focal_px"), [round(ph["fx"], 1), round(ph["fy"], 1)])
        ed.set(cam + ("distortion", "principal_point_px"), [round(ph["cx"], 1), round(ph["cy"], 1)])
        ed.set(cam + ("distortion", "coeffs"), [round(x, 5) for x in ph["dist"]])
        if ed.get(*cam, "extrinsics_file") is not None:
            ed.delete(cam + ("extrinsics_file",))     # that pose was solved with the old lens
            self.say("the previous camera-pose file was solved with the old lens: redo the Camera pose step", False)
        ed.set(("cameras", "sources", f"{self.camera}_lens"),
               f"measured {cs.today()} - intrinsics checkerboard {res['n_frames']} frames rms {ph['rms']:.3f} px "
               f"(tools/calibrate.py)")
        ed.save(backup_dir=self.backups)
        cs.replace_file(tmp, intr, self.backups)
        self.mode = "view"
        self.state.mark("lens", rms=ph["rms"])
        return {"message": f"lens saved: {res['n_frames']} frames, error {ph['rms']:.3f} px, field of view "
                           f"{ph['fov_h_deg']:.1f} deg", "result": ph}

    def _pose_capture(self) -> dict:
        import cv2
        import solve_extrinsics
        with self._lock:
            raw = None if self.latest_raw is None else self.latest_raw.copy()
        if raw is None:
            raise ValueError("no camera frame yet")
        c = self.rig().camera(self.camera)
        if "intrinsics_file" not in c:
            raise ValueError("calibrate the lens first (the Lens step): the pose cannot be separated from the focal length")
        folder = self.home / f"{self.camera}_pose_{time.strftime('%Y%m%d-%H%M%S')}"
        folder.mkdir(parents=True, exist_ok=True)
        img = folder / "board_on_table.png"
        cv2.imwrite(str(img), raw)
        try:
            p1 = solve_extrinsics.run(self._pose_args(img, None, None, None))
        except SystemExit as e:
            raise ValueError(str(e)) from None
        self.pose_frame, self.pose_pass1 = img, {"corner_px": p1["corner_px"], "arrow_to_px": p1["arrow_to_px"],
                                                 "check": "/pose_check.jpg"}
        return {"message": "photo taken: measure the circled corner and read the arrow"}

    def _pose_args(self, img, x, y, yaw):
        intr = self.rig().camera_file(self.camera, "intrinsics_file")
        return SimpleNamespace(image=str(img), intrinsics=str(intr), squares_mm=self.a.squares_mm, cols=self.a.cols,
                               rows=self.a.rows, corner_x=x, corner_y=y, board_yaw_deg=yaw,
                               config_plane_offset_m=0.0, out=str(Path(img).parent))

    def _solve_pose(self, x, y, yaw) -> dict:
        import solve_extrinsics
        if self.pose_frame is None:
            raise ValueError("take the photo first")
        try:
            res = solve_extrinsics.run(self._pose_args(self.pose_frame, x, y, yaw))
        except SystemExit as e:
            raise ValueError(str(e)) from None
        alts = res.pop("alternatives", {})
        if res["implausible"]:
            good = [k for k, v in alts.items() if v["plausible"]]
            hint = f" The arrow reading that gives a plausible camera: {good[0]} deg." if len(good) == 1 else ""
            raise ValueError("this pose is implausible for a camera between the arms: " + "; ".join(res["implausible"])
                             + ". The likeliest cause is the arrow's direction." + hint)
        if res["reprojection_rms_px"] > 1.0:
            raise ValueError(f"poor fit ({res['reprojection_rms_px']:.2f} px): is the board flat, and the squares "
                             f"{self.a.squares_mm:g} mm?")
        _, ext, _, ext_name = cs.camera_entry_paths(self.calib_dir, self.camera)
        intr = self.rig().camera_file(self.camera, "intrinsics_file")   # the lens the pose was solved with
        res["image"], res["intrinsics"] = cs.portable(self.pose_frame), cs.portable(intr)
        ext.parent.mkdir(parents=True, exist_ok=True)
        tmp = ext.with_name(ext.name + ".tmp")          # into place only once robot.yaml is saved
        try:
            return self._save_pose(res, intr, ext, ext_name, tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def _save_pose(self, res, intr, ext, ext_name, tmp) -> dict:
        from stack.rig.camera_model import CameraModel
        tmp.write_text(json.dumps(res, indent=1) + "\n")
        model = CameraModel.from_files(intr, tmp)
        ang = model.pose_angles()
        ed = cs.RigEditor(self.rig_path)
        cam = ("cameras", self.camera)
        ed.set(cam + ("extrinsics_file",), ext_name)
        ed.set(cam + ("eye_m",), [round(v, 3) for v in ang["eye_m"]])
        t = round(ang["tilt_down_deg"], 1)
        ed.set(cam + ("tilt_down_deg",), t, comment=f"from straight down ({90 - t:.1f} below horizontal)")
        ed.set(cam + ("yaw_deg",), round(ang["yaw_deg"], 1))
        ed.set(cam + ("roll_deg",), round(ang["roll_deg"], 1))
        ed.set(("cameras", "sources", self.camera),
               f"measured {cs.today()} - extrinsics solve_extrinsics rms {res['reprojection_rms_px']:.3f} px, one "
               f"board placement (tools/calibrate.py)")
        ed.save(backup_dir=self.backups)
        cs.replace_file(tmp, ext, self.backups)
        self.mode = "view"
        self.state.mark("pose", rms=res["reprojection_rms_px"])
        e = ang["eye_m"]
        return {"message": f"pose saved: camera at ({e[0]:+.3f}, {e[1]:+.3f}, {e[2]:.3f}) m, tilt "
                           f"{ang['tilt_down_deg']:.1f} deg, yaw {ang['yaw_deg']:+.1f} deg; error "
                           f"{res['reprojection_rms_px']:.3f} px"}

    def close(self):
        """After the server has stopped: wait for an action in progress (the lock), then put back what an unfinished
        arm step changed."""
        self._stop.set()
        with self._lock:                                # held to the end: no action can start in between
            self.closed = True
            for arm in self.arms.values():
                try:
                    arm.close()                      # a recording in progress: the servos get their old values
                    if arm.touched:
                        print(f"! {arm.name}: {arm.message}", flush=True)
                except Exception as e:
                    print(f"! {arm.name}: close: {e}", flush=True)


def make_handler(cal: Calibrator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/status":
                try:
                    return self._send(200, json.dumps(cal.status()).encode())
                except Exception as e:           # the page must keep working (a Restore may be waiting)
                    return self._send(200, json.dumps({"error": f"status: {e}"}).encode())
            if self.path == "/values":
                return self._send(200, json.dumps(cal.values()).encode())
            if self.path.startswith("/pose_check.jpg"):
                f = cal.pose_frame.parent / "extrinsics_check.jpg" if cal.pose_frame else None
                if f is None or not f.exists():
                    return self._send(404, b"{}")
                return self._send(200, f.read_bytes(), "image/jpeg")
            if self.path == "/stream":
                cal.start_camera()
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while not cal._stop.is_set():
                        frame = cal.latest_jpeg
                        if frame is not None and frame is not last:
                            last = frame
                            self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                             f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")
                        else:
                            time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self._send(404, b"{}")

        def do_POST(self):
            from urllib.parse import urlparse

            from stack.robotd.http import _host_allowed
            if self.path != "/action":
                return self._send(404, b"{}")
            # As robotd's own server: another web page open in the same browser can POST here without asking (a
            # "simple" request), or reach it by DNS rebinding. Require JSON (which forces a CORS preflight this
            # server never answers), a same-origin Origin when there is one, and a Host that is an IP or localhost.
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            origin = self.headers.get("Origin")
            if ctype != "application/json" or (origin and urlparse(origin).netloc != (self.headers.get("Host") or "")) \
                    or not _host_allowed(self.headers.get("Host") or ""):
                return self._send(403, json.dumps({"ok": False, "error": "refused: not from this page"}).encode())
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= 64_000:
                return self._send(413, b"{}")
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
                out = {"ok": True, **cal.act(d)}
                if out.get("message"):
                    cal.say(out["message"])
            except Exception as e:
                out = {"ok": False, "error": f"{e}"}
                cal.say(str(e), False)
            self._send(200, json.dumps(out, default=float).encode())
    return Handler


def main() -> int:
    from stack.robotd.config import DEFAULT_CONFIG
    from stack.rig.rig import DEFAULT_RIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="where the page listens (default local only: use an SSH "
                                                        "tunnel from another machine)")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--trial", nargs="?", const="auto",
                    help="write a copy of robot.yaml, the camera files and the arms' files into this folder (default "
                         "$ROBOT_USE_HOME/calibration/trial-<time>), and give the servos their previous calibration "
                         "back after the arm step. Keep it in the data folder: robot.yaml holds the camera's pose, "
                         "which the operating agent must not read")
    ap.add_argument("--rig", default=str(DEFAULT_RIG), help="the robot.yaml to update (default the repository's)")
    ap.add_argument("--calib-dir", default=str(REPO / "calib"), help="where the camera's calibration files go")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="robotd's config: the arms' ports and ids, cameras")
    ap.add_argument("--camera", default="ego", help="the fixed camera to calibrate")
    ap.add_argument("--squares-mm", type=float, default=25.0, help="the printed board's square, measured, mm")
    ap.add_argument("--cols", type=int, default=9, help="the board's INNER corners across")
    ap.add_argument("--rows", type=int, default=6, help="the board's INNER corners down")
    ap.add_argument("--demo", action="store_true", help="no hardware: a blank camera and simulated arms, writing "
                                                        "into a trial folder only (to try the page)")
    ap.add_argument("--camera-only", action="store_true",
                    help="the measurements and the camera steps only: no arm steps, and no arm's serial port is ever "
                         "opened (the arms may be off)")
    a = ap.parse_args()
    cal = Calibrator(a)
    if a.demo:
        cal.cam_size = cal.cam_size or (1280, 720)
        cal.cam_dev = cal.cam_dev or "demo"
    problems = cal.preflight(camera_only=cal.camera_only)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(cal))
    srv.daemon_threads = True
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):       # SIGHUP: the SSH session dropped
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"calibration page: http://{a.host}:{a.port}/" + ("   (TRIAL: writes into " + str(cal.home) + ")"
                                                              if cal.trial else ""), flush=True)
    for p in problems:
        print(f"! {p}", flush=True)
    stop.wait()
    srv.shutdown()                              # no new action starts
    cal.close()                                 # waits for one in progress; before any print: after a hangup the
    try:                                        # terminal is gone and writing to it raises
        print("stopped", flush=True)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
