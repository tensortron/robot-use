"""Arm and camera I/O for robotd, plus mocks with the same interface for offline tests.

Arms go through lerobot's ``FeetechMotorsBus`` (so calibration and normalisation are lerobot's own)
with the port handler hardened by ``busio`` so no transaction can block forever. robotd in step 2
never enables torque and never writes a goal; the only write is torque *off*, from the operator.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import busio
from ..rig.kinematics import ARM_JOINTS

log = logging.getLogger(__name__)

CURRENT_MA_PER_UNIT = 6.5        # STS3215 Present_Current
LOAD_PCT_PER_UNIT = 0.1          # STS3215 Present_Load, signed (lerobot decodes the sign bit)


def device_holders(paths) -> dict[str, list[int]]:
    """{path: [pids]} of processes (other than this one) holding each device node open."""
    out = {}
    for p in paths:
        real = os.path.realpath(p)
        if not os.path.exists(real):
            continue
        try:
            r = subprocess.run(["fuser", real], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        pids = [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]
        if pids:
            out[p] = pids
    return out


@dataclass
class ArmReading:
    q: np.ndarray                    # (6,) degrees x5 + gripper 0..100
    load_pct: np.ndarray             # (6,) signed, % of max torque
    current_ma: np.ndarray           # (6,)
    read_ms: float


class ArmIO:
    """One follower arm on its Feetech bus, read-only apart from torque off."""

    def __init__(self, name: str, port: str, calibration_id: str):
        self.name, self.port, self.calibration_id = name, port, calibration_id
        self.bus = None
        self.calibration = None

    def _load_calibration(self):
        from lerobot.motors import MotorCalibration

        from .calibration import calibration_path

        path = calibration_path(self.calibration_id)            # the same file the joint limits come from
        raw = json.loads(path.read_text())
        return {j: MotorCalibration(**raw[j]) for j in ARM_JOINTS}

    def connect(self):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        self.calibration = self._load_calibration()
        motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
                  for i, j in enumerate(ARM_JOINTS)}
        self.bus = FeetechMotorsBus(port=self.port, motors=motors, calibration=self.calibration)
        busio.harden_port_handler(self.bus.port_handler)
        self.bus.connect(handshake=True)

    def firmware_limits(self) -> dict[str, dict]:
        """Level 0 check: servo Min/Max_Position_Limit against the calibration file."""
        out = {}
        for j in ARM_JOINTS:
            lo = self.bus.read("Min_Position_Limit", j, normalize=False, num_retry=2)
            hi = self.bus.read("Max_Position_Limit", j, normalize=False, num_retry=2)
            c = self.calibration[j]
            out[j] = {"servo": [int(lo), int(hi)], "calibration": [c.range_min, c.range_max],
                      "match": int(lo) == c.range_min and int(hi) == c.range_max}
        return out

    def torque_state(self) -> list[int]:
        return [int(self.bus.read("Torque_Enable", j, num_retry=2)) for j in ARM_JOINTS]

    def read(self) -> ArmReading:
        t0 = time.perf_counter()
        try:
            pos = self.bus.sync_read("Present_Position", num_retry=1)
            load = self.bus.sync_read("Present_Load", normalize=False, num_retry=1)
            cur = self.bus.sync_read("Present_Current", normalize=False, num_retry=1)
        except Exception:
            busio.release_busy(self.bus.port_handler)
            raise
        return ArmReading(np.array([pos[j] for j in ARM_JOINTS], float),
                          np.array([load[j] for j in ARM_JOINTS], float) * LOAD_PCT_PER_UNIT,
                          np.array([cur[j] for j in ARM_JOINTS], float) * CURRENT_MA_PER_UNIT,
                          (time.perf_counter() - t0) * 1000)

    def torque_off(self):
        try:
            self.bus.disable_torque(num_retry=2)
        except Exception:
            busio.release_busy(self.bus.port_handler)
            raise

    # ---------------------------------------------------------------- motion (step 3)
    def configure_for_motion(self):
        """lerobot SOFollower.configure(): position mode, PID, gripper current/torque protection.

        Torque stays OFF afterwards. (lerobot's `torque_disabled()` context re-enables torque on exit, with
        whatever goals the servos last held - on the rig on 2026-10-03 that drove the wrist ~22 deg toward a
        stale goal and the current spike browned out the arm. hold_here() is the only place torque comes on.)
        """
        from lerobot.motors.feetech import OperatingMode

        try:
            self.bus.disable_torque(num_retry=2)
            self.bus.configure_motors()
            for j in ARM_JOINTS:
                self.bus.write("Operating_Mode", j, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", j, 16)
                self.bus.write("I_Coefficient", j, 0)
                self.bus.write("D_Coefficient", j, 32)
            self.bus.write("Max_Torque_Limit", "gripper", 500)
            self.bus.write("Protection_Current", "gripper", 250)
            self.bus.write("Overload_Torque", "gripper", 25)
        except Exception:
            busio.release_busy(self.bus.port_handler)
            raise

    def hold_here(self, q=None) -> None:
        """Stiffen exactly where the arm is: read the present position from the servos, write it as the goal in
        raw ticks (no unit conversion in between), and only then enable torque. `q` is ignored on purpose: a
        reading from a tick ago may already be stale for an arm someone is holding."""
        try:
            present = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
            self.bus.sync_write("Goal_Position", present, normalize=False)
            self.bus.enable_torque(num_retry=2)
        except Exception:
            busio.release_busy(self.bus.port_handler)
            raise

    def write_goal(self, q) -> None:
        try:
            self.bus.sync_write("Goal_Position", dict(zip(ARM_JOINTS, map(float, q))))
        except Exception:
            busio.release_busy(self.bus.port_handler)
            raise

    def close(self):
        if self.bus is not None and self.bus.is_connected:
            try:
                self.bus.disconnect(disable_torque=False)     # never drop an arm on shutdown
            except Exception as e:
                log.warning("%s: disconnect: %s", self.name, e)
                busio.close_servo_port(self.bus.port_handler)


class CameraIO:
    """Grabs frames on its own thread and keeps the newest (BGR) with its timestamp."""

    def __init__(self, name: str, path: str, width: int, height: int, fps: int = 30):
        self.name, self.path, self.width, self.height, self.fps = name, path, width, height, fps
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._times: list[float] = []
        self._stop = threading.Event()
        self._thread = None
        self.error: str | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.name}", daemon=True)
        self._thread.start()

    def _open(self):
        import cv2

        c = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        c.set(cv2.CAP_PROP_FPS, self.fps)
        return c

    def _run(self):
        cap = None
        backoff = 1.0
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                if not os.path.exists(self.path):        # unplugged: do not let OpenCV warn every second
                    self.error = f"{self.path} does not exist (unplugged?)"
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
                cap = self._open()
                if not cap.isOpened():
                    self.error = f"cannot open {self.path}"
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 10.0)
                    continue
                backoff = 1.0
            ok, frame = cap.read()
            now = time.time()
            if not ok:
                self.error = "read failed"
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            self.error = None
            with self._lock:
                self._frame, self._stamp = frame, now
                self._times.append(now)
                self._times = [t for t in self._times if now - t < 2.0]
        if cap is not None:
            cap.release()

    def latest(self):
        """(frame BGR or None, unix time of capture)."""
        with self._lock:
            return self._frame, self._stamp

    def fps_measured(self) -> float:
        with self._lock:
            ts = list(self._times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)


# --------------------------------------------------------------------------- mocks
class MockArmIO:
    """Offline stand-in with simple servo physics.

    With torque on, each joint follows its goal with a first-order lag (``tau``) under a speed limit;
    with torque off it stays put (as if resting). ``block[joint] = (value, direction)`` puts an
    obstacle in the way: the joint cannot pass ``value`` moving in ``direction`` (+1/-1), and its
    current rises with how hard it is pushed. ``fail_reads`` injects bus failures.
    """

    BASE_MA = 40.0
    MA_PER_DEG_PUSH = 60.0

    def __init__(self, name: str, q=None, tau: float = 0.04, max_speed: float = 300.0):
        self.name = name
        self.q = np.array(q if q is not None else [0.0, -95.0, 90.0, 50.0, 0.0, 2.0], float)
        self.goal = self.q.copy()
        self.torque = [1] * 6
        self.fail_reads = 0
        self.fail_writes = 0
        self.limits_match = True
        self.tau, self.max_speed = tau, max_speed
        self.block: dict[str, tuple[float, int]] = {}
        self.configured = 0
        self.writes = 0
        self._t = time.monotonic()
        self._lock = threading.Lock()

    def _step(self):
        now = time.monotonic()
        dt, self._t = min(now - self._t, 0.1), now
        if not any(self.torque) or dt <= 0:
            return
        err = self.goal - self.q
        v = np.clip(err * (1 - np.exp(-dt / self.tau)) / dt, -self.max_speed, self.max_speed)
        q = self.q + v * dt
        for j, (val, d) in self.block.items():
            i = ARM_JOINTS.index(j)
            if d > 0 and q[i] > val:
                q[i] = max(val, self.q[i]) if self.q[i] <= val else self.q[i]
            if d < 0 and q[i] < val:
                q[i] = min(val, self.q[i]) if self.q[i] >= val else self.q[i]
        self.q = q

    def connect(self):
        pass

    def firmware_limits(self):
        return {j: {"servo": [0, 4095], "calibration": [0, 4095] if self.limits_match else [1, 4095],
                    "match": self.limits_match} for j in ARM_JOINTS}

    def torque_state(self):
        return list(self.torque)

    def read(self) -> ArmReading:
        with self._lock:
            if self.fail_reads:
                self.fail_reads -= 1
                raise ConnectionError(f"{self.name}: injected read failure")
            self._step()
            push = np.abs(self.goal - self.q) if any(self.torque) else np.zeros(6)
            cur = self.BASE_MA + self.MA_PER_DEG_PUSH * np.maximum(push - 0.5, 0)
            load = np.clip(push * 10, 0, 100) * np.sign(self.goal - self.q)
            return ArmReading(self.q.copy(), load, cur, 1.0)

    def torque_off(self):
        with self._lock:
            self.torque = [0] * 6

    def configure_for_motion(self):
        with self._lock:
            if any(self.torque):
                raise RuntimeError("configure_for_motion with torque on would drop the arm")
            self.configured += 1

    def hold_here(self, q):
        with self._lock:
            self.goal = np.asarray(q, float).copy()
            self.torque = [1] * 6
            self._t = time.monotonic()

    def write_goal(self, q):
        with self._lock:
            if self.fail_writes:
                self.fail_writes -= 1
                raise ConnectionError(f"{self.name}: injected write failure")
            self._step()
            self.goal = np.asarray(q, float).copy()
            self.writes += 1

    def close(self):
        pass


class MockCameraIO:
    """Offline stand-in: a grey frame with the camera's name and a moving bar."""

    def __init__(self, name: str, width: int = 640, height: int = 360, fps: int = 30, **_):
        self.name, self.width, self.height = name, width, height
        self.error = None

    def start(self):
        pass

    def latest(self):
        import cv2

        f = np.full((self.height, self.width, 3), 90, np.uint8)
        x = int((time.time() * 100) % self.width)
        f[:, x:x + 4] = 200
        cv2.putText(f, f"mock {self.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return f, time.time()

    def fps_measured(self):
        return 30.0

    def stop(self):
        pass
