"""The twin's hardware: a MuJoCo world stepping in real time, and per-arm adapters with the ArmIO interface.

robotd runs unchanged on top of SimArmIO: the same planner, safety checks, contact monitor, supervisor and
API. Units match the real servos as lerobot reports them: arm joints in degrees (through the rig's joint
signs and offsets), the gripper 0..100, load in % and a current estimate in mA.

Fidelity, stated plainly:
- kinematics: exact (the same model as robotd, checked to < 0.01 mm);
- servo tracking: the upstream MJCF position actuators (gains derived for lerobot's P = 16), saturating at
  the servo's force range, so pushing into something produces the lag and effort the contact monitor uses;
- current: an estimate from the LOAD torque: the actuator force minus what the joint's own damping and
  inertia take (the model's damping stands in for the servo's speed limit, not a load), scaled 40 mA idle
  + 2000 mA at the actuator's force limit. So free motion reads low and pushing on something reads high,
  as on the real servos; the numbers themselves are not measured;
- gripper: real 0..100 maps linearly onto the model's -10..100 deg jaw range (the real gripper spans
  131 deg); grasp outcomes in the twin are indicative only.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from stack.robotd.hardware import ArmReading

from .scene import ARM_JOINTS, Scene

GRIP_I = ARM_JOINTS.index("gripper")
IDLE_MA, MA_AT_LIMIT = 40.0, 2000.0      # current estimate: idle, and at the actuator's force limit


def gripper_real_to_rad(v):
    return math.radians(-10.0 + 1.10 * float(v))


def gripper_rad_to_real(r):
    return (math.degrees(float(r)) + 10.0) / 1.10


class SimWorld:
    """Owns the MuJoCo model and data; steps physics in real time on its own thread."""

    def __init__(self, scene: Scene, start: dict | None = None):
        import mujoco

        self.mj = mujoco
        self.scene = scene
        self.model = scene.model
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.torque = {a: True for a in scene.arms}
        for arm, q in (start or {}).items():
            self._set_arm_qpos(arm, q)
        mujoco.mj_forward(self.model, self.data)
        self._stop = threading.Event()
        self._thread = None
        self.sim_time_at_start = 0.0

    # ---- unit conversions (lerobot degrees <-> model radians), via the rig's joint signs/offsets
    def to_model(self, arm, q6):
        m = self.scene.mounts[arm]
        q = np.asarray(q6, float)
        rad = [m.signs[i] * math.radians(q[i]) + math.radians(m.offsets_deg[i]) for i in range(5)]
        return rad + [gripper_real_to_rad(q[GRIP_I])]

    def from_model(self, arm, rad6):
        m = self.scene.mounts[arm]
        deg = [(math.degrees(rad6[i]) - m.offsets_deg[i]) / m.signs[i] for i in range(5)]
        return np.array(deg + [gripper_rad_to_real(rad6[GRIP_I])])

    def _set_arm_qpos(self, arm, q6):
        info = self.scene.arms[arm]
        rad = self.to_model(arm, q6)
        for i in range(6):
            self.data.qpos[info["qpos"][i]] = rad[i]
            self.data.ctrl[info["act"][i]] = rad[i]

    # ---- physics thread
    def start(self):
        self._thread = threading.Thread(target=self._run, name="twin-physics", daemon=True)
        self._thread.start()

    def _run(self):
        dt = self.model.opt.timestep
        t0 = time.monotonic()
        sim0 = self.data.time
        while not self._stop.is_set():
            target = sim0 + (time.monotonic() - t0)
            n = 0
            # one physics step per lock hold (~0.1 ms), so robotd's 50 Hz reads never wait behind a batch
            while self.data.time < target and n < 20:
                with self.lock:
                    for arm, on in self.torque.items():
                        if not on:                    # limp: the actuator follows the joint instead of driving it
                            info = self.scene.arms[arm]
                            for i in range(6):
                                self.data.ctrl[info["act"][i]] = self.data.qpos[info["qpos"][i]]
                    self.mj.mj_step(self.model, self.data)
                n += 1
            time.sleep(0.0005)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    # ---- queries for previews
    def object_pose(self, name: str) -> list:
        o = self.scene.objects[name]
        with self.lock:
            return [float(v) for v in self.data.qpos[o["qpos"]:o["qpos"] + 3]]

    def tip(self, arm: str) -> list:
        with self.lock:
            return [float(v) for v in self.data.site_xpos[self.scene.arms[arm]["tip_site"]]]


class SimArmIO:
    """One twin arm with the same interface as stack.robotd.hardware.ArmIO."""

    def __init__(self, world: SimWorld, name: str):
        self.world, self.name = world, name
        self.info = world.scene.arms[name]
        self.configured = 0
        fr = world.model.actuator_forcerange
        self.force_limit = np.array([max(abs(fr[a][0]), abs(fr[a][1])) or 3.35 for a in self.info["act"]])

    def connect(self):
        pass

    def firmware_limits(self):
        return {j: {"servo": [0, 4095], "calibration": [0, 4095], "match": True, "note": "twin"} for j in ARM_JOINTS}

    def torque_state(self):
        return [1 if self.world.torque[self.name] else 0] * 6

    def read(self) -> ArmReading:
        t0 = time.perf_counter()
        w, info = self.world, self.info
        with w.lock:
            rad = [float(w.data.qpos[a]) for a in info["qpos"]]
            force = np.array([float(w.data.actuator_force[a]) for a in info["act"]])
            dof = info["dof"]
            internal = np.array([w.model.dof_damping[d] * w.data.qvel[d] + w.model.dof_armature[d] * w.data.qacc[d]
                                 + np.sign(w.data.qvel[d]) * w.model.dof_frictionloss[d] for d in dof])
        q = w.from_model(self.name, rad)
        external = force - internal
        load = np.clip(external / self.force_limit * 100.0, -100, 100)
        cur = IDLE_MA + np.minimum(np.abs(external) / self.force_limit, 1.0) * MA_AT_LIMIT
        return ArmReading(q, load, cur, (time.perf_counter() - t0) * 1000)

    def write_goal(self, q6):
        rad = self.world.to_model(self.name, q6)
        with self.world.lock:
            for i in range(6):
                self.world.data.ctrl[self.info["act"][i]] = rad[i]

    def hold_here(self, q6):
        self.write_goal(q6)
        self.world.torque[self.name] = True

    def torque_off(self):
        self.world.torque[self.name] = False

    def configure_for_motion(self):
        self.configured += 1

    def close(self):
        pass
