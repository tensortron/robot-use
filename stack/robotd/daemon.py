"""robotd core: owns the followers and the cameras, runs the loop, executes checked motions.

Every motion request is planned (planner.py), checked in full before the first tick
(safety.check_paths), gated by the supervisor (approval), and then executed tick by tick with the
contact monitor watching. The loop re-checks every commanded step against the speed caps as a last
line of defence. Nothing here clips a command: a request is either accepted as checked, or rejected
with the rule that failed.

``motion_enabled: false`` (the default; a session turns motion on) keeps robotd read-only:
every motion request is rejected and nothing but torque-off is ever written.
"""

from __future__ import annotations

import json
import math
import re
import logging
import sys
from dataclasses import dataclass, field
import threading
import time
from datetime import datetime
from pathlib import Path as FsPath

import numpy as np

from .config import RobotdConfig
from ..rig.kinematics import ARM_JOINTS, ArmKinematics
from .motion import (ARM_I, GRIPPER_I, PROGRAM_APPROVAL_EXPIRY_S, ConditionMonitor, ContactMonitor, Lease, Motion,
                     ProgramRequest, Supervisor, backoff_target, stop_path, validate_until)
from .planner import PlanError, plan_joint_move, plan_linear, plan_waypoints
from .safety import (ArmLimits, ArmModel, Path, _keypoints, _segments, check_clearance, check_paths, load_arm_limits, rest_pose,
                     segment_distance)

log = logging.getLogger(__name__)

FAULT_AFTER_FAILED_READS = 3
PENDING_EXPIRY_S = 300.0                # a person reads what is about to happen before pressing ARM
MAX_WAIT_S = 30.0


GIL_SWITCH_S = 0.0005
LAUNCH_WINDOW_S = 0.1         # launch-time clearance: the other arm may be up to this far further along its path
                              # (covers the check's own duration; doubled when a long path's check overruns it)


_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")    # program ids are shown on the operator page


class EventLog:
    """Append-only JSONL, one file per day."""

    def __init__(self, directory: FsPath):
        self.dir = FsPath(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> FsPath:
        return self.dir / f"{datetime.now():%Y-%m-%d}.jsonl"

    def write(self, event: str, **data):
        rec = {"ts": round(time.time(), 3), "event": event, **data}
        line = json.dumps(rec, default=_jsonable)
        with self._lock, open(self.path, "a") as fh:
            fh.write(line + "\n")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    return str(o)


def _ranges(vals, step):
    """[-60, -58, -56, -20, -18] -> [[-60, -56], [-20, -18]]"""
    out = []
    for v in vals:
        if out and v - out[-1][1] <= step + 1e-9:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


def _r(a, n=2):
    return [round(float(x), n) for x in a]


from .compute import BACKOFF_SPEED, Compute, Rejected  # noqa: E402  (Rejected: violations are dicts)


@dataclass
class Snapshot:
    """Taken under the lock before planning: where every arm starts, and the remaining path of any arm that
    is moving (t = 0 is the snapshot time) with the motion ids, so registration can tell if it changed."""

    start: dict
    others: dict = field(default_factory=dict)
    running: dict = field(default_factory=dict)


class ArmState:
    def __init__(self, model: ArmModel, contact):
        self.model = model
        self.connected = False
        self.fault: str | None = None
        self.sticky_fault = False          # a fault during motion or with torque on: needs an operator reset
        self.failed_reads = 0
        self.reading = None
        self.read_at = 0.0
        self.torque: list[int] | None = None
        self.firmware: dict | None = None
        self.pose_warnings: list = []      # hull margins this arm sits inside of (as measured), until it moves
        self.launches = 0                  # motions launched on this arm (stale-result guard for the pose check)
        self.q_cmd: np.ndarray | None = None      # the goal the servos are chasing, when robotd set it
        self.v_cmd = np.zeros(len(ARM_JOINTS))
        self.prepared = False
        self.last_command = time.monotonic()
        self.monitor = ContactMonitor(contact)
        self.cond = ConditionMonitor(contact)

    @property
    def torque_on(self) -> bool:
        return bool(self.torque) and any(self.torque)


class RobotD:
    def __init__(self, cfg: RobotdConfig, arm_ios: dict, cam_ios: dict, cam_model=None,
                 limits: dict[str, ArmLimits] | None = None, event_log: EventLog | None = None,
                 compute_workers: int = 0):
        self.cfg = cfg
        self.arm_ios = arm_ios
        self.cam_ios = cam_ios
        self.cam_model = cam_model
        self.events = event_log or EventLog(cfg.log_dir)
        self.arms: dict[str, ArmState] = {}
        lims = {}
        for name, ac in cfg.arms.items():
            lims[name] = limits[name] if limits and name in limits else load_arm_limits(ac.calibration_id)
            self.arms[name] = ArmState(ArmModel(name, ArmKinematics(ac.mount), lims[name]), cfg.contact)
        self.models = {n: st.model for n, st in self.arms.items()}
        self.rest = {n: rest_pose(m, cfg.safety) for n, m in self.models.items()}
        # planning, path checks and reach run here: inline (tests, the twin) or in worker processes (the real
        # daemon), so that they never compete with the 50 Hz loop for this process's interpreter
        self.compute = Compute(cfg, lims, workers=compute_workers)
        bad = self.compute.local.check_rest_poses()
        if bad:                                         # every go_rest / idle rest would be refused
            why = ("the rest pose fails the safety checks (a recalibration? the rest roll?): " + "; ".join(bad))
            if cfg.motion_enabled:
                self.compute.close()
                raise RuntimeError("robotd: " + why)
            log.warning("%s - motion is disabled, so robotd starts; with motion it would refuse to", why)
        try:
            self.rest_warnings = self.compute.local.rest_pose_warnings()  # thin margin: warn only (sprint 2026-10-07)
        except Exception as e:                         # advisory: never a reason not to start
            self.rest_warnings = [f"rest-pose clearance check unavailable: {e}"]
        for w in self.rest_warnings:
            log.warning("rest pose clearance is thin: %s", w)
        self.supervisor = Supervisor(cfg.approval_mode, cfg.armed_timeout_s)
        self.motions: dict[str, Motion] = {}
        self.lease: Lease | None = None
        self._lock = threading.RLock()                                 # state, motions, supervisor
        self._launch_lock = threading.Lock()   # launches one at a time, each checked against the ones before it
        # Lock order. The loop holds _lock and may block on _bus[arm] (_write); _prepare_and_launch holds _bus[arm]
        # and then takes _lock (bus -> main). That cannot deadlock only because _prepare_and_launch touches the
        # bus of arms that need stiffening (torque off or no goal), and such an arm has no running motion for
        # the loop to write to. Keep it that way: never write to, or stiffen, an arm another path may be moving.
        self._bus = {name: threading.Lock() for name in arm_ios}       # one transaction at a time per bus
        self._hist_ver, self._hist_cache = 0, (-1, b"[]")             # the operator page's history, cached per change
        self._bg_inflight: set[tuple[str, str]] = set()                 # (kind, arm) internal motions being planned
        self._bg_lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = None
        self.started_at = 0.0
        self.ticks = 0
        self.overruns = 0
        self._tick_ms: list[float] = []
        self._hz_window: list[float] = []
        self._last_state_log = 0.0
        self._period = 1.0 / cfg.loop_hz

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        # Planning a long move is ~150 ms of pure Python in a request thread. At the default 5 ms GIL switch
        # interval every bus read in the loop thread waits up to 5 ms to get the interpreter back, which
        # stretched a tick to 77 ms on the rig (2026-10-03) and tripped the overrun guard. 0.5 ms keeps the
        # loop's worst tick near its own work; planning gets ~30% slower, which does not matter.
        sys.setswitchinterval(GIL_SWITCH_S)
        if self.compute.pool is not None:    # the real daemon (one per process): keep long-lived startup
            import gc                        # objects (models, config) out of the collector's way
            gc.collect()
            gc.freeze()
        self.started_at = time.time()
        self.events.write("start", backend=self.cfg.backend, motion_enabled=self.cfg.motion_enabled, loop_hz=self.cfg.loop_hz,
                          approval_mode=self.cfg.approval_mode, contact=self.cfg.contact.__dict__,
                          arms={n: {"port": a.port, "calibration_id": a.calibration_id, "mount": a.mount.__dict__}
                                for n, a in self.cfg.arms.items()},
                          rest=self.rest, cameras=list(self.cam_ios), hull_checks=self.compute.local._hulls_on())
        if not self.compute.local._hulls_on():
            log.warning("gripper/camera hull checks are OFF (hull_checks: false, no rig, or not two arms)")
        for name, io in self.arm_ios.items():
            st = self.arms[name]
            try:
                io.connect()
                st.connected = True
                st.firmware = io.firmware_limits()
                bad = [j for j, v in st.firmware.items() if not v["match"]]
                if bad:
                    st.fault = (f"firmware position limits differ from the calibration file on {bad}: "
                                f"re-run lerobot-calibrate for {self.cfg.arms[name].calibration_id}")
                    st.sticky_fault = True
                st.torque = io.torque_state()
            except Exception as e:
                st.fault = f"connect failed: {e}"
            self.events.write("arm_connect", arm=name, connected=st.connected, fault=st.fault,
                              firmware=st.firmware, torque=st.torque)
            if st.connected:
                self._check_pose_soon(name)
        if self.rest_warnings:
            self.events.write("rest_clearance_warning", warnings=self.rest_warnings)
        for io in self.cam_ios.values():
            io.start()
        self._thread = threading.Thread(target=self._loop, name="robotd-loop", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.settle(5.0)
        if self._thread:
            self._thread.join(timeout=3)
        self.compute.close()
        for io in self.arm_ios.values():
            io.close()
        for io in self.cam_ios.values():
            io.stop()
        self.events.write("stop", ticks=self.ticks, overruns=self.overruns)

    # ------------------------------------------------------------------ loop
    def _loop(self):
        period = self._period
        next_t = time.perf_counter()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            for name, io in self.arm_ios.items():
                self._read_arm(name, io)
            with self._lock:
                self._tick_motions()
            work = time.perf_counter() - t0
            with self._lock:
                self.ticks += 1
                self._tick_ms.append(work * 1000)
                self._tick_ms = self._tick_ms[-500:]
                if work > period:
                    self.overruns += 1
                if work > 3 * period:
                    self._abort_all("loop overrun: a tick took %.0f ms (budget %.0f)" % (work * 1000, period * 1000))
                now = time.time()
                self._hz_window.append(now)
                self._hz_window = [t for t in self._hz_window if now - t < 2.0]
                self._changed.notify_all()
            if now - self._last_state_log >= 1.0:
                self._last_state_log = now
                self.events.write("state", **self._arm_snapshot())
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.perf_counter()        # fell behind: do not try to catch up in a burst

    def _read_arm(self, name, io):
        st = self.arms[name]
        if not st.connected:
            return
        bus = self._bus[name]
        if not bus.acquire(blocking=False):
            return              # the bus is busy (an arm being configured / stiffened): keep the last reading
        try:
            try:
                r = io.read()
            finally:
                bus.release()
            with self._lock:
                st.reading, st.read_at, st.failed_reads = r, time.time(), 0
                if st.fault and st.fault.startswith("bus:") and not st.sticky_fault:
                    st.fault = None
                    self.events.write("arm_recovered", arm=name)
        except Exception as e:
            with self._lock:
                st.failed_reads += 1
                if st.failed_reads == FAULT_AFTER_FAILED_READS:
                    moving = any(m.status == "running" and name in m.arms for m in self.motions.values())
                    st.fault = f"bus: {st.failed_reads} reads in a row failed ({e})"
                    if moving or st.torque_on:
                        st.sticky_fault = True
                        st.fault += ("; it happened while the arm was powered: check the arm (12 V, USB), "
                                     "then reset it from the operator page")
                    self.events.write("arm_fault", arm=name, error=str(e), sticky=st.sticky_fault)
                    for m in self._active(name):
                        self._finish(m, "faulted", st.fault)

    # ------------------------------------------------------------------ motion execution (loop thread)
    def _active(self, arm=None):
        return [m for m in self.motions.values() if m.active and (arm is None or arm in m.arms)]

    def _finish(self, m: Motion, status, reason=None, **report):
        m.finish(status, reason, **report)
        if (p := self.supervisor.program_for(m.lease)) is not None:   # the operator page's timeline
            m.info["program"] = p.program_id
        self.events.write("motion_end", **m.as_dict())
        if p is not None:
            self._hist_ver += 1
            p.note({"motion_id": m.id, "kind": m.kind, "arms": m.arms, "status": m.status, "reason": m.reason,
                    "duration_s": round(m.duration, 2), "report": m.report, "ended": time.time()})
        if m.samples:                                 # can be thousands of rows: serialise off the loop thread
            recs = [dict(motion_id=m.id, kind=m.kind, arm=arm, status=status, speed=m.info.get("speed"),
                         joints=[ARM_JOINTS[i] for i in ARM_I],
                         columns="lag_deg per joint, then current_ma per joint", rows=rows)
                    for arm, rows in m.samples.items()]
            m.samples = {}
            threading.Thread(target=lambda: [self.events.write("contact_samples", **r) for r in recs],
                             name="robotd-samples", daemon=True).start()
        self._changed.notify_all()

    def _tick_motions(self):
        now = time.monotonic()
        if self.lease is not None and self.lease.expired:
            self._lease_expired()
        if self.supervisor.expire():
            self.events.write("disarmed", reason="armed timeout")
        for p in self.supervisor.programs.values():
            if p.status in ("pending", "approved") and time.time() - p.requested > PROGRAM_APPROVAL_EXPIRY_S:
                p.status, p.reason = "expired", f"not started within {PROGRAM_APPROVAL_EXPIRY_S:.0f} s"
                self._hist_ver += 1
                self.events.write("program_expired", program=p.program_id)
        for m in list(self.motions.values()):
            if m.status == "pending_approval" and time.time() - m.created > PENDING_EXPIRY_S:
                self._finish(m, "expired", f"not approved within {PENDING_EXPIRY_S:.0f} s")
        for m in [m for m in self.motions.values() if m.status == "running"]:
            t = now - m.started
            done_arms = m.done_arms
            for arm in m.arms:
                if arm in done_arms or m.status != "running":
                    continue
                self._tick_arm(m, arm, t, done_arms)
            if m.status == "running" and len(done_arms) == len(m.arms):
                if m.until and "condition_met" not in m.report:
                    m.report["condition_met"] = None                  # the path ended without meeting it
                self._finish(m, "condition_met" if m.report.get("condition_met") else "done", None)
        self._idle_rest(now)

    def _tick_arm(self, m: Motion, arm: str, t: float, done_arms: list):
        st = self.arms[arm]
        r = st.reading
        if st.fault or r is None:
            self._finish(m, "faulted", st.fault or "no reading")
            return
        path = m.paths[arm]
        # step conditions first: an expected touch must end the step before it counts as an obstruction
        if m.until and st.q_cmd is not None and arm not in (m.report.get("condition_met") or {}):
            hit = st.cond.update(m.until, t, st.q_cmd, r.q, r.current_ma)
            if hit:
                self._condition_met(m, arm, hit, done_arms, t)
                return
        # Level 2: contact, against what the servo has been chasing
        if st.q_cmd is not None and m.kind not in ("stop",):
            hit = st.monitor.update(st.q_cmd, r.q, r.current_ma)
            if self.cfg.contact.mode == "record_only":
                lag = np.abs(np.asarray(st.q_cmd) - np.asarray(r.q))
                m.samples.setdefault(arm, []).append(
                    [round(float(lag[i]), 2) for i in ARM_I] + [round(float(r.current_ma[i])) for i in ARM_I])
            if hit:
                if self.cfg.contact.mode == "record_only":
                    if not m.report.get("contact_recorded"):
                        m.report["contact_recorded"] = {"arm": arm, **hit}
                        self.events.write("contact_recorded", motion=m.id, arm=arm, **hit)
                else:
                    self._blocked(m, arm, hit)
                    return
            if arm in m.grip_effort:
                closing = st.v_cmd[GRIPPER_I] < -1e-6 or (path.q[-1, GRIPPER_I] < r.q[GRIPPER_I] - 1.0)
                if st.monitor.grip(closing, r.current_ma[GRIPPER_I], m.grip_effort[arm]):
                    q = st.q_cmd.copy()
                    q[GRIPPER_I] = r.q[GRIPPER_I]
                    # goal := where the jaws stopped. The goal jumps, the gripper does not move: it stops
                    # squeezing, so this bypasses the step check (as the blocked handler does)
                    if self._write(arm, q, None):
                        m.report["grasped"] = {"arm": arm, "gripper": round(float(r.q[GRIPPER_I]), 1),
                                               "current_ma": round(float(r.current_ma[GRIPPER_I]))}
                        done_arms.append(arm)
                    return
        if t <= path.duration:
            q_cmd = path.at(t)
            if not self._write(arm, q_cmd, st.q_cmd):
                return
        else:
            final = path.q[-1]
            if st.q_cmd is None or np.max(np.abs(st.q_cmd - final)) > 1e-9:
                if not self._write(arm, final, st.q_cmd):
                    return
            err = np.abs(r.q - final)
            if np.max(err[ARM_I]) <= self.cfg.settle_tol_deg:
                done_arms.append(arm)
            elif t > path.duration + self.cfg.settle_timeout_s:
                i = ARM_I[int(np.argmax(err[ARM_I]))]
                m.report.setdefault("not_settled", {})[arm] = {"joint": ARM_JOINTS[i], "error_deg": round(float(err[i]), 1)}
                done_arms.append(arm)

    def _write(self, arm, q_cmd, prev) -> bool:
        """Command one goal; the last-line step check runs first. False if the motion was ended."""
        st = self.arms[arm]
        cfg = self.cfg.safety
        if prev is not None:
            cap = np.array([cfg.vmax[j] for j in ARM_JOINTS]) * self._period * 2.0 + 0.5
            step = np.abs(np.asarray(q_cmd) - prev)
            if np.any(step > cap):
                i = int(np.argmax(step - cap))
                self._abort_arm(arm, f"step check: {arm}.{ARM_JOINTS[i]} commanded to jump {step[i]:.1f} "
                                     f"in one tick (limit {cap[i]:.1f}); motion aborted, arm holding")
                return False
        try:
            with self._bus[arm]:
                self.arm_ios[arm].write_goal(q_cmd)
        except Exception as e:
            st.fault, st.sticky_fault = f"bus: write failed during motion ({e})", True
            self.events.write("arm_fault", arm=arm, error=str(e), sticky=True)
            for m in self._active(arm):
                self._finish(m, "faulted", st.fault)
            return False
        q_cmd = np.asarray(q_cmd, float).copy()
        st.v_cmd = (q_cmd - prev) / self._period if prev is not None else np.zeros_like(q_cmd)
        st.q_cmd = q_cmd
        st.last_command = time.monotonic()
        return True

    def _condition_met(self, m: Motion, arm: str, hit: dict, done_arms: list, t: float = 0.0):
        """End this arm's part of the step. Touch-type conditions hold where the arm is (stop pushing) and the
        arm is done at once; others decelerate at the acceleration caps as the REST OF THIS MOTION, so the
        step ends when the arm is actually still (the next step must not find it busy)."""
        st = self.arms[arm]
        m.report.setdefault("condition_met", {})[arm] = hit
        self.events.write("condition_met", motion=m.id, arm=arm, **hit)
        if hit["type"] in ("contact", "gripper_stalled"):
            hold = st.reading.q.copy()
            if hit["type"] == "contact":
                hold[GRIPPER_I] = st.q_cmd[GRIPPER_I]
            self._write(arm, hold, None)
            done_arms.append(arm)
            return
        p = stop_path(st.q_cmd, st.v_cmd, self.cfg.safety)
        m.paths[arm] = Path(p.t + t, p.q)          # path.at(t) continues from here: decelerate, then hold

    def _lease_expired(self):
        self._end_lease("lease expired", f"lease of {self.lease.holder!r} expired (the program stopped renewing it)",
                        "lease_expired")

    def _end_lease(self, why: str, stop_reason: str | None = None, event: str = "lease_revoked"):
        """Locked: take the lease away. Its running program ends, its motions stop, the session disarms; the
        runner sees its renewals fail and aborts the program."""
        lease, self.lease = self.lease, None
        affected = [m for m in self.motions.values() if m.active and m.lease == lease.token]
        arms = sorted({a for m in affected for a in m.arms})
        for a in arms:                                # first, so its stopped motions are filed under its program
            self._stop_arm(a, reason=stop_reason or why)
        for p in self.supervisor.programs.values():
            if p.lease == lease.token and p.status == "running":
                p.status, p.reason, p.ended, p.progress = "ended", why, time.time(), {}
                self._hist_ver += 1
        self.supervisor.disarm()
        self.events.write(event, holder=lease.holder, stopped=arms, reason=why)

    def _abort_arm(self, arm, reason):
        for m in self._active(arm):
            self._finish(m, "stopped", reason)
        self.events.write("abort", arm=arm, reason=reason)

    def _abort_all(self, reason):
        for arm in self.arms:
            if self._active(arm):
                self._abort_arm(arm, reason)

    def _blocked(self, m: Motion, arm: str, hit: dict):
        st = self.arms[arm]
        r = st.reading
        self._finish(m, "blocked", f"{arm}.{hit['joint']} is blocked: lagging {hit['lag_deg']} deg behind its "
                                   f"command while drawing {hit['current_ma']} mA; stopped, backed off "
                                   f"{self.cfg.contact.backoff_deg:g} deg and holding. Tip at "
                                   f"{_r(st.model.kin.fk(r.q)[:3, 3], 3)} m. The next motion on this arm needs "
                                   f"operator approval.", blocked={"arm": arm, **hit})
        self.supervisor.rearm_required.add(arm)
        # stop pushing: goal = where the arm actually is, then back off slowly along the reverse
        hold = r.q.copy()
        hold[GRIPPER_I] = st.q_cmd[GRIPPER_I] if st.q_cmd is not None else hold[GRIPPER_I]
        v = st.v_cmd.copy()                    # before the hold write zeroes it
        self._write(arm, hold, None)
        back = backoff_target(hold, v if np.any(v) else m.paths[arm].q[-1] - hold, self.cfg.contact.backoff_deg)
        self._submit_background("backoff", arm, ("to", arm, back, BACKOFF_SPEED),
                                rest=True, info={"after": m.id})
        st.monitor.reset()
        # a blocked arm in a two-arm motion stops the other arm too
        for other in m.arms:
            if other != arm:
                self._stop_arm(other)

    def _idle_rest(self, now):
        if not self.cfg.motion_enabled:
            return
        if self.lease is not None and not self.lease.expired:
            return                                # a program owns the arms (e.g. thinking at a checkpoint)
        for arm, st in self.arms.items():
            if (st.fault or not st.torque_on or st.q_cmd is None or self._active(arm)
                    or now - st.last_command < self.cfg.idle_rest_s):
                continue
            if np.max(np.abs(st.q_cmd - self.rest[arm])[ARM_I]) < 2.0:
                st.last_command = now
                continue
            st.last_command = now
            self._submit_background("idle_rest", arm, ("rest", [arm]),
                                    rest=True)

    # ------------------------------------------------------------------ request side (HTTP threads)
    def _start_q(self, arm) -> np.ndarray | None:
        """Where a new path for this arm starts: the servo goal if robotd holds the arm, else measured."""
        st = self.arms[arm]
        if st.torque_on and st.q_cmd is not None:
            return st.q_cmd.copy()
        return None if st.reading is None else st.reading.q.copy()

    def _remaining(self, m: Motion, arm: str, now: float) -> Path:
        """What is left of a running motion for one arm, re-timed so t = 0 is now."""
        p = m.paths[arm]
        el = now - m.started
        keep = p.t > el
        t = np.concatenate([[0.0], p.t[keep] - el])
        q = np.vstack([p.at(el)[None, :], p.q[keep]])
        return Path(t, q)

    def _check_lease(self, lease):
        if self.lease is not None and not self.lease.expired:
            if lease != self.lease.token:
                raise Rejected(f"the robot is leased by {self.lease.holder!r}; only it can command motion now")
        elif lease is not None:
            raise Rejected("that lease has expired or was never granted; acquire a new one")

    def _preflight(self, arms, validate_only=False, internal=False, lease=None) -> Snapshot:
        """Locked, quick: can these arms take a motion now? Snapshot where every arm is (and is going)."""
        if not self.cfg.motion_enabled and not validate_only:
            raise Rejected("motion is disabled on this robot (read-only mode: motion_enabled is false in "
                           "robotd.yaml until the rig checks pass)")
        if not validate_only and not internal:
            self._check_lease(lease)
        for a in arms:
            if a not in self.arms:
                raise Rejected(f"unknown arm {a!r}; arms are {sorted(self.arms)}")
            st = self.arms[a]
            if st.fault:
                raise Rejected(f"{a} arm is faulted: {st.fault}")
            if st.reading is None:
                raise Rejected(f"{a} arm has no position reading yet")
            if not validate_only:
                busy = self._active(a)
                if busy:
                    raise Rejected(f"{a} arm is busy with motion {busy[0].id} ({busy[0].status}); wait for it "
                                   f"(motion_status) or stop it")
        now = time.monotonic()
        snap = Snapshot({a: self._start_q(a) for a in self.arms})
        for a in self.arms:
            if a in arms:
                continue
            run = [m for m in self.motions.values() if m.status == "running" and a in m.arms]
            if run:
                snap.others[a] = self._remaining(run[0], a, now)
                snap.running[a] = run[0].id
        return snap

    def _plan(self, kind, arms, spec, snap: Snapshot, rest=False, grip_effort=None, info=None, until=None,
              lease=None) -> Motion:
        """Unlocked, slow (planning + full path checks, in self.compute): must never hold the loop's lock."""
        try:
            until = validate_until(until)
        except ValueError as e:
            raise Rejected(str(e))
        paths, meta = self.compute.plan(spec, snap.start, snap.others, rest=rest)
        info = dict(info or {})
        if "scale" in meta:
            info["time_scale_applied"] = round(meta["scale"], 3)
        m = Motion(kind, list(arms), paths, rest=rest, grip_effort=grip_effort or {}, info=info,
                   until=until, lease=lease)
        if meta.get("link_clearances"):    # report-only (sprint 2026-10-05 task 4b): link surfaces near the
            self.events.write("link_clearance", motion_id=m.id, kind=kind, arms=list(arms),   # table / obstacles /
                              clearances=meta["link_clearances"])                             # the other arm
        return m

    def _unchanged(self, snap: Snapshot, arms) -> None:
        """Locked: raise if anything the plan was checked against changed while it was being planned."""
        tol = self.cfg.safety.start_tolerance_deg
        for a in self.arms:
            running = [m.id for m in self.motions.values() if m.status == "running" and a in m.arms and a not in arms]
            if (running[0] if running else None) != snap.running.get(a):
                raise Rejected(f"the {a} arm started or stopped moving while the motion was being planned; send it again")
            if a in snap.running:
                continue                          # moving: its path was checked, not its position
            now_q = self._start_q(a)
            if (snap.start[a] is None) != (now_q is None) or (now_q is not None and np.max(np.abs(now_q - snap.start[a])) > tol):
                raise Rejected(f"{a} arm moved while the motion was being planned; send it again")

    def _register(self, m: Motion, snap: Snapshot, internal=False) -> Motion:
        """Locked, quick: nothing changed while planning? then pend for approval or launch."""
        self._preflight(m.arms, internal=internal, lease=m.lease)
        self._unchanged(snap, m.arms)
        self.motions[m.id] = m
        self._trim()
        self.supervisor.touch()
        if self.lease is not None and m.lease == self.lease.token:
            self.lease.renew()
        if not internal and self.supervisor.needs_approval(m):
            m.status = "pending_approval"
            m.start_q = {a: snap.start[a].copy() for a in m.arms}
            self.events.write("motion_pending", **m.as_dict())
        else:
            m.status = "starting"                   # stiffened (if needed), checked and launched by the caller
        return m

    def _submit_flow(self, kind, arms, spec, rest=False, grip_effort=None, info=None, internal=False,
                     until=None, lease=None) -> Motion:
        with self._lock:
            snap = self._preflight(arms, internal=internal, lease=lease)
        m = self._plan(kind, arms, spec, snap, rest, grip_effort, info, until, lease)
        with self._lock:
            self._register(m, snap, internal)
        if m.status == "starting":
            self._prepare_and_launch(m)
        return m

    def _submit_background(self, kind, arm, spec, **kw):
        """Internal motions (back-off, idle rest) decided in the loop but planned off it.

        An idle rest is planned at most once per arm at a time: the loop asks again every idle period, and an arm
        only counts as busy once a motion is registered, so planning that took longer than the period used to pile
        threads up (8 at once in the idle-rest test, starving each other; found on an x86 machine 2026-10-14). A
        back-off is asked for once per contact, so it never piles up, and is never dropped: a second contact while
        the first back-off is still being planned gets its own, planned from where the arm now holds."""
        key = (kind, arm)
        with self._bg_lock:
            if kind == "idle_rest" and key in self._bg_inflight:
                return
            self._bg_inflight.add(key)

        def run():
            try:
                self._submit_flow(kind, [arm], spec, internal=True, **kw)
            except Rejected as e:
                self.events.write(f"{kind}_rejected", arm=arm, reason=str(e))
            except Exception as e:                     # never die silently: the arm keeps holding; say why
                self.events.write(f"{kind}_failed", arm=arm, reason=f"{type(e).__name__}: {e}")
                log.error("%s on %s failed: %s", kind, arm, e)
            finally:
                with self._bg_lock:
                    self._bg_inflight.discard(key)
        try:
            threading.Thread(target=run, name=f"robotd-{kind}", daemon=True).start()
        except BaseException:
            with self._bg_lock:
                self._bg_inflight.discard(key)
            raise

    def _needs_prep(self, arm) -> bool:
        st = self.arms[arm]
        return not st.torque_on or st.q_cmd is None

    def _prepare_and_launch(self, m: Motion):
        """Unlocked: stiffen arms whose torque is off (configure once, then goal := present, then torque on).
        This is ~100 ms of bus traffic per arm; holding the main lock for it stalled the 50 Hz loop into its
        overrun guard on the rig (2026-10-03). Then, locked and quick, launch - unless it was stopped meanwhile."""
        for a in m.arms:
            if not self._needs_prep(a):
                continue
            st = self.arms[a]
            try:
                # bus lock first, then confirm under the main lock that the motion still wants this arm: an
                # operator torque-off (or stop) meanwhile must not be undone by stiffening it here. torque_off
                # takes the same bus lock, so it runs entirely before or entirely after this block.
                with self._bus[a]:
                    with self._lock:
                        if m.status != "starting":
                            return
                        q_hold = st.reading.q.copy()
                    io = self.arm_ios[a]
                    if not st.torque_on and not st.prepared:
                        io.configure_for_motion()
                        st.prepared = True
                    io.hold_here(q_hold)
                    torque = io.torque_state()
                    reading = io.read()
                    with self._lock:
                        st.torque = torque
                        st.reading, st.read_at = reading, time.time()
                        st.q_cmd, st.v_cmd = reading.q.copy(), np.zeros(len(ARM_JOINTS))
            except Exception as e:
                with self._lock:
                    self._finish(m, "faulted", f"{a}: could not enable torque ({e})")
                return
        self._checked_launch(m)

    def _checked_launch(self, m: Motion):
        """Launch, after re-checking the arm-to-arm clearance against what the other arm is doing NOW.

        A plan is checked against the other arm at planning time, but the motion starts later (planning,
        stiffening, approval): the other arm may have moved, started or finished meanwhile, and two motions
        approved together were never checked against each other. So every launch takes the launch lock (one at
        a time: the second of two is checked against the first, already running), snapshots the other arm
        (its remaining path, or where it holds), checks clearance outside the main lock with the other arm
        anywhere up to LAUNCH_WINDOW_S further along its path, and launches only if the other arm is still
        doing what was checked and no more than LAUNCH_WINDOW_S has passed. (Review 2026-10-03.)"""
        with self._launch_lock:
            window = LAUNCH_WINDOW_S
            slow = False
            for _ in range(4):                         # a very long path can take longer to check: widen
                with self._lock:
                    if m.status != "starting":
                        return
                    t0 = time.monotonic()
                    running = self._running_others(m)
                    current = {a: self._start_q(a) for a in self.arms}
                v = []
                if len(self.models) == 2:
                    others = {a: self._remaining(x, a, t0) for a, x in running.items()}
                    try:
                        v = self.compute.clearance(m.paths, current, others, window)
                    except Exception as e:             # fail closed: a motion never launches unchecked
                        with self._lock:
                            if m.status == "starting":
                                self._finish(m, "faulted", f"launch clearance check unavailable: {e}")
                        return
                with self._lock:
                    if m.status != "starting":
                        return
                    if not self.cfg.motion_enabled:    # robotd is stopping: a motion still starting never launches
                        self._finish(m, "rejected", "at launch: motion is off (robotd is stopping)")
                        self.events.write("launch_rejected", motion_id=m.id, kind=m.kind, reason=m.reason)
                        return
                    if v:
                        self._finish(m, "rejected", "at launch, against the other arm: "
                                     + "; ".join(x["message"] for x in v))
                        self.events.write("launch_rejected", motion_id=m.id, kind=m.kind, reason=m.reason)
                        return
                    if {a: x.id for a, x in self._running_others(m).items()} != {a: x.id for a, x in running.items()}:
                        continue                       # the other arm started or stopped: check again
                    tol = self.cfg.safety.start_tolerance_deg
                    if any(a not in running and (current[a] is None or self._start_q(a) is None
                                                 or np.max(np.abs(self._start_q(a) - current[a])) > tol)
                           for a in self.arms if a not in m.arms):
                        continue                       # the idle arm moved (e.g. was pushed): check again
                    if running and time.monotonic() - t0 > window:   # only a MOVING arm can be further along
                        window *= 2
                        slow = True
                        continue
                    self._launch(m)
                    if m.approved_by_operator:
                        self.supervisor.approved(m.arms)     # re-approval done only once it really runs
                    return
            with self._lock:
                if m.status == "starting":
                    self._finish(m, "rejected", (f"at launch: the clearance check against the moving other arm "
                                                 f"kept taking longer than {window / 2:g} s; send it again")
                                 if slow else "at launch: the other arm kept starting, stopping or moving while "
                                              "this motion was being checked; send it again")
                    self.events.write("launch_rejected", motion_id=m.id, kind=m.kind, reason=m.reason)

    def _running_others(self, m: Motion) -> dict:
        return {a: x for a in self.arms if a not in m.arms
                for x in self.motions.values() if x.status == "running" and a in x.arms}

    def _launch(self, m: Motion):
        """Locked, no bus I/O: start executing a motion whose arms are already holding under robotd's goal."""
        self.motions[m.id] = m
        for a in m.arms:
            st = self.arms[a]
            if self._needs_prep(a):
                self._finish(m, "faulted", f"{a}: not stiffened before launch (internal error)")
                return
            st.monitor.reset()
            st.cond.reset()
            st.pose_warnings = []                     # it moves now: the as-it-sits warning no longer applies
            st.launches += 1
        m.status, m.started = "running", time.monotonic()
        self.events.write("motion_start", **m.as_dict())

    def _trim(self, keep=200):
        done = [k for k, m in self.motions.items() if not m.active]
        for k in done[:-keep]:
            del self.motions[k]

    def _submit(self, kind, arms, spec, **kw) -> dict:
        try:
            m = self._submit_flow(kind, arms, spec, **kw)
        except Rejected as e:
            self.events.write("motion_rejected", kind=kind, arms=arms, reason=str(e))
            with self._lock:
                if (p := self.supervisor.program_for(kw.get("lease"))) is not None:
                    self._hist_ver += 1
                    p.note({"motion_id": None, "kind": kind, "arms": arms, "status": "rejected", "reason": str(e),
                            "duration_s": 0.0, "report": {}, "ended": time.time()})
            return {"status": "rejected", "reason": str(e), "violations": list(e.violations)}
        with self._lock:
            return m.as_dict()

    # ---- motion requests (plans are plain-data specs, run by self.compute; see compute.py)
    def move_joints(self, arm: str, targets: dict, speed: float = 0.5, until=None, lease=None) -> dict:
        return self._submit("move_joints", [arm], ("joints", arm, dict(targets), speed),
                            info={"targets": targets, "speed": speed}, until=until, lease=lease)

    def set_gripper(self, arm: str, value: float, speed: float = 0.5, max_effort: float = 0.5, until=None,
                    lease=None) -> dict:
        if not 0.0 < float(max_effort) <= 1.0:
            return {"status": "rejected", "reason": "max_effort must be in (0, 1]"}
        return self._submit("set_gripper", [arm], ("gripper", arm, value, speed),
                            grip_effort={arm: float(max_effort)}, info={"value": value, "speed": speed,
                                                                        "max_effort": max_effort},
                            until=until, lease=lease)

    def move_pose(self, arm: str, xyz, pitch=None, roll=None, linear: bool = False, speed: float = 0.5,
                  until=None, lease=None) -> dict:
        try:
            xyz = [float(v) for v in xyz]
        except (TypeError, ValueError):
            return {"status": "rejected", "reason": "xyz must be three numbers"}
        out = self._submit("move_pose", [arm], ("pose", arm, xyz, pitch, roll, bool(linear), speed),
                           info={"xyz": xyz, "pitch": pitch, "roll": roll, "linear": linear, "speed": speed},
                           until=until, lease=lease)
        if out.get("status") != "rejected":
            with self._lock:
                m = self.motions[out["motion_id"]]
                out["joint_solution"] = dict(zip(ARM_JOINTS, _r(m.paths[arm].q[-1], 2)))
        return out


    def solve_ik(self, arm: str, xyz, pitch=None, roll=None) -> dict:
        with self._lock:
            start = self._start_q(arm) if arm in self.arms else None
        if start is None:
            return {"ok": False, "reason": f"no reading for arm {arm!r}"}
        try:
            q = self.compute.local.ik(arm, start, xyz, pitch, roll)
        except PlanError as e:
            return {"ok": False, "reason": str(e)}
        p = self.models[arm].kin.pose(q)
        return {"ok": True, "joints": dict(zip(ARM_JOINTS, _r(q, 2))), "tip_xyz_m": _r(p["xyz"], 4),
                "pitch_deg": round(p["pitch"], 1)}

    def check_reach(self, arm: str, points: list) -> dict:
        """Can the arm's tip be at each point (and with which pitches)? IK within the usable joint limits, plus the
        static checks of that pose (floor, workspace, obstacles, clearance from the other arm where it is now).
        points: [{"xyz": [x, y, z], "pitch": deg | None, "roll": deg | None}], at most 20. Says nothing about the
        path there - that is what planning a motion checks. Computed in self.compute."""
        if arm not in self.models:
            return {"ok": False, "reason": f"unknown arm {arm!r}"}
        if not isinstance(points, list) or not 1 <= len(points) <= 20:
            return {"ok": False, "reason": "points: a list of 1..20 {xyz, pitch?, roll?}"}
        clean = []
        for pt in points:
            try:
                xyz = [float(v) for v in pt["xyz"]]
                pitch, roll = pt.get("pitch"), pt.get("roll")
                assert len(xyz) == 3 and all(math.isfinite(v) for v in xyz + [float(x) for x in (pitch, roll)
                                                                              if x is not None])
            except Exception:
                return {"ok": False, "reason": f"bad point {pt!r}: needs finite xyz [x, y, z] (pitch, roll optional)"}
            clean.append({"xyz": xyz, "pitch": None if pitch is None else float(pitch),
                          "roll": None if roll is None else float(roll)})
        with self._lock:
            current = {a: self._start_q(a) for a in self.arms}
        if current[arm] is None:
            return {"ok": False, "reason": f"no reading for arm {arm!r}"}
        try:
            return self.compute.reach(arm, clean, current)
        except Rejected as e:
            return {"ok": False, "reason": str(e)}

    def view_pose(self, arm: str, target, distance: float = 0.15, elevations=None, start: int = 0) -> dict:
        """A pose from which the arm's wrist camera looks at `target` (see compute.Planner.view_pose). Read-only."""
        if arm not in self.models:
            return {"ok": False, "reason": f"unknown arm {arm!r}"}
        try:
            target = [float(v) for v in target]
            distance = float(distance)
            elev = [float(e) for e in elevations] if elevations else None
            start = int(start or 0)
            assert 0 <= start <= 100 and len(target) == 3 and all(math.isfinite(v) for v in target + [distance] + (elev or []))
        except Exception:
            return {"ok": False, "reason": "target must be finite [x, y, z]; distance and elevations numbers; "
                                           "start an index 0..100"}
        if not 0.06 <= distance <= 0.5:
            return {"ok": False, "reason": "distance must be 0.06 .. 0.5 m"}
        if elev is not None and not (1 <= len(elev) <= 10 and all(0 <= e <= 85 for e in elev)):
            return {"ok": False, "reason": "elevations: 1..10 values in 0..85 (degrees below horizontal)"}
        with self._lock:
            current = {a: self._start_q(a) for a in self.arms}
        if current[arm] is None:
            return {"ok": False, "reason": f"no reading for arm {arm!r}"}
        try:
            return self.compute.view_pose(arm, target, current, distance=distance, start=start,
                                          **({"elevations": tuple(elev)} if elev else {}))
        except Rejected as e:
            return {"ok": False, "reason": str(e)}

    def run_trajectory(self, arm: str, waypoints: list, time_scale: str = "reject", until=None, lease=None) -> dict:
        arms = ["left", "right"] if arm == "both" else [arm]
        return self._submit("run_trajectory", arms, ("trajectory", arm, list(waypoints), time_scale),
                            info={"waypoints": len(waypoints), "time_scale": time_scale}, until=until, lease=lease)

    def validate_trajectory(self, arm: str, waypoints: list, time_scale: str = "reject") -> dict:
        arms = ["left", "right"] if arm == "both" else [arm]
        meta = {}
        try:
            with self._lock:
                snap = self._preflight(arms, validate_only=True)
            _, meta = self.compute.plan(("trajectory", arm, list(waypoints), time_scale), snap.start, snap.others,
                                        report=True)
            ok, reason, viol = True, None, []
        except Rejected as e:
            ok, reason, viol, meta = False, str(e), list(e.violations), getattr(e, "meta", {}) or {}
        rep = {"ok": ok, "reason": reason, "violations": viol}
        if "report" in meta:
            rep.update(meta["report"])
        if "scale" in meta:
            rep["time_scale_applied"] = round(meta["scale"], 3)
        return rep

    def go_rest(self, arm: str, lease=None) -> dict:
        arms = ["left", "right"] if arm == "both" else [arm]
        return self._submit("go_rest", arms, ("rest", arms), rest=True, lease=lease)

    def motion_status(self, motion_id: str, wait_s: float = 0.0) -> dict:
        deadline = time.monotonic() + min(max(float(wait_s), 0.0), MAX_WAIT_S)
        with self._lock:
            m = self.motions.get(motion_id)
            if m is None:
                return {"status": "unknown", "reason": f"no motion {motion_id!r}"}
            while m.active and time.monotonic() < deadline:
                self._changed.wait(timeout=deadline - time.monotonic())
            d = m.as_dict()
            if m.status == "running":
                d["elapsed_s"] = round(time.monotonic() - m.started, 2)
            return d

    # ---- stop / approval (both sides)
    def _stop_arm(self, arm, reason="stopped on request") -> bool:
        st = self.arms[arm]
        moved = False
        for m in self._active(arm):
            if m.status == "pending_approval":
                self._finish(m, "stopped", f"{reason} (before it was approved)")
                continue
            self._finish(m, "stopped", reason)
            moved = True
        if moved and st.q_cmd is not None and np.any(st.v_cmd):
            p = stop_path(st.q_cmd, st.v_cmd, self.cfg.safety)
            if p.duration > 0:
                sm = Motion("stop", [arm], {arm: p})
                self._launch(sm)
        return moved

    def halt(self, arm: str = "both") -> dict:
        arms = list(self.arms) if arm == "both" else [arm]
        with self._lock:
            if any(a not in self.arms for a in arms):
                return {"ok": False, "reason": f"unknown arm {arm!r}"}
            stopped = [a for a in arms if self._stop_arm(a)]
            self.supervisor.disarm()
            self.events.write("stop_pressed", arms=arms, stopped=stopped)
            return {"ok": True, "stopped": stopped, "note": "decelerating to a hold; the session is disarmed"}

    def approve(self) -> dict:
        """Operator ARM: arm the session and start pending motions that are still valid.

        If the other arm has started moving since a request was planned, the request is re-checked against
        that arm's remaining path (outside the lock, as planning is) before it may start."""
        with self._lock:
            pending = [m for m in self.motions.values() if m.status == "pending_approval"]
            self.supervisor.arm()                     # clears no re-approval by itself (see below)
            snaps = {m.id: self._preflight(m.arms, validate_only=True) for m in pending}
        verdicts, together = {}, {}
        # motions started together on different arms: check them as one two-arm motion (hulls included) - each
        # was planned against the other arm standing still (review 2026-10-05); the later one gives way
        ordered = sorted(pending, key=lambda x: x.created)
        for i, m1 in enumerate(ordered):
            for m2 in ordered[i + 1:]:
                if set(m1.arms) & set(m2.arms) or m2.id in verdicts:
                    continue
                start = dict(snaps[m2.id].start)
                start.update({a: m1.start_q[a] for a in m1.arms})
                start.update({a: m2.start_q[a] for a in m2.arms})
                others = {a: p for a, p in snaps[m2.id].others.items() if a not in m1.arms and a not in m2.arms}
                try:                             # rest allowances only if both are rests (one doesn't relax the other)
                    v = self.compute.check({**m1.paths, **m2.paths}, start, others, rest=m1.rest and m2.rest)
                    if v:
                        together[m2.id] = m1.id
                        verdicts[m2.id] = "; ".join(x["message"] for x in v)
                except Rejected as e:
                    together[m2.id] = m1.id
                    verdicts[m2.id] = str(e)
        for m in pending:
            snap = snaps[m.id]
            if m.id in verdicts:
                continue
            if snap.others:
                start = dict(snap.start)
                start.update({a: m.start_q[a] for a in m.arms})
                try:
                    v = self.compute.check(m.paths, start, snap.others, rest=m.rest)
                    verdicts[m.id] = "; ".join(x["message"] for x in v) if v else None
                except Rejected as e:
                    verdicts[m.id] = str(e)
        with self._lock:
            started, refused = [], []
            for m in pending:
                if m.status != "pending_approval":
                    continue
                moved = [a for a in m.arms if self._start_q(a) is None
                         or np.max(np.abs(self._start_q(a) - m.start_q[a])) > self.cfg.safety.start_tolerance_deg]
                if moved:
                    self._finish(m, "rejected", f"{moved} moved while waiting for approval; send the motion again")
                    continue
                if verdicts.get(m.id):
                    why = (f"approved together with the other arm's motion {together[m.id]}, they would meet"
                           if m.id in together else "the other arm is moving")
                    self._finish(m, "rejected", f"rejected at approval ({why}): {verdicts[m.id]}")
                    refused.append(m)
                    continue
                try:
                    self._unchanged(Snapshot(snaps[m.id].start, snaps[m.id].others, snaps[m.id].running), m.arms)
                except Rejected as e:
                    self._finish(m, "rejected", f"at approval: {e}")
                    continue
                m.status = "starting"
                m.approved_by_operator = True          # clears the arms' re-approval once it actually runs
                started.append(m)
            self.events.write("armed", started=[m.id for m in started], mode=self.supervisor.mode)
        for m in started:                            # stiffen and launch outside the lock, one at a time
            self._prepare_and_launch(m)
        return {"ok": True, "armed": self.supervisor.armed,
                "started": [m.id for m in started if m.status in ("running", "done")],
                "status": {m.id: m.status for m in started + refused},
                "reasons": {m.id: m.reason for m in started + refused if m.reason}}

    # ---- programs (approved as a whole by the operator)
    def request_program(self, program_id: str, holder: str, title: str, code: str, preview: dict | None = None,
                        step_mode: bool = False) -> dict:
        with self._lock:
            if not self.cfg.motion_enabled:
                return {"ok": False, "reason": "motion is disabled on this robot (read-only mode)"}
            if program_id in self.supervisor.programs:
                return {"ok": False, "reason": f"program {program_id!r} already requested"}
            if not _SAFE_ID.fullmatch(str(program_id)):            # shown on the operator page: plain ids only
                return {"ok": False, "reason": "program_id: letters, digits, '-' and '_' only (up to 64)"}
            self._hist_ver += 1
            p = ProgramRequest(str(program_id), str(holder), str(title), str(code), preview or {}, bool(step_mode))
            if self.supervisor.mode == "none":
                p.status, p.reason = "approved", "approval_mode none"
            self.supervisor.programs[p.program_id] = p
            self.events.write("program_requested", **p.as_dict())
            self._changed.notify_all()
            return {"ok": True, **p.as_dict(code=False)}

    def program(self, program_id: str, wait_s: float = 0.0) -> dict:
        deadline = time.monotonic() + min(max(float(wait_s), 0.0), MAX_WAIT_S)
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            if p is None:
                return {"ok": False, "reason": f"no program {program_id!r}"}
            while p.status == "pending" and time.monotonic() < deadline:
                self._changed.wait(timeout=deadline - time.monotonic())
            return {"ok": True, **p.as_dict(code=False)}

    def decide_program(self, program_id: str, approve: bool, step_mode: bool | None = None) -> dict:
        """Operator: approve (optionally switching step mode) or deny a pending program."""
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            if p is None or p.status != "pending":
                return {"ok": False, "reason": f"no pending program {program_id!r}"}
            if step_mode is not None:
                p.step_mode = bool(step_mode)
            p.status = "approved" if approve else "denied"
            p.reason = None if approve else "denied by the operator"
            self._hist_ver += 1
            if approve:
                self.supervisor.arm()
            self.events.write("program_" + p.status, program=p.program_id, step_mode=p.step_mode)
            self._changed.notify_all()
            return {"ok": True, **p.as_dict(code=False)}

    def start_program(self, program_id: str, lease: str) -> dict:
        """Bind an approved program to the lease its runner holds; its motions then need no approval."""
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            if p is None or p.status != "approved":
                return {"ok": False, "reason": f"program {program_id!r} is not approved"}
            if self.lease is None or self.lease.expired or self.lease.token != lease:
                return {"ok": False, "reason": "start needs the current lease"}
            p.status, p.lease, p.started = "running", lease, time.time()
            self._hist_ver += 1
            self.events.write("program_started", program=p.program_id, holder=self.lease.holder)
            return {"ok": True, **p.as_dict(code=False)}

    def end_program(self, program_id: str, lease: str, outcome: str = "") -> dict:
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            if p is None or p.lease != lease:
                return {"ok": False, "reason": f"not the running program {program_id!r}"}
            p.status, p.reason, p.ended = "ended", str(outcome)[:500], time.time()
            p.progress = {}
            self._hist_ver += 1
            self.events.write("program_ended", program=p.program_id, outcome=p.reason)
            self._prune_programs()
            return {"ok": True}

    PROGRAMS_KEPT = 100                       # finished programs kept in memory (the page shows HISTORY_SHOWN)
    HISTORY_SHOWN = 30

    def _prune_programs(self):
        """Locked: forget the oldest finished programs beyond PROGRAMS_KEPT (pending and running ones stay)."""
        done = [k for k, p in self.supervisor.programs.items() if p.status in ("ended", "denied", "expired")]
        for k in done[:-self.PROGRAMS_KEPT]:
            del self.supervisor.programs[k]
            self._hist_ver += 1

    def program_progress(self, program_id: str, lease: str, step: int | None = None, kind: str | None = None,
                         checkpoint: str | None = None) -> dict:
        """The runner: where its running program is (a step's number and kind, or a checkpoint it waits at). Shown
        on the operator page; changes nothing else."""
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            if p is None or p.status != "running" or p.lease != lease or self.lease is None or self.lease.token != lease:
                return {"ok": False, "reason": f"not the running program {program_id!r}"}
            p.progress = {"step": None if step is None else int(step), "kind": None if kind is None else str(kind)[:40],
                          "checkpoint": None if checkpoint is None else str(checkpoint)[:80], "at": time.time()}
            self._hist_ver += 1
            return {"ok": True}

    HISTORY_WITH_MOTIONS = 3                  # the newest programs come with their motions; older ones on request

    def history_json(self) -> bytes:
        """Operator: the most recent programs, newest first (read-only, bounded), as JSON. Serialised once per change
        and outside the lock: each open page polls it every second, and robotd's loop must not wait on it."""
        with self._lock:
            ver = self._hist_ver
            if self._hist_cache[0] == ver:
                return self._hist_cache[1]
            ps = sorted(self.supervisor.programs.values(), key=lambda p: p.requested, reverse=True)
            snap = [p.as_dict(code=False, history=True) for p in ps[:self.HISTORY_SHOWN]]
        for d in snap[self.HISTORY_WITH_MOTIONS:]:
            d["motions"] = None                   # fetch /history/<program_id> to see them
        body = json.dumps(snap, default=str).encode()
        with self._lock:
            if self._hist_ver == ver:
                self._hist_cache = (ver, body)
        return body

    def history(self) -> list[dict]:
        return json.loads(self.history_json())

    def program_history(self, program_id: str) -> dict:
        """Operator: one program's record with its motions (the timeline, when an older program is opened)."""
        with self._lock:
            p = self.supervisor.programs.get(program_id)
            return {"ok": False, "reason": f"no program {program_id!r}"} if p is None \
                else {"ok": True, **p.as_dict(code=False, history=True)}

    def pending_programs(self) -> list[dict]:
        with self._lock:
            return [p.as_dict() for p in self.supervisor.programs.values() if p.status == "pending"]

    # ---- leases
    def acquire_lease(self, holder: str, ttl_s: float = 3.0) -> dict:
        if not 0.5 <= float(ttl_s) <= 30:
            return {"ok": False, "reason": "ttl_s must be in [0.5, 30]"}
        with self._lock:
            if self.lease is not None and not self.lease.expired:
                return {"ok": False, "reason": f"the robot is already leased by {self.lease.holder!r}"}
            self.lease = Lease(str(holder), float(ttl_s))
            self.events.write("lease_acquired", holder=self.lease.holder, ttl_s=ttl_s)
            return {"ok": True, "token": self.lease.token, "ttl_s": ttl_s}

    def renew_lease(self, token: str, ttl_s: float | None = None) -> dict:
        with self._lock:
            if self.lease is None or self.lease.token != token or self.lease.expired:
                return {"ok": False, "reason": "lease expired or unknown"}
            self.lease.renew(None if ttl_s is None else min(max(float(ttl_s), 0.5), 30.0))
            return {"ok": True, **self.lease.as_dict()}

    def release_lease(self, token: str) -> dict:
        with self._lock:
            if self.lease is None or self.lease.token != token:
                return {"ok": False, "reason": "not the current lease"}
            holder, self.lease = self.lease.holder, None
            self.events.write("lease_released", holder=holder)
            return {"ok": True}

    def reset_arm(self, arm: str) -> dict:
        with self._lock:
            st = self.arms.get(arm)
            if st is None:
                return {"ok": False, "reason": f"unknown arm {arm!r}"}
            if st.failed_reads or st.reading is None or time.time() - st.read_at > 0.5:
                return {"ok": False, "reason": f"{arm} is still not reading; check 12 V and USB first"}
            if self._active(arm):
                return {"ok": False, "reason": f"{arm} has an active motion; stop it first"}
            q = st.reading.q.copy()
        try:                                          # bus I/O outside the main lock (it would stall the loop)
            with self._bus[arm]:
                torque = self.arm_ios[arm].torque_state()
                if torque is not None and any(torque):
                    self.arm_ios[arm].hold_here(q)
        except Exception as e:
            return {"ok": False, "reason": f"{arm}: {e}"}
        with self._lock:
            st.torque = torque
            st.q_cmd = q if st.torque_on else None
            st.v_cmd = np.zeros(len(ARM_JOINTS))
            st.fault, st.sticky_fault = None, False
            self.supervisor.rearm_required.add(arm)
            self.events.write("arm_reset", arm=arm, torque=st.torque)
            return {"ok": True, "torque": st.torque}

    def set_approval_mode(self, mode: str) -> dict:
        if mode not in ("first_motion", "every_motion"):
            return {"ok": False, "reason": "the operator page can choose first_motion or every_motion; "
                                           "'none' is set in robotd.yaml only"}
        with self._lock:
            self.supervisor.mode = mode
            self.supervisor.disarm()
            self.events.write("approval_mode", mode=mode)
            return {"ok": True, "mode": mode}

    # ------------------------------------------------------------------ views
    def _arm_snapshot(self) -> dict:
        out = {}
        for name, st in self.arms.items():
            a = {"connected": st.connected, "fault": st.fault, "torque": st.torque, "pose_warnings": list(st.pose_warnings)}
            r = st.reading
            if r is not None:
                pose = st.model.kin.pose(r.q)
                a.update({"joints": dict(zip(ARM_JOINTS, _r(r.q))), "load_pct": dict(zip(ARM_JOINTS, _r(r.load_pct, 1))),
                          "current_ma": dict(zip(ARM_JOINTS, _r(r.current_ma, 0))),
                          "tip_xyz_m": _r(pose["xyz"], 4), "pitch_deg": round(pose["pitch"], 1),
                          "roll_deg": round(pose["roll"], 1), "read_ms": round(r.read_ms, 2),
                          "age_s": round(time.time() - st.read_at, 3)})
            if st.q_cmd is not None and st.torque_on:          # where robotd is holding / driving the arm; the
                cp = st.model.kin.pose(st.q_cmd)                # measured pose differs by the servos' steady error
                a["command"] = {"joints": dict(zip(ARM_JOINTS, _r(st.q_cmd))), "tip_xyz_m": _r(cp["xyz"], 4),
                                "pitch_deg": round(cp["pitch"], 1), "roll_deg": round(cp["roll"], 1)}
            busy = [m for m in self.motions.values() if m.active and name in m.arms]
            a["motion"] = busy[0].as_dict() if busy else None
            out[name] = a
        return out

    def state(self) -> dict:
        with self._lock:
            arms = self._arm_snapshot()
            tick = np.array(self._tick_ms) if self._tick_ms else np.zeros(1)
            hz = (len(self._hz_window) - 1) / (self._hz_window[-1] - self._hz_window[0]) if len(self._hz_window) > 1 else 0.0
            backend = self.cfg.backend
            loop = {"target_hz": self.cfg.loop_hz, "measured_hz": round(hz, 1), "ticks": self.ticks,
                    "overruns": self.overruns, "work_ms_median": round(float(np.median(tick)), 2),
                    "work_ms_p99": round(float(np.percentile(tick, 99)), 2)}
            sup = {**self.supervisor.as_dict(), "mode": self.supervisor.mode if self.cfg.motion_enabled else "read_only",
                   "pending": [m.as_dict() | {"expires_in_s": round(PENDING_EXPIRY_S - (time.time() - m.created))}
                               for m in self.motions.values() if m.status == "pending_approval"],
                   "recent": [m.as_dict() | {"ended_s_ago": round(time.time() - (m.ended or time.time()))}
                              for m in list(self.motions.values())[-8:] if not m.active and m.kind != "validate"][::-1],
                   "lease": self.lease.as_dict() if self.lease is not None and not self.lease.expired else None}
        cams = {n: {"fps": round(io.fps_measured(), 1), "error": io.error} for n, io in self.cam_ios.items()}
        warnings = [f"rest pose: {w}" for w in self.rest_warnings]
        if self.cam_model is not None and self.cfg.overlay is not None and self.cfg.overlay.camera in cams:
            src = dict(self.cam_model.source)
            cams[self.cfg.overlay.camera]["model"] = src
            est = [k for k, v in src.items() if v == "estimated"]
            if est:
                warnings.append(f"{self.cfg.overlay.camera} camera: its {' and '.join(est)} are estimates from "
                                f"robot.yaml, not calibrated: positions read from it are approximate")
        return {"motion_enabled": self.cfg.motion_enabled, "backend": backend, "uptime_s": round(time.time() - self.started_at, 1),
                "arms": arms, "loop": loop, "cameras": cams, "supervisor": sup, "warnings": warnings}

    def config_summary(self) -> dict:
        return {"motion_enabled": self.cfg.motion_enabled, "approval_mode": self.cfg.approval_mode,
                "mcp_localhost_only": self.cfg.mcp_localhost_only, "contact": dict(self.cfg.contact.__dict__)}

    def limits(self) -> dict:
        cfg = self.cfg.safety
        out = {"units": {"joints": "deg", "gripper": "0..100 (0 closed)", "xyz": "m, rig frame"},
               "rig_frame": "origin on the work surface midway between the arm bases; +x forward, +y robot's left, +z up",
               "speed_caps": cfg.vmax, "acceleration_caps": cfg.amax,
               "workspace_box": {"min": cfg.workspace_min, "max": cfg.workspace_max},
               "floor": {"work_surface_z": cfg.work_surface_z, "board_top_z": cfg.board_top_z,
                         "board_outline": {"front_x": cfg.board_front_x, "half_width_y": cfg.board_half_width_y,
                                           "note": "board top is floor for x <= front_x and |y| <= half_width_y; "
                                                   "the work surface (z = 0) elsewhere"},
                         "tip_clearance_m": cfg.tip_clearance_m},
               "arm_clearance_m": cfg.arm_clearance_m,
               "obstacles": [{"name": o["name"], "min": list(o["min"]), "max": list(o["max"])} for o in cfg.obstacles],
               "obstacle_clearance_m": cfg.obstacle_clearance_m, "arms": {}}
        for name, st in self.arms.items():
            lo, hi = st.model.limits.usable(cfg)
            out["arms"][name] = {
                "hard": {j: _r(st.model.limits.hard[j], 1) for j in ARM_JOINTS},
                "usable": {j: [round(float(lo[i]), 1), round(float(hi[i]), 1)] for i, j in enumerate(ARM_JOINTS)},
                "rest_pose": dict(zip(ARM_JOINTS, _r(self.rest[name], 1)))}
        return out

    def frame(self, camera: str, overlay: bool = False):
        """(BGR frame copy or None, capture time)."""
        io = self.cam_ios.get(camera)
        if io is None:
            raise KeyError(camera)
        f, ts = io.latest()
        if f is None:
            return None, ts
        f = f.copy()
        if overlay and self.cam_model is not None:
            from ..rig.camera_model import draw_overlay

            arms = {}
            with self._lock:
                for name, st in self.arms.items():
                    if st.reading is not None:
                        pose = st.model.kin.pose(st.reading.q)
                        arms[name] = {"keypoints": st.model.kin.keypoints(st.reading.q), "tip": pose["xyz"],
                                      "pitch": pose["pitch"]}
            draw_overlay(f, self.cam_model, arms)
            sc = self.cfg.safety
            from ..rig.camera_model import draw_static
            board = (None if sc.board_front_x is None or sc.board_half_width_y is None
                     else (sc.board_front_x, sc.board_half_width_y, sc.board_top_z))
            draw_static(f, self.cam_model, sc.obstacles, board)
        return f, ts

    # ------------------------------------------------------------------ operator actions
    def _check_pose_soon(self, arm: str, wait_s: float = 4.0) -> None:
        """In the background, once the arm has a fresh reading: the hull margins it sits inside of as it is
        (e.g. a torque-off arm settled near its own shoulder). Shown on the operator page and logged, until it
        moves; its first motion must move away from it (a gripper-only motion passes the hull rules if nothing
        gets closer; sprint 2026-10-07, task 3)."""
        def run():
            t0 = time.time()
            with self._lock:
                gen = self.arms[arm].launches
            prev, still_since = None, None               # wait until the arm has stopped moving (a torque-off arm
            while time.time() - t0 < wait_s and not self._stop.is_set():      # sags for a moment), up to wait_s
                st = self.arms[arm]
                if st.reading is not None and st.read_at > t0:
                    q = st.reading.q[:5].copy()
                    if prev is not None and np.max(np.abs(q - prev)) < 0.2:
                        still_since = still_since or time.time()
                        if time.time() - still_since >= 0.5:
                            break
                    else:
                        still_since = None
                    prev = q
                time.sleep(0.05)
            with self._lock:
                current = {a: (None if s.reading is None else s.reading.q.copy()) for a, s in self.arms.items()}
            try:
                w = self.compute.local.pose_warnings(arm, current)
            except Exception as e:                     # advisory only: never in the way of the arm
                w = [f"pose check unavailable: {e}"]
            with self._lock:
                if self.arms[arm].launches != gen:     # it moved meanwhile: the result is stale
                    return
                self.arms[arm].pose_warnings = w
            if w:
                self.events.write("pose_warning", arm=arm, warnings=w)
                log.warning("%s arm sits inside a margin: %s", arm, "; ".join(w))
        t = threading.Thread(target=run, name=f"pose-check-{arm}", daemon=True)
        self._pose_threads = [x for x in getattr(self, "_pose_threads", []) if x.is_alive()] + [t]
        t.start()

    def settle(self, timeout: float = 10.0) -> None:
        """Wait for the background pose checks (tests; stop())."""
        for t in list(getattr(self, "_pose_threads", [])):
            t.join(timeout)

    def torque_off(self, arm: str) -> dict:
        if arm not in self.arm_ios:
            raise KeyError(arm)
        st = self.arms[arm]
        with self._lock:
            for m in self._active(arm):
                self._finish(m, "stopped", "operator turned torque off")
            # someone may be handling the arm: nothing re-stiffens it without the operator approving a motion
            self.supervisor.disarm()
            self.supervisor.rearm_required.add(arm)
            if self.lease is not None and not self.lease.expired:
                self._end_lease("the operator turned torque off on the " + arm + " arm")
        try:
            with self._bus[arm]:
                self.arm_ios[arm].torque_off()
                st.torque = self.arm_ios[arm].torque_state()
            ok = st.torque is not None and not any(st.torque)
            with self._lock:
                st.q_cmd, st.v_cmd = None, np.zeros(len(ARM_JOINTS))
            self.events.write("torque_off", arm=arm, ok=ok, torque=st.torque)
            self._check_pose_soon(arm)
            return {"ok": ok, "torque": st.torque}
        except Exception as e:
            self.events.write("torque_off", arm=arm, ok=False, error=str(e))
            return {"ok": False, "error": str(e)}

    def stop_motion(self) -> dict:
        """Operator STOP (both arms)."""
        return self.halt("both")
