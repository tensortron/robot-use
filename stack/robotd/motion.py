"""Motion bookkeeping for robotd step 3: motion records, the contact monitor, the supervisor.

Design doc: "Level 2: contact monitor" and "Level 3: supervisor". The daemon owns the loop and the
bus; this module holds the decisions so they can be unit-tested without a loop.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..rig.kinematics import ARM_JOINTS
from .safety import Path, SafetyConfig

GRIPPER_I = ARM_JOINTS.index("gripper")
ARM_I = [i for i, j in enumerate(ARM_JOINTS) if j != "gripper"]

ACTIVE = ("pending_approval", "starting", "running")       # starting: arms being stiffened, outside the lock
_ids = itertools.count(1)


@dataclass
class ContactConfig:
    mode: str = "act"                         # "act" | "record_only" (phase 2 threshold recording)
    lag_deg: float = 8.0                      # placeholder until phase 2 measures free-motion lag
    current_ma: float = 500.0                 # placeholder, per joint (arm joints)
    ticks: int = 5                            # >= 100 ms at 50 Hz
    backoff_deg: float = 3.0
    grip_full_effort_ma: float = 800.0        # gripper current at max_effort = 1 (placeholder)

    def __post_init__(self):
        if self.mode not in ("act", "record_only"):
            raise ValueError("contact.mode must be 'act' or 'record_only'")


@dataclass
class Motion:
    kind: str
    arms: list[str]
    paths: dict[str, Path]
    rest: bool = False
    grip_effort: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"m{next(_ids)}")
    status: str = "pending_approval"
    reason: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None                # monotonic
    ended: float | None = None
    report: dict = field(default_factory=dict)
    start_q: dict = field(default_factory=dict, repr=False)     # where each arm was when planned (internal)
    done_arms: list = field(default_factory=list, repr=False)
    lease: str | None = field(default=None, repr=False)         # the lease it was commanded under
    until: list = field(default_factory=list)                   # step conditions (validate_until)
    # contact.mode record_only: per arm, every tick's [lag_deg x5, current_ma x5] for the arm joints, written
    # out as one `contact_samples` event when the motion ends (a bench check sets the thresholds from these)
    samples: dict = field(default_factory=dict, repr=False)
    approved_by_operator: bool = field(default=False, repr=False)

    @property
    def duration(self) -> float:
        return max((p.duration for p in self.paths.values()), default=0.0)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def finish(self, status: str, reason: str | None = None, **report):
        self.status, self.reason, self.ended = status, reason, time.time()
        self.report.update(report)

    def as_dict(self) -> dict:
        d = {"motion_id": self.id, "kind": self.kind, "arms": self.arms, "status": self.status,
             "reason": self.reason, "duration_s": round(self.duration, 3), **self.info}
        if self.until:
            d["until"] = self.until
        if self.report:
            d["report"] = self.report
        return d


class ContactMonitor:
    """Per arm: an arm joint is blocked when it lags its command AND draws high current, for N ticks."""

    def __init__(self, cfg: ContactConfig):
        self.cfg = cfg
        self.count = np.zeros(len(ARM_JOINTS), int)
        self.grip_count = 0
        self.peak_lag = np.zeros(len(ARM_JOINTS))
        self.peak_ma = np.zeros(len(ARM_JOINTS))

    def reset(self):
        self.count[:] = 0
        self.grip_count = 0

    def update(self, q_cmd, q_meas, current_ma) -> dict | None:
        """Returns a blocked report for the first joint over threshold, else None (gripper excluded)."""
        lag = np.abs(np.asarray(q_cmd) - np.asarray(q_meas))
        cur = np.asarray(current_ma)
        self.peak_lag = np.maximum(self.peak_lag, lag)
        self.peak_ma = np.maximum(self.peak_ma, cur)
        over = (lag > self.cfg.lag_deg) & (cur > self.cfg.current_ma)
        over[GRIPPER_I] = False
        self.count = np.where(over, self.count + 1, 0)
        hit = [i for i in ARM_I if self.count[i] >= self.cfg.ticks]
        if not hit:
            return None
        i = max(hit, key=lambda k: lag[k])
        return {"joint": ARM_JOINTS[i], "lag_deg": round(float(lag[i]), 1), "current_ma": round(float(cur[i])),
                "commanded_deg": round(float(q_cmd[i]), 1), "measured_deg": round(float(q_meas[i]), 1)}

    def grip(self, closing: bool, current_ma: float, effort: float) -> bool:
        """True once the gripper has pushed at ``effort`` x full current for N ticks while closing."""
        if not closing:
            self.grip_count = 0
            return False
        self.grip_count = self.grip_count + 1 if current_ma > effort * self.cfg.grip_full_effort_ma else 0
        return self.grip_count >= self.cfg.ticks


def backoff_target(q_meas, velocity, deg: float) -> np.ndarray:
    """``deg`` along the reverse of the commanded arm-joint velocity (largest joint moves ``deg``)."""
    q = np.asarray(q_meas, float).copy()
    v = np.asarray(velocity, float).copy()
    v[GRIPPER_I] = 0.0
    n = np.max(np.abs(v))
    if n < 1e-6:
        return q
    q -= deg * v / n
    return q


def stop_path(q_cmd, velocity, cfg: SafetyConfig) -> Path:
    """Decelerate every joint from ``velocity`` to rest at its acceleration cap."""
    q0 = np.asarray(q_cmd, float)
    v0 = np.asarray(velocity, float)
    a = np.array([cfg.amax[j] for j in ARM_JOINTS])
    T = float(np.max(np.abs(v0) / a)) if np.any(v0) else 0.0
    if T <= 0:
        return Path(np.array([0.0]), q0[None, :].copy())
    n = max(int(math.ceil(T / cfg.tick_s)), 1)
    t = np.linspace(0.0, T, n + 1)
    ti = np.minimum(t[:, None], (np.abs(v0) / a)[None, :])            # each joint stops at its own time
    q = q0 + v0 * ti - 0.5 * np.sign(v0) * a * ti ** 2
    return Path(t, q)


# --------------------------------------------------------------------------- leases
@dataclass
class Lease:
    """Exclusive command access for one client (a running program). Expires unless renewed."""

    holder: str
    ttl_s: float
    token: str = field(default_factory=lambda: __import__("secrets").token_hex(12))
    expires: float = 0.0

    def __post_init__(self):
        self.renew()

    def renew(self, ttl_s: float | None = None):
        if ttl_s is not None:
            self.ttl_s = ttl_s
        self.expires = time.monotonic() + self.ttl_s

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.expires

    def as_dict(self) -> dict:
        return {"holder": self.holder, "ttl_s": self.ttl_s, "expires_in_s": round(self.expires - time.monotonic(), 2)}


# --------------------------------------------------------------------------- step conditions
# Evaluated by robotd every tick while the step runs ("move until ..."). Meeting one ends the step
# successfully with status condition_met; unexpected contact is still `blocked`.
CONTACT_LEVELS = {"light": (0.5, 0.6, 3), "firm": (0.8, 0.85, 4)}   # fractions of the blocked thresholds, ticks
CONDITION_TYPES = ("contact", "gripper_stalled", "joint_crosses", "time")


def validate_until(until) -> list:
    """Normalise a step's `until` (a condition dict or a list of them, any-of). Raises ValueError."""
    if until is None:
        return []
    items = until if isinstance(until, list) else [until]
    out = []
    for c in items:
        if not isinstance(c, dict) or c.get("type") not in CONDITION_TYPES:
            raise ValueError(f"until: each condition needs a type in {CONDITION_TYPES}, got {c!r}")
        t = c["type"]
        if t == "contact":
            level = c.get("level", "light")
            if level not in CONTACT_LEVELS:
                raise ValueError(f"until contact: level must be one of {list(CONTACT_LEVELS)}")
            joints = c.get("joints")
            if joints is not None and (not joints or any(j not in ARM_JOINTS or j == "gripper" for j in joints)):
                raise ValueError("until contact: joints must be arm joint names")
            out.append({"type": t, "level": level, "joints": joints})
        elif t == "gripper_stalled":
            eff = float(c.get("effort", 0.3))
            if not 0 < eff <= 1:
                raise ValueError("until gripper_stalled: effort must be in (0, 1]")
            out.append({"type": t, "effort": eff})
        elif t == "joint_crosses":
            if c.get("joint") not in ARM_JOINTS or c.get("direction") not in (1, -1):
                raise ValueError("until joint_crosses: needs joint (a joint name), value and direction (+1 or -1)")
            out.append({"type": t, "joint": c["joint"], "value": float(c["value"]), "direction": int(c["direction"])})
        else:
            s = float(c.get("s", -1))
            if s <= 0:
                raise ValueError("until time: s must be > 0")
            out.append({"type": t, "s": s})
    return out


class ConditionMonitor:
    """Per arm, per step: has any `until` condition been met? Contact thresholds sit below the blocked
    thresholds (ContactConfig), so an expected touch ends the step before it would count as an obstruction."""

    def __init__(self, cfg: ContactConfig):
        self.cfg = cfg
        self.count: dict[int, int] = {}

    def reset(self):
        self.count = {}

    def update(self, until: list, t: float, q_cmd, q_meas, current_ma) -> dict | None:
        lag = np.abs(np.asarray(q_cmd) - np.asarray(q_meas))
        cur = np.asarray(current_ma)
        for k, c in enumerate(until):
            typ = c["type"]
            if typ == "time":
                if t >= c["s"]:
                    return {"type": "time", "s": c["s"]}
                continue
            if typ == "joint_crosses":
                i = ARM_JOINTS.index(c["joint"])
                if (q_meas[i] - c["value"]) * c["direction"] >= 0:
                    return {"type": typ, "joint": c["joint"], "value": round(float(q_meas[i]), 2)}
                continue
            if typ == "contact":
                fl, fc, ticks = CONTACT_LEVELS[c["level"]]
                idx = [ARM_JOINTS.index(j) for j in (c["joints"] or [ARM_JOINTS[i] for i in ARM_I])]
                hit = [i for i in idx if lag[i] > fl * self.cfg.lag_deg and cur[i] > fc * self.cfg.current_ma]
            else:                                                     # gripper_stalled
                ticks = self.cfg.ticks
                hit = [GRIPPER_I] if cur[GRIPPER_I] > c["effort"] * self.cfg.grip_full_effort_ma else []
            self.count[k] = self.count.get(k, 0) + 1 if hit else 0
            if self.count[k] >= ticks:
                i = max(hit, key=lambda j: lag[j])
                return {"type": typ, **({"level": c["level"]} if typ == "contact" else {}), "joint": ARM_JOINTS[i],
                        "lag_deg": round(float(lag[i]), 1), "current_ma": round(float(cur[i]))}
        return None


PROGRAM_APPROVAL_EXPIRY_S = 600.0


@dataclass
class ProgramRequest:
    """A program asking the human to approve it as a whole (docs/ARCHITECTURE.md)."""

    program_id: str
    holder: str
    title: str
    code: str
    preview: dict
    step_mode: bool
    status: str = "pending"            # pending -> approved -> running -> ended | denied | expired
    requested: float = field(default_factory=time.time)
    lease: str | None = None
    reason: str | None = None
    started: float | None = None       # unix times, for the operator page's timeline
    ended: float | None = None
    progress: dict = field(default_factory=dict)      # from its runner: {"step", "kind", "checkpoint"}
    motions: list = field(default_factory=list)       # compact records of its motions and refused steps

    MAX_MOTIONS = 100                  # the newest kept: a long program's latest motions and its end matter most

    def note(self, rec: dict):
        self.motions.append(rec)
        if len(self.motions) > self.MAX_MOTIONS:
            del self.motions[0]

    def as_dict(self, code=True, history=False) -> dict:
        d = {"program_id": self.program_id, "holder": self.holder, "title": self.title, "status": self.status,
             "step_mode": self.step_mode, "preview": self.preview, "reason": self.reason}
        if not history:                # the history is cached: an age in it would be stale
            d["age_s"] = round(time.time() - self.requested, 1)
        if code:
            d["code"] = self.code
        if history:
            d.update(requested=self.requested, started=self.started, ended=self.ended, progress=self.progress,
                     n_motions=len(self.motions), motions=list(self.motions))
        return d


class Supervisor:
    """Approval state. Modes: first_motion (default), every_motion, none.

    Programs are approved as a whole: once a program is approved and running under a lease, its motions
    need no further approval, except in step mode (every step waits for ARM) or on an arm that was
    blocked (re-approval after an obstruction always applies)."""

    MODES = ("first_motion", "every_motion", "none")

    def __init__(self, mode: str = "first_motion", armed_timeout_s: float = 1800.0):
        if mode not in self.MODES:
            raise ValueError(f"approval_mode must be one of {self.MODES}")
        self.mode = mode
        self.armed_timeout_s = armed_timeout_s
        self.armed = False
        self.last_activity = time.monotonic()
        self.rearm_required: set[str] = set()      # arms that were blocked: always need approval next
        self.programs: dict[str, ProgramRequest] = {}

    def touch(self):
        self.last_activity = time.monotonic()

    def expire(self):
        if self.armed and time.monotonic() - self.last_activity > self.armed_timeout_s:
            self.armed = False
            return True
        return False

    def program_for(self, lease: str | None) -> ProgramRequest | None:
        if lease is None:
            return None
        return next((p for p in self.programs.values() if p.lease == lease and p.status == "running"), None)

    def needs_approval(self, motion: Motion) -> bool:
        if motion.rest and motion.kind == "idle_rest":
            return False
        if any(a in self.rearm_required for a in motion.arms):
            return True
        prog = self.program_for(motion.lease)
        if prog is not None:
            return prog.step_mode
        if self.mode == "none":
            return False
        if self.mode == "every_motion":
            return True
        return not self.armed

    def arm(self):
        """Operator ARM. Does not clear any arm's re-approval: only approving a motion on that arm does."""
        self.armed = self.mode != "every_motion"
        self.touch()                                 # or a long-idle session would expire on the next tick

    def approved(self, arms):
        """The operator approved a motion on these arms (shown and started): their re-approval is done."""
        for a in arms:
            self.rearm_required.discard(a)
        self.touch()

    def disarm(self):
        self.armed = False

    def as_dict(self) -> dict:
        return {"mode": self.mode, "armed": self.armed, "rearm_required": sorted(self.rearm_required),
                "armed_timeout_s": self.armed_timeout_s,
                "programs": [p.as_dict(code=False) for p in self.programs.values()
                             if p.status in ("pending", "approved", "running")]}
