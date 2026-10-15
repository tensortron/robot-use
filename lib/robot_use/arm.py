"""Arms, results, waiting, two-arm moves and checkpoints for robot programs (docs/ARCHITECTURE.md).

Every method that moves the robot is one **step**: it blocks until the motion ends and returns a
StepResult, or raises StepFailed (rejected by the safety layer, blocked by an obstruction, stopped, faulted)
with the reason. Positions are rig-frame metres; `pitch` is the gripper's approach angle (0 forward and
level, -90 straight down); `roll` the wrist_roll joint (degrees); `speed` a fraction (0..1] of the caps.

    right = arm("right")
    right.open()
    right.move_to(block.above(0.08), pitch=-90)
    right.move_to(block.at(0.02), pitch=-90, until=contact())     # stops on touching
    g = right.close(effort=0.4)                                    # g.grasped: is it holding something?
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import runtime
from .estimate import Estimate

GRIPPER_CLOSED = 0.0          # the lowest gripper value the safety layer accepts (its usable range is 0..98)


@dataclass
class StepResult:
    status: str                      # done | condition_met
    raw: dict

    @property
    def condition_met(self):
        """The condition that ended the step ({arm: {...}}), or None."""
        return (self.raw.get("report") or {}).get("condition_met")

    @property
    def grasped(self):
        """For a close: {"arm", "gripper", "current_ma"} if the jaws stopped on something, else None."""
        return (self.raw.get("report") or {}).get("grasped")

    @property
    def not_settled(self):
        return (self.raw.get("report") or {}).get("not_settled")

    @property
    def motion_id(self) -> str:
        return self.raw.get("motion_id")


WORK_ROLL = -90.0
"""The recommended wrist roll for approaches and table grasps (sprint 2026-10-05, D9): the wrist camera on top of
the gripper, the jaws opening sideways. At roll +90 the camera hangs underneath and low poses are refused.
move_to(roll=None) still keeps the current roll: pass roll=WORK_ROLL to use it. (look_at picks its own roll.)"""


class NoViewpoint(Exception):
    """look_at found no reachable, clear viewpoint (``.tried`` says why per elevation). Not a failed step."""

    def __init__(self, message: str, tried=None):
        super().__init__(message)
        self.tried = tried or []


def _result(raw: dict) -> StepResult:
    return StepResult(raw.get("status"), raw)


def _xyz(target) -> list:
    if isinstance(target, Estimate):
        raise TypeError("pass a point, e.g. block.above(0.08) or block.centre, not the Estimate itself")
    t = [float(v) for v in target]
    if len(t) != 3:
        raise ValueError("a target needs [x, y, z]")
    return t


class Arm:
    def __init__(self, name: str):
        if name not in ("left", "right"):
            raise ValueError("arm is 'left' or 'right'")
        self.name = name

    # ------------------------------------------------------------------ observation
    def state(self) -> dict:
        """This arm's state: joints, gripper, tip_xyz_m (measured), pitch_deg, roll_deg, load_pct, current_ma, fault;
        while torque is on also `command` ({tip_xyz_m, pitch_deg, roll_deg}: where robotd holds the arm - the
        measured tip can sag 1-2 cm below it when reaching out)."""
        return runtime.state()["arms"][self.name]

    @property
    def tip(self) -> list:
        return list(self.state()["tip_xyz_m"])

    @property
    def joints(self) -> dict:
        return dict(self.state()["joints"])

    @property
    def gripper(self) -> float:
        return float(self.state()["joints"]["gripper"])

    # ------------------------------------------------------------------ steps
    def move_to(self, target, pitch: float | None = None, roll: float | None = None, linear: bool = False,
                speed: float = 0.5, until=None) -> StepResult:
        """Move the gripper tip to target [x, y, z]. pitch/roll None lets the robot choose (keeps roll).
        linear=True follows a straight line (slower; rejected where the line leaves the reachable space)."""
        args = {"arm": self.name, "xyz": _xyz(target), "pitch": pitch, "roll": roll, "linear": bool(linear),
                "speed": float(speed)}
        if until is not None:
            args["until"] = until
        return _result(runtime.step("move_pose", **args))

    def move_by(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, linear: bool = True, speed: float = 0.3,
                until=None) -> StepResult:
        """Move the tip relative to where it is now, keeping its pitch and roll (a straight line by default)."""
        s = self.state()
        c = s.get("command") or s                  # where robotd holds the arm, not where it sags to (the
        t = c["tip_xyz_m"]                         # measured pose is off by the servos' steady error)
        return self.move_to([t[0] + dx, t[1] + dy, t[2] + dz], pitch=c["pitch_deg"], roll=c["roll_deg"],
                            linear=linear, speed=speed, until=until)

    def move_joints(self, speed: float = 0.5, until=None, **targets) -> StepResult:
        """Joint targets in degrees (gripper 0..100), e.g. move_joints(shoulder_pan=20, wrist_flex=60)."""
        args = {"arm": self.name, "targets": {k: float(v) for k, v in targets.items()}, "speed": float(speed)}
        if until is not None:
            args["until"] = until
        return _result(runtime.step("move_joints", **args))

    def open(self, width: float = 95.0, speed: float = 0.8) -> StepResult:
        """Open the gripper to `width` (0 closed .. 100 fully open, about 8 cm between the tips). The safety
        layer keeps 0..98: values outside are rejected, not clipped."""
        return _result(runtime.step("set_gripper", arm=self.name, value=float(width), speed=float(speed), max_effort=1.0))

    def close(self, effort: float = 0.5, speed: float = 0.6) -> StepResult:
        """Close until the jaws meet resistance at `effort` (0..1) of the full grip, then hold.
        result.grasped is set if they stopped on something; None if they closed on air."""
        return _result(runtime.step("set_gripper", arm=self.name, value=GRIPPER_CLOSED, speed=float(speed),
                                    max_effort=float(effort)))

    def rest(self) -> StepResult:
        """Slowly to the folded rest pose."""
        return _result(runtime.step("go_rest", arm=self.name))

    # ------------------------------------------------------------------ looking and reach (no motion)
    def reach(self, points) -> dict:
        """Can this arm's tip be at each point, and at which pitches? `points`: [x, y, z] lists or
        {"xyz", "pitch"?, "roll"?} dicts (up to 20). Read-only: no motion, not a step."""
        pts = [p if isinstance(p, dict) else {"xyz": _xyz(p)} for p in points]
        return runtime.reach(self.name, pts)

    def look_at(self, target, distance: float = 0.15, speed: float = 0.4, elevations=None) -> dict:
        """Move this arm so that ITS WRIST CAMERA looks straight at `target` (an [x, y, z] point, or an Estimate:
        its centre) from `distance` metres, from the arm's own side. Use the idle arm as a second viewpoint, then
        `checkpoint(..., views=["wrist_<this arm>"])`. The camera is tilted ~25 deg from the fingers, so this aims
        the camera, not the jaws. Tries viewing elevations (degrees below horizontal) with the camera on top of
        the gripper (roll -90), then beside it (roll 0), then underneath it (roll +90), keeping clear of the table, obstacles and the
        other arm; if the move to one is refused on the way there, the next is tried. Returns the viewpoint
        reached ({"camera", "xyz", "pitch", "roll", "elevation", "distance" (both as achieved), "axis_miss_mm",
        "step": StepResult}); raises NoViewpoint if none works (try another distance) - not a failed step."""
        t = target.centre if isinstance(target, Estimate) else _xyz(target)
        tried, start = [], 0
        while True:
            v = runtime.view_pose(self.name, t, distance, elevations, start)
            if not v or not v.get("ok"):
                tried += (v or {}).get("tried") or []
                raise NoViewpoint(f"{self.name}: no viewpoint on {t} at {distance} m: {(v or {}).get('reason')}",
                                  tried)
            tried += v.get("skipped") or []
            try:
                step = self.move_to(v["xyz"], pitch=v["pitch"], roll=v["roll"], speed=speed)
            except runtime.StepFailed as e:
                if e.result.get("status") != "rejected":
                    raise                               # it moved and then failed: a real failed step
                tried.append({**{k: v["requested"][k] for k in ("roll", "elevation")},
                              "reason": f"the move there: {e.result.get('reason')}"})
                start = int(v["candidate"]) + 1
                if start >= len(v.get("candidates") or []):
                    raise NoViewpoint(f"{self.name}: no viewpoint on {t} at {distance} m: every move there was "
                                      f"refused", tried)
                continue
            return {**{k: v[k] for k in ("camera", "xyz", "pitch", "roll", "elevation", "distance", "camera_xyz",
                                         "axis_miss_mm")}, "step": step}

    # ------------------------------------------------------------------ two-arm targets
    def target(self, xyz=None, pitch=None, roll=None, gripper=None, **joints) -> dict:
        """A target for parallel(): a tip pose, or joint values, plus an optional gripper value."""
        if xyz is not None and joints:
            raise ValueError("give xyz or joints, not both")
        t = {"pose": {"xyz": _xyz(xyz), "pitch": pitch, "roll": roll}} if xyz is not None else {"joints": joints}
        if gripper is not None:
            t["gripper"] = float(gripper)
        return t


_ARMS = {}


def arm(name: str) -> Arm:
    if name not in _ARMS:
        _ARMS[name] = Arm(name)
    return _ARMS[name]


def parallel(left: dict, right: dict, duration_s: float = 2.0) -> StepResult:
    """Move both arms at once, as ONE motion (so their clearance is checked along both paths).
    left/right come from arm(...).target(...). duration_s is a minimum; it stretches to respect the caps."""
    wp = [{"t": float(duration_s), "left": left, "right": right}]
    return _result(runtime.step("trajectory", arm="both", waypoints=wp, time_scale="auto"))


class WaitTimeout(Exception):
    pass


def wait_until(predicate, timeout_s: float, poll_s: float = 0.1, what: str = "condition") -> float:
    """Poll predicate() until it is true; returns the seconds waited. A timeout is required: raises WaitTimeout.
    Nothing moves while waiting (the arms hold); use until= conditions to wait *during* a motion."""
    if not timeout_s or timeout_s <= 0:
        raise ValueError("wait_until needs a positive timeout_s")
    t0 = time.monotonic()
    while True:
        if predicate():
            return time.monotonic() - t0
        if time.monotonic() - t0 > timeout_s:
            raise WaitTimeout(f"{what} not met within {timeout_s:.1f} s")
        time.sleep(max(0.02, float(poll_s)))


def checkpoint(name: str, *estimates: Estimate, views=("ego",), need: str = "") -> str:
    """Pause here with the arms holding; the agent looks at the evidence (the named camera views) and decides.

    Returns "continue" or "refine". On "refine" the given Estimates have already been moved (each within its
    own tolerance; a bigger correction needs a revised program). An abort ends the program."""
    by_name = {e.name: e for e in estimates}
    if len(by_name) != len(estimates):
        raise ValueError("estimates at one checkpoint need distinct names")
    r = runtime.checkpoint(name, estimates={n: e.as_checkpoint() for n, e in by_name.items()}, views=list(views),
                           need=need)
    for n, xy in (r.get("estimates") or {}).items():
        e = by_name.get(n)
        if e is not None and [round(v, 9) for v in xy] != [round(v, 9) for v in e.xy]:
            e.refine(xy=xy, source=f"checkpoint {name!r}")
    return r.get("decision", "continue")
