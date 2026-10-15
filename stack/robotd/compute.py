"""robotd's heavy computation - planning, path checks, reach - away from its 50 Hz loop.

The loop and the HTTP request threads share one Python interpreter. Planning a straight line, checking a long
path or scanning reach is tens to hundreds of milliseconds of Python and numpy, and while a request thread
runs it the loop waits for the interpreter between its bus reads: a stress test on 2026-10-03 stretched ticks
from 9 ms to 98 ms, and on the rig motions were aborted by the overrun guard (61-62 ms ticks). Switching
interval tweaks only shorten the waits; the cure is to run this work in other processes.

`Planner` is pure: built from the config and the arm limits, it turns plain-data requests into checked paths.
`Compute` runs a Planner either inline (tests, the twin) or in a pool of worker processes (the real daemon).
Requests and results are plain data (specs, numpy arrays, Path, Violation), never closures.

    spec                                                  meaning
    ("joints", arm, {joint: deg}, speed)                  joint move
    ("gripper", arm, value, speed)                        gripper move
    ("pose", arm, xyz, pitch, roll, linear, speed)        tip pose (joint-interpolated, or a straight line)
    ("trajectory", arm, waypoints, time_scale)            timed waypoints (arm may be "both")
    ("rest", [arms])                                      slowly to the rest pose
    ("to", arm, q6, speed)                                joint move to a given vector (back-off)
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..rig.kinematics import ARM_JOINTS, ArmKinematics
from .motion import ARM_I, GRIPPER_I
from .planner import PlanError, plan_joint_move, plan_linear, plan_waypoints
from .safety import (Path, _keypoints, _segments, check_clearance, check_paths, rest_pose, segment_distance)

REST_SPEED = 0.3                   # fraction of the speed caps, most of the way to rest
REST_FINAL_DEG = 15.0              # the last stretch into rest ...
REST_FINAL_DEG_S = 10.0            # ... at this joint speed (deg/s)
# The wrist roll never turns while the arm is folded (sprint 2026-10-05, task 6b): folded at roll +90 the wrist
# camera is ~34 mm inside the shoulder; rolling in place from 0 to -90 grazes it. A rest motion that must change the
# roll first goes to this pre-rest pose (rest + lift 10, elbow -40), where the whole -90..+90 roll keeps the camera
# and jaws 7+ mm from the arm's own shoulder, base and upper arm (model), turns the wrist there, then folds.
PRE_REST_DELTA = np.array([0.0, 10.0, -40.0, 0.0, 0.0, 0.0])
REST_ROLL_TOL_DEG = 5.0
# "Folded": lift within FOLD_LIFT_DEG of the rest's (folded back) and elbow within FOLD_ELBOW_DEG of the rest's
# (bent in). The pre-rest pose (elbow -40) is outside; 20 deg out still had the camera in the shoulder at +90.
# Every path is refused if it turns the wrist more than REST_ROLL_TOL_DEG while folded; joint and pose moves that
# start folded and change the roll go out to the pre-rest pose first (review 2026-10-06: moves out of rest to a
# roll-0 or +90 pose brought the camera to 0 mm from the shoulder).
# the geometric rules (not joint ranges, speeds or path form): what "inside a margin" means for the operator warnings
GEOMETRY_RULES = frozenset({"floor", "obstacle", "workspace", "arm_clearance",
                            "camera_clearance", "jaws_clearance", "gripper_clearance"})
FOLD_LIFT_DEG = 25.0
FOLD_ELBOW_DEG = 35.0
BACKOFF_SPEED = 0.2                # after a blocked motion


class Rejected(Exception):
    """A request the safety layer refuses: the reason, and the violations (dicts) if any."""

    def __init__(self, message: str, violations=None):
        super().__init__(message)
        self.violations = list(violations or [])       # dicts (Violation.as_dict()), so they cross processes
        self.meta = {}

    def __reduce__(self):                               # survive pickling back from a worker process
        return (_rejected, (str(self), self.violations, self.meta))


def _rejected(message, violations, meta):
    e = Rejected(message, violations)
    e.meta = meta
    return e


def _r(a, n=2):
    return [round(float(x), n) for x in a]


def _ranges(vals, step):
    """[-60, -58, -56, -20, -18] -> [[-60, -56], [-20, -18]]"""
    out = []
    for v in vals:
        if out and v - out[-1][1] <= step + 1e-9:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


_HULLS: dict = {}                       # one HullChecker per (rig, margins) per process, shared under its lock
_HULLS_BUILD = None                     # serialises the first build (threading.Lock, made on first use)


def _hull_checker(cfg):
    import json
    import threading

    from .hulls import HullChecker

    global _HULLS_BUILD
    if _HULLS_BUILD is None:
        _HULLS_BUILD = threading.Lock()
    key = (str(getattr(cfg.rig, "path", "")), json.dumps(cfg.safety.hull_margins_m, sort_keys=True))
    with _HULLS_BUILD:
        if key not in _HULLS:
            hc = HullChecker(cfg.rig, cfg.safety)
            _HULLS[key] = (hc, hc.lock)                   # the checker's own re-entrant lock
    return _HULLS[key]


class Planner:
    def __init__(self, cfg, limits: dict):
        from .safety import ArmModel

        self.cfg = cfg
        self.models = {n: ArmModel(n, ArmKinematics(ac.mount), limits[n]) for n, ac in cfg.arms.items()}
        self.rest = {n: rest_pose(m, cfg.safety) for n, m in self.models.items()}
        for n in self.models:                           # the routing needs the pre-rest pose out of the fold
            if self._folded(n, self.rest[n] + PRE_REST_DELTA):
                raise RuntimeError(f"{n}: the pre-rest pose (rest + PRE_REST_DELTA) is inside the fold zone; "
                                   "FOLD_ELBOW_DEG / FOLD_LIFT_DEG and PRE_REST_DELTA disagree")

    REST_SETTLE_DEG = 2.5                # how far a torque-off or loaded arm settles off its rest (lift, elbow)

    def rest_pose_warnings(self) -> list:
        """Clearance of each rest pose with the arm settled a little off it (lift and elbow +/- REST_SETTLE_DEG) and
        the gripper fully open or closed: failures are warnings, not refusals (sprint 2026-10-07, G2) - the
        commanded rest must pass (check_rest_poses); this only says the margin is thin."""
        if not self._hulls_on():
            return []
        out = set()
        cur = dict(self.rest)
        for arm, r in self.rest.items():
            for dl in (-self.REST_SETTLE_DEG, self.REST_SETTLE_DEG):
                for de in (-self.REST_SETTLE_DEG, self.REST_SETTLE_DEG):
                    for g in (self.cfg.safety.gripper_usable[0], self.cfg.safety.gripper_usable[1]):
                        q = r.copy()
                        q[1] += dl
                        q[2] += de
                        q[5] = g
                        for x in self.full_check({arm: Path(np.array([0.0]), q[None, :])}, {**cur, arm: q}, query=True):
                            if x.rule in GEOMETRY_RULES or x.rule == "hull_check_incomplete":
                                out.add(f"{arm} rest settled {dl:+.1f} lift / {de:+.1f} elbow: " + x.message.split(" at t=")[0])
        return sorted(out)

    def pose_warnings(self, arm, current) -> list:
        """Hull margins the arm is inside of as it sits now (sprint 2026-10-07, task 3): the first motion must
        move away, or be gripper-only."""
        if not self._hulls_on() or current.get(arm) is None:
            return []
        q = np.asarray(current[arm], float)
        cur = {a: v for a, v in current.items() if v is not None}
        return [x.message.split(" at t=")[0] for x in self.full_check({arm: Path(np.array([0.0]), q[None, :])}, cur,
                                                                       query=True)
                if x.rule in GEOMETRY_RULES or x.rule == "hull_check_incomplete"]

    def check_rest_poses(self) -> list:
        """The rest poses must pass every check (own arm included): a rest that ends inside a margin can never be
        reached (re-review 2026-10-06: the left arm's has 1.1 mm to spare, and depends on its calibration)."""
        cur = dict(self.rest)
        return [x.message for x in self.full_check({a: Path(np.array([0.0]), q[None, :]) for a, q in cur.items()},
                                                    cur, rest=True)]

    def full_check(self, paths, start, others=None, rest=False, query=False) -> list:
        """check_paths (segments, capsules, tip, ranges, rates) + the convex-hull checks of the gripper end."""
        return self.full_check_report(paths, start, others, rest, query=query)[0]

    def _hulls_on(self) -> bool:
        return self.cfg.safety.hull_checks and self.cfg.rig is not None and len(self.models) == 2

    def full_check_report(self, paths, start, others=None, rest=False, query=False):
        """(violations, report-only link clearances). A motion needs a reading of every arm (the launch check
        does too - review 2026-10-05); a read-only query (reach, view_pose: query=True) leaves an arm without one
        out of both checks (the hull scene masks it) rather than skipping the hull checks."""
        known = {a: q for a, q in start.items() if q is not None}
        unread = [a for a in self.models if a not in known and a not in paths and a not in (others or {})]
        if unread and not query:
            raise Rejected(f"no reading for the {', '.join(unread)} arm: the arm-to-arm check needs it; "
                           "nothing was moved")
        models = {a: m for a, m in self.models.items() if a in known or a in paths}
        v = list(check_paths(models, self.cfg.safety, paths, known, rest=rest, others=others, hulls=self._hulls_on()))
        v += self._folded_roll_violations(paths)
        report = {}
        if self._hulls_on():
            hc, lock = _hull_checker(self.cfg)
            with lock:
                hv, report = hc.check(paths, known, others, rest=rest)
            v += hv
        return v, report

    # ------------------------------------------------------------------ building blocks
    def full_target(self, arm, start_q, targets: dict) -> np.ndarray:
        unknown = set(targets) - set(ARM_JOINTS)
        if unknown:
            raise PlanError(f"unknown joints {sorted(unknown)}; joints are {list(ARM_JOINTS)}")
        q = np.asarray(start_q, float).copy()
        for j, v in targets.items():
            q[ARM_JOINTS.index(j)] = float(v)
        return q

    def ik(self, arm, start_q, xyz, pitch=None, roll=None):
        m = self.models[arm]
        lo, hi = m.limits.usable(self.cfg.safety)
        r = m.kin.solve_ik(xyz, pitch, roll, seed_deg=start_q[:5], lower_deg=lo[:5], upper_deg=hi[:5])
        if not r.ok:
            raise PlanError(f"{arm}: {r.message}")
        q = np.asarray(start_q, float).copy()
        q[:5] = r.q
        return q

    def _folded(self, arm, q) -> bool:
        r = self.rest[arm]
        return bool(q[1] < r[1] + FOLD_LIFT_DEG and q[2] > r[2] - FOLD_ELBOW_DEG)

    def _joint_move(self, arm, start_q, target_q, speed) -> Path:
        """A joint move. One that would turn the wrist while the arm is folded (anywhere along it) goes via the
        pre-rest pose instead: out (same roll), turn the wrist there, then on to the target."""
        start_q, target_q = np.asarray(start_q, float), np.asarray(target_q, float)
        sf = self.cfg.safety
        direct = plan_joint_move(start_q, target_q, sf, speed)
        if abs(target_q[4] - start_q[4]) <= REST_ROLL_TOL_DEG or not self._folded_roll_violations({arm: direct}):
            return direct                               # (any point of it folded while turning: route instead)
        out = self.rest[arm] + PRE_REST_DELTA
        out[4], out[5] = start_q[4], start_q[5]
        turned = out.copy()
        turned[4] = target_q[4]
        parts = [plan_joint_move(start_q, out, sf, speed), plan_joint_move(out, turned, sf, speed),
                 plan_joint_move(turned, target_q, sf, speed)]
        t, q = parts[0].t, parts[0].q
        for p in parts[1:]:
            t, q = np.concatenate([t, t[-1] + p.t[1:]]), np.vstack([q, p.q[1:]])
        return Path(t, q)

    def _folded_roll_violations(self, paths) -> list:
        """The wrist must not turn while the arm is folded (its camera and jaws would sweep the shoulder)."""
        from .safety import Violation
        out = []
        for arm, p in paths.items():
            if arm not in self.rest:
                continue
            ts = np.r_[np.arange(0.0, p.duration, self.cfg.safety.tick_s), p.duration]
            q = np.array([p.at(t) for t in ts])                   # sampled: a path may have only its two ends
            fold = np.array([self._folded(arm, x) for x in q])
            worst, t_first, turned, first = 0.0, None, 0.0, None
            for k in range(1, len(q)):                  # the whole turn of each folded stretch
                if fold[k] and fold[k - 1]:
                    turned += abs(q[k, 4] - q[k - 1, 4])
                    if turned > REST_ROLL_TOL_DEG and first is None:
                        first = ts[k]
                else:
                    turned, first = 0.0, None
                if turned > worst:
                    worst, t_first = turned, first
            if worst > REST_ROLL_TOL_DEG:
                out.append(Violation(
                    "folded_roll", arm, "wrist_roll", float(t_first), float(worst),
                    f"at most {REST_ROLL_TOL_DEG:.0f} deg while folded",
                    f"{arm} wrist would turn {worst:.1f} deg while the arm is folded near its rest (from t={t_first:.2f}s; "
                    f"at most {REST_ROLL_TOL_DEG:.0f}): its camera would sweep the shoulder. move_to, move_joints and "
                    f"go_rest route around this by themselves; for a straight-line move or a trajectory, first move "
                    f"the arm out of the fold (or into it) without turning the wrist"))
        return out

    def _rest_route(self, arm, start) -> Path:
        """rest_path; if that is refused (e.g. the way out to the pre-rest pose dips into the fold at a roll that
        sweeps the shoulder) and the arm is not folded, turn the wrist where it is first, then fold. The first
        route that passes the full check (the other arm holding) wins; else the default, to be refused with its
        reasons (re-review 2026-10-06: an arm must not be stranded away from rest)."""
        start_q = np.asarray(start[arm], float)
        rest_q = self.rest[arm]
        default = self.rest_path(start_q, rest_q)
        if abs(start_q[4] - rest_q[4]) <= REST_ROLL_TOL_DEG or self._folded(arm, start_q) or not self._hulls_on():
            return default                              # (turning in place is safe only if the own-arm hulls see it)
        cur = {a: q for a, q in start.items() if q is not None}
        if not self.full_check({arm: default}, cur, rest=True):
            return default
        turned = start_q.copy()
        turned[4] = rest_q[4]
        p1 = plan_joint_move(start_q, turned, self.cfg.safety, REST_SPEED)
        p2 = self._fold_path(turned, rest_q)
        alt = Path(np.concatenate([p1.t, p1.t[-1] + p2.t[1:]]), np.vstack([p1.q, p2.q[1:]]))
        return default if self.full_check({arm: alt}, cur, rest=True) else alt

    def rest_path(self, start_q, rest_q) -> Path:
        """To the rest pose: if the wrist roll must change, via the pre-rest pose (out, roll, fold - see
        PRE_REST_DELTA), else straight; the last REST_FINAL_DEG slowly."""
        start_q, rest_q = np.asarray(start_q, float), np.asarray(rest_q, float)
        if abs(start_q[4] - rest_q[4]) <= REST_ROLL_TOL_DEG:
            return self._fold_path(start_q, rest_q)
        sf = self.cfg.safety
        pre = rest_q + PRE_REST_DELTA
        pre[4] = start_q[4]
        turned = pre.copy()
        turned[4] = rest_q[4]
        parts = [plan_joint_move(start_q, pre, sf, REST_SPEED), plan_joint_move(pre, turned, sf, REST_SPEED),
                 self._fold_path(turned, rest_q)]
        t, q = parts[0].t, parts[0].q
        for p in parts[1:]:
            t, q = np.concatenate([t, t[-1] + p.t[1:]]), np.vstack([q, p.q[1:]])
        return Path(t, q)

    def _fold_path(self, start_q, rest_q) -> Path:
        d = rest_q - start_q
        span = float(np.max(np.abs(d[ARM_I]))) if len(ARM_I) else 0.0
        if span <= REST_FINAL_DEG:
            return plan_joint_move(start_q, rest_q, self.cfg.safety, REST_FINAL_DEG_S / 60.0)
        mid = rest_q - d * (REST_FINAL_DEG / span)
        p1 = plan_joint_move(start_q, mid, self.cfg.safety, REST_SPEED)
        p2 = plan_joint_move(mid, rest_q, self.cfg.safety, REST_FINAL_DEG_S / 60.0)
        return Path(np.concatenate([p1.t, p1.t[-1] + p2.t[1:]]), np.vstack([p1.q, p2.q[1:]]))

    def resolve_waypoints(self, arm, start_q, waypoints):
        """[{t, joints?|pose?, gripper?}] -> [(t, q6)]; pose waypoints solved by IK seeded by the previous one;
        a pose with linear=true is densified along a straight line from the previous waypoint's tip."""
        out, prev_q, prev_t = [], np.asarray(start_q, float).copy(), 0.0
        for k, w in enumerate(waypoints):
            if "t" not in w:
                raise PlanError(f"waypoint {k}: missing t")
            t = float(w["t"])
            q = prev_q.copy()
            if "joints" in w and "pose" in w:
                raise PlanError(f"waypoint {k}: give joints or pose, not both")
            if "joints" in w:
                q = self.full_target(arm, prev_q, w["joints"])
            elif "pose" in w:
                pz = w["pose"]
                if pz.get("linear"):
                    p0 = self.models[arm].kin.pose(prev_q)["xyz"]
                    p1 = np.asarray(pz["xyz"], float)
                    n = max(int(np.linalg.norm(p1 - p0) / 0.01), 1)
                    for i in range(1, n):
                        f = i / n
                        qi = self.ik(arm, prev_q, p0 + f * (p1 - p0), pz.get("pitch"), pz.get("roll"))
                        out.append((prev_t + f * (t - prev_t), qi))
                        prev_q = qi
                q = self.ik(arm, prev_q, pz["xyz"], pz.get("pitch"), pz.get("roll"))
            if "gripper" in w:
                q[GRIPPER_I] = float(w["gripper"])
            out.append((t, q))
            prev_q, prev_t = q, t
        return out

    def trajectory_paths(self, arm, waypoints, time_scale, start):
        arms = ["left", "right"] if arm == "both" else [arm]
        paths, scales = {}, {}
        for a in arms:
            if arm == "both" and any(not isinstance(w, dict) or a not in w or "t" not in w for w in waypoints):
                raise PlanError(f"arm='both': every waypoint needs 't' and a '{a}' entry")
            wps = [({"t": w["t"], **w[a]} if arm == "both" else w) for w in waypoints]
            paths[a], scales[a] = plan_waypoints(start[a], self.resolve_waypoints(a, start[a], wps),
                                                 self.cfg.safety, time_scale)
        if arm == "both" and time_scale == "auto" and len(set(round(s, 6) for s in scales.values())) > 1:
            k = max(scales.values())
            for a in arms:                       # keep the two arms on one clock
                wps = [({"t": w["t"] * k / scales[a], **w[a]}) for w in waypoints]
                paths[a], _ = plan_waypoints(start[a], self.resolve_waypoints(a, start[a], wps), self.cfg.safety,
                                             "reject")
        return paths, max(scales.values())

    # ------------------------------------------------------------------ requests
    def build(self, spec, start) -> tuple[dict, dict]:
        """spec -> (paths {arm: Path}, meta). Raises PlanError (or ValueError) when it cannot be planned."""
        kind, sf = spec[0], self.cfg.safety
        if kind == "joints":
            _, arm, targets, speed = spec
            return {arm: self._joint_move(arm, start[arm], self.full_target(arm, start[arm], targets), speed)}, {}
        if kind == "gripper":
            _, arm, value, speed = spec
            return {arm: plan_joint_move(start[arm], self.full_target(arm, start[arm], {"gripper": value}), sf,
                                         speed)}, {}
        if kind == "pose":
            _, arm, xyz, pitch, roll, linear, speed = spec
            if linear:
                if self._folded(arm, start[arm]) and roll is not None and abs(float(roll) - start[arm][4]) > REST_ROLL_TOL_DEG:
                    raise PlanError(f"{arm}: a straight-line move out of the folded rest cannot turn the wrist (it "
                                    "would turn while folded); move out first without linear=True")
                return {arm: plan_linear(self.models[arm], sf, start[arm], xyz, pitch, roll, speed)}, {}
            return {arm: self._joint_move(arm, start[arm], self.ik(arm, start[arm], xyz, pitch, roll), speed)}, {}
        if kind == "trajectory":
            _, arm, waypoints, time_scale = spec
            paths, scale = self.trajectory_paths(arm, waypoints, time_scale, start)
            return paths, {"scale": scale}
        if kind == "rest":
            _, arms = spec
            return {a: self._rest_route(a, start) for a in arms}, {}
        if kind == "to":
            _, arm, q, speed = spec
            return {arm: plan_joint_move(start[arm], np.asarray(q, float), sf, speed)}, {}
        raise ValueError(f"unknown plan spec {kind!r}")

    def plan(self, spec, start, others, rest=False, report=False) -> tuple[dict, dict]:
        """Build and fully check. Returns (paths, meta); raises Rejected with the violations."""
        try:
            paths, meta = self.build(spec, start)
        except (PlanError, ValueError) as e:
            raise Rejected(str(e))
        v, hull_report = self.full_check_report(paths, start, others, rest=rest)
        if hull_report:                    # report-only link clearances (task 4b): sent back for robotd's log
            meta = {**meta, "link_clearances": [
                {"arm": a, "part": p, "target": k, "name": n, "min_mm": round(d * 1000, 1)}
                for (a, p, k, n), d in sorted(hull_report.items())]}
        if report:
            meta = {**meta, "report": self.path_report(paths, start)}
        if v:
            e = Rejected("rejected: " + "; ".join(x.message for x in v), [x.as_dict() for x in v])
            e.meta = meta
            raise e
        return paths, meta

    def check(self, paths, start, others, rest=False):
        return [x.as_dict() for x in self.full_check(paths, start, others, rest=rest)]

    def clearance(self, paths, current, others, window_s):
        """Launch-time arm-to-arm re-check: link capsules, and the gripper-end hulls (camera, jaws, gripper; wrist) -
        capsules alone kept the wrist axes ~90 mm apart, less than two cameras sticking out 73 mm (review
        2026-10-05). With the hulls on, the two wrist+gripper capsules are not compared (sprint 2026-10-08): the
        hulls check that pair, and they run wherever those two segments are near."""
        v = check_clearance(self.models, self.cfg.safety, paths, current, others, window_s=window_s,
                            hulls=self._hulls_on())
        if not v and self._hulls_on():
            from .hulls import GRIP_REACH_M
            hc, lock = _hull_checker(self.cfg)
            near = self._grippers_near(paths, current, others, window_s, 2 * GRIP_REACH_M + hc.broad_m)
            if near.any():
                with lock:
                    v = hc.arm_check(paths, current, others, window_s=window_s, near=near)
        return [x.as_dict() for x in v]

    def _grippers_near(self, paths, current, others, window_s, within_m) -> np.ndarray:
        """(shifts, samples) bool: the two arms' wrist+gripper segments within `within_m`, on check_clearance's
        timeline (a moving other arm up to window_s further along)."""
        cfg = self.cfg.safety
        others = {a: p for a, p in (others or {}).items() if a not in paths}
        T = max([p.duration for p in paths.values()] + [p.duration for p in others.values()], default=0.0)
        n = int(np.ceil(T / cfg.tick_s)) + 1
        K = int(np.ceil(window_s / cfg.tick_s)) if others else 0
        seg = {}
        for a, m in self.models.items():
            src = paths.get(a) or others.get(a)
            if src is None:
                seg[a] = np.repeat(_segments(m, np.asarray(current[a], float)[None, :])[:, 3], n + K, 0)
            else:
                tt = np.arange(n + K if a in others else n) * cfg.tick_s
                seg[a] = _segments(m, np.array([src.at(x) for x in tt]))[:, 3]
        (na, sa), (nb, sb) = list(seg.items())
        near = np.zeros((K + 1, n), bool)
        for k in range(K + 1):
            A = sa[k:k + n] if na in others else sa[:n]
            B = sb[k:k + n] if nb in others else sb[:n]
            near[k] = segment_distance(A[:, 0], A[:, 1], B[:, 0], B[:, 1]) < within_m
        return near

    def path_report(self, paths, start) -> dict:
        cfg = self.cfg.safety
        rep = {"duration_s": round(max(p.duration for p in paths.values()), 3), "arms": {}}
        for a, p in paths.items():
            dt = np.diff(p.t) if len(p.t) > 1 else np.array([1.0])
            v = np.abs(np.diff(p.q, axis=0)) / dt[:, None] if len(p.t) > 1 else np.zeros((1, 6))
            kp = _keypoints(self.models[a], p.q)
            floor = cfg.floor_z(kp[:, -1, 0], kp[:, -1, 1])
            rep["arms"][a] = {"peak_speed": {j: round(float(v[:, i].max()), 1) for i, j in enumerate(ARM_JOINTS)},
                              "speed_caps": cfg.vmax,
                              "lowest_tip_above_surface_mm": round(float(np.min(kp[:, -1, 2] - floor)) * 1000, 1),
                              "end_tip_xyz_m": _r(kp[-1, -1], 4)}
        if len(self.models) == 2:
            T = rep["duration_s"]
            grid = np.linspace(0, T, max(int(T / cfg.tick_s), 1) + 1)
            qs = {a: (np.array([paths[a].at(t) for t in grid]) if a in paths else np.repeat(start[a][None], len(grid), 0))
                  for a in self.models}
            sa, sb = _segments(self.models["left"], qs["left"]), _segments(self.models["right"], qs["right"])
            d = segment_distance(sa[:, :, None, 0], sa[:, :, None, 1], sb[:, None, :, 0], sb[:, None, :, 1])
            rep["closest_arm_approach_mm"] = round(float(np.min(d) - 2 * cfg.capsule_radius_m) * 1000, 1)
        return rep

    def reach(self, arm, points, current) -> dict:
        m = self.models[arm]
        lo, hi = m.limits.usable(self.cfg.safety)
        seed = current[arm]
        out = []
        for pt in points:
            xyz, pitch, roll = pt["xyz"], pt.get("pitch"), pt.get("roll")
            roll_v = float(seed[4]) if roll is None else float(roll)
            ok_pitches = m.kin.reachable_pitches(xyz, roll_v, lo[:5], hi[:5], range(-90, 31, 2))
            res = {"xyz": xyz, "pitch_ranges": _ranges(ok_pitches, 2.0)}
            r = m.kin.solve_ik(xyz, None if pitch is None else float(pitch), roll_v, seed_deg=seed[:5],
                               lower_deg=lo[:5], upper_deg=hi[:5])
            if r.ok:
                q = np.asarray(seed, float).copy()
                q[:5] = r.q
                cur = {a: (q if a == arm else current[a]) for a in self.models}
                cur = {a: v for a, v in cur.items() if v is not None}
                v = self.full_check({arm: Path(np.array([0.0]), q[None, :])}, cur, query=True)
                p = m.kin.pose(q)
                res.update({"reachable": True, "pitch": round(p["pitch"], 1), "joints": dict(zip(ARM_JOINTS, _r(q, 1))),
                            "pose_ok": not v, "pose_problems": [x.message for x in v],
                            "other_arm_checked": len(cur) == len(self.models)})
            else:
                res.update({"reachable": False, "reason": r.message,
                            "closest": {"xyz": _r(r.reached_xyz, 4), "pitch": round(r.reached_pitch_deg, 1)}})
            out.append(res)
        return {"ok": True, "arm": arm, "points": out,
                "note": "pitch: 0 level, -90 straight down; pitch_ranges lists the pitches (-90..30, 2 deg steps) "
                        "with an exact solution within the joint limits here - a slightly conservative subset; "
                        "pose_ok adds floor/obstacle/other-arm checks of the pose itself, not of the path to it"}


    # ------------------------------------------------------------------ wrist-camera viewpoints
    def _wrist_cam(self, arm, q):
        """(optical centre, unit optical axis) of the arm's wrist camera at joint vector q (CAD mount, robot.yaml)."""
        c = self.cfg.rig.camera(f"wrist_{arm}")
        k = self.models[arm].kin
        f = k.frames(np.asarray(q, float)[:5])
        G = f[k._joint_pos[4]]                                     # gripper_link, after the wrist roll
        x, y, z, w = c["rot_xyzw"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        axis = G[:3, :3] @ R[:, 2]
        return G[:3, 3] + G[:3, :3] @ np.asarray(c["offset_m"], float), axis / np.linalg.norm(axis)

    VIEW_SPEED = 0.4                    # look_at's default speed (the path's shape does not depend on it)

    def _view_move_problem(self, arm, xyz, pitch, roll, current) -> str | None:
        """Why the move look_at would make to this viewpoint is not acceptable, or None."""
        cur = {a: q for a, q in current.items() if q is not None}
        if cur.get(arm) is None:
            return None                                 # no reading: daemon.view_pose refuses earlier
        try:
            paths, _ = self.build(("pose", arm, list(xyz), pitch, roll, False, self.VIEW_SPEED), cur)
        except (PlanError, ValueError) as e:
            return f"the move there cannot be planned: {e}"
        v = self.full_check(paths, cur, query=True)        # hulls included: floor, obstacles, arms, own arm
        if v:
            return "the move there: " + "; ".join(x.message for x in v)
        return None

    VIEW_ELEVATIONS = (45, 35, 55, 25, 65, 15)
    VIEW_ROLLS = (-90.0, 0.0, 90.0)

    def view_pose(self, arm, target, current, distance=0.15, elevations=VIEW_ELEVATIONS, rolls=VIEW_ROLLS, start=0):
        """A gripper pose from which this arm's WRIST CAMERA looks straight at `target` from `distance` metres,
        its optical axis `elevation` degrees below horizontal, coming from the arm's own side. Default roll -90
        (sprint 2026-10-05, D9): the camera on top of the gripper; then roll 0 (camera beside it) and +90 (camera
        underneath) (D9 amendment, user 2026-10-06: with the camera on top, aiming it down puts the gripper below
        it - 9/16 table targets at 15 cm with -90 alone, 15/16 with the three). The camera axis is not the fingers' axis (tilted
        ~25 deg and offset sideways), so this solves for the camera: a tip-pose estimate by IK, then pan, lift,
        elbow and wrist flex refined together until the axis passes through the target at that distance and
        elevation. The search starts from a fixed seed, never the arm's current pose, so the same request gives
        the same answer wherever the arm is (review 2026-10-05). Tries the elevations in order; returns the first
        pose that is reachable and clear (floor, obstacles, the other arm where it is now) with the ACHIEVED
        distance, elevation and axis miss, or the reasons. Candidates are (roll, elevation) pairs, every
        elevation at the first roll first; `start` skips the first ones (look_at, after a refused move)."""
        from scipy.optimize import least_squares

        m = self.models[arm]
        lo, hi = m.limits.usable(self.cfg.safety)
        target = np.asarray(target, float)
        grip = float(current[arm][5]) if current.get(arm) is not None else float(self.rest[arm][5])
        pan_y = self.cfg.rig.pan_axis_y if arm == "left" else -self.cfg.rig.pan_axis_y   # the arm's pan axis
        h = target[:2] - np.array([0.0, pan_y])
        if np.linalg.norm(h) < 1e-6:
            return {"ok": False, "reason": "the target is on the arm's pan axis"}
        h = h / np.linalg.norm(h)
        rest_seed = np.r_[np.clip(self.rest[arm][:5], lo[:5], hi[:5]), grip]  # fixed: independent of the arm's pose
        elev = lambda ax: float(np.degrees(np.arcsin(np.clip(-ax[2], -1, 1))))
        cands = [(float(r), float(e)) for r in rolls for e in elevations]
        order = [[r, e] for r, e in cands]

        def residual(x, e, d):
            q = seed.copy()
            q[:4] = x
            cam, axis = self._wrist_cam(arm, q)
            return np.r_[(cam + d * axis - target) * 1000.0, elev(axis) - e]   # mm, deg

        if not 0 <= int(start) < len(cands):
            return {"ok": False, "arm": arm, "reason": f"start must be 0..{len(cands) - 1} (the candidate list)",
                    "tried": [], "candidates": order}
        tried = []
        for idx in range(int(start), len(cands)):
            roll, e = cands[idx]
            seed = rest_seed.copy()
            seed[4] = roll
            er = np.radians(float(e))
            a = np.array([np.cos(er) * h[0], np.cos(er) * h[1], -np.sin(er)])   # wanted optical axis
            cam_want = target - float(distance) * a
            q = seed.copy()
            cam, axis = self._wrist_cam(arm, q)
            p0 = m.kin.pose(q)
            tip = np.asarray(p0["xyz"], float) + (cam_want - cam)
            pitch = p0["pitch"] + elev(axis) - float(e)
            why = None
            for _ in range(6):                          # a tip pose putting the camera near cam_want
                r = m.kin.solve_ik(tip, pitch, roll, seed_deg=q[:5], lower_deg=lo[:5], upper_deg=hi[:5])
                if not r.ok:
                    why = r.message
                    break
                q = seed.copy()
                q[:5] = r.q
                cam, axis = self._wrist_cam(arm, q)
                if np.linalg.norm(cam_want - cam) < 0.002 and abs(elev(axis) - e) < 0.5:
                    break
                tip = tip + (cam_want - cam)
                pitch += float(e) - elev(axis)
            if why is None:                             # all four joints together: camera on the axis, exactly
                sol = least_squares(residual, q[:4], bounds=(lo[:4], hi[:4]), args=(float(e), float(distance)),
                                    xtol=1e-10, ftol=1e-10)
                q = seed.copy()
                q[:4] = sol.x
                res = residual(sol.x, float(e), float(distance))
                if np.max(np.abs(res[:3])) > 1.0 or abs(res[3]) > 0.5:
                    why = (f"out of reach: the camera gets within {np.linalg.norm(res[:3]):.0f} mm and "
                           f"{abs(res[3]):.1f} deg of the viewpoint")
            if why is not None:
                tried.append({"roll": roll, "elevation": e, "reason": why})
                continue
            cam, axis = self._wrist_cam(arm, q)
            d = target - cam
            miss = float(np.linalg.norm(d - (d @ axis) * axis))
            cur = {k2: (q if k2 == arm else v) for k2, v in current.items() if v is not None}
            v = self.full_check({arm: Path(np.array([0.0]), q[None, :])}, cur, query=True)
            if v:
                tried.append({"roll": roll, "elevation": e, "reason": "; ".join(x.message for x in v)})
                continue
            p = m.kin.pose(q)
            # the move look_at will send (the rounded pose, planned from where the arm is now) must pass too, and
            # neither it nor the viewpoint may fold the gripper end onto the arm itself (review 2026-10-06: a
            # +90 viewpoint had the camera inside the shoulder; a refused move cost the program a failed step)
            why = self._view_move_problem(arm, _r(p["xyz"], 4), round(p["pitch"], 1), round(float(q[4]), 1),
                                          current)
            if why:
                tried.append({"roll": roll, "elevation": e, "reason": why})
                continue
            return {"ok": True, "arm": arm, "camera": f"wrist_{arm}", "xyz": _r(p["xyz"], 4),
                    "pitch": round(p["pitch"], 1), "roll": round(float(q[4]), 1),
                    "elevation": round(elev(axis), 1), "distance": round(float(d @ axis), 4),
                    "requested": {"roll": roll, "elevation": e, "distance": float(distance)}, "candidate": idx,
                    "camera_xyz": _r(cam, 4), "axis_miss_mm": round(miss * 1000, 1),
                    "candidates": order, "skipped": tried,
                    "joints": dict(zip(ARM_JOINTS, _r(q, 1)))}
        return {"ok": False, "arm": arm, "reason": "no reachable, clear viewpoint at this distance", "tried": tried,
                "candidates": order}


# ---------------------------------------------------------------------- the process pool
_WORKER: Planner | None = None


def _init_worker(cfg, limits, hulls=True):
    global _WORKER
    try:
        os.nice(5)                                    # below the loop's process
    except OSError:
        pass
    # Exit if robotd dies uncleanly. (Not PR_SET_PDEATHSIG: it fires when the spawning THREAD exits, and pools
    # are rebuilt from short-lived request threads - review 2026-10-03.)
    parent = os.getppid()

    def watch():
        import time
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)
    import threading
    threading.Thread(target=watch, name="parent-watch", daemon=True).start()
    _WORKER = Planner(cfg, limits)
    if hulls and cfg.safety.hull_checks and cfg.rig is not None:
        _hull_checker(cfg)                            # build the MuJoCo scene now, not on the first plan


def _call(method, args, kwargs):
    return getattr(_WORKER, method)(*args, **kwargs)


class Compute:
    """Runs Planner methods inline, or in worker processes. Every call blocks its caller (a request or background
    thread) until the answer comes back; while it waits it holds no lock and no interpreter time.

    Two pools: `workers` processes for plans, path checks and reach, and one more process for the launch-time
    clearance check alone, so a launch never queues behind a long plan. A broken pool (a worker crashed) or a
    timeout is rebuilt once and the request refused with the reason (Rejected): robotd stays up and fails closed.
    """

    TIMEOUT_S = 60.0
    CLEARANCE_TIMEOUT_S = 5.0

    def __init__(self, cfg, limits: dict, workers: int = 0):
        import threading

        self.local = Planner(cfg, limits)            # inline use, and the rest poses / models robotd reports
        self._args = (cfg, limits)
        self._rebuild = threading.Lock()
        self.workers = workers
        self.pool = self.clear_pool = None
        if workers > 0:
            self.pool = self._new_pool(workers)
            self.clear_pool = self._new_pool(1)        # launch-time arm-to-arm clearance (capsules + hulls)

    def _new_pool(self, n, hulls=True):
        ctx = multiprocessing.get_context("spawn")    # no fork of a threaded process
        pool = ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                   initargs=self._args + (hulls,))
        warm = [pool.submit(_call, "full_target", (next(iter(self._args[0].arms)), np.zeros(6), {}), {})
                for _ in range(n)]                    # start every worker now, not on the first busy request
        for f in warm:
            f.result(timeout=120)
        return pool

    @staticmethod
    def _kill(pool):
        """Shut a pool down and kill its workers (shutdown alone leaves a busy - possibly hung - worker running)."""
        procs = list((getattr(pool, "_processes", None) or {}).values())
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, method, *args, clearance=False, timeout=None, **kwargs):
        if self.pool is None:
            return getattr(self.local, method)(*args, **kwargs)
        from concurrent.futures import CancelledError
        from concurrent.futures import TimeoutError as FutTimeout
        from concurrent.futures.process import BrokenProcessPool

        pool = self.clear_pool if clearance else self.pool
        try:
            return pool.submit(_call, method, args, kwargs).result(
                timeout=timeout or (self.CLEARANCE_TIMEOUT_S if clearance else self.TIMEOUT_S))
        except (BrokenProcessPool, FutTimeout, CancelledError, RuntimeError) as e:
            why = ("planning timed out" if isinstance(e, FutTimeout)
                   else "a planning worker stopped" if isinstance(e, BrokenProcessPool)
                   else f"the planning workers were being restarted ({type(e).__name__})")
            with self._rebuild:                        # one rebuild, of the pool that actually failed
                current = self.clear_pool if clearance else self.pool
                if current is pool and pool is not None:
                    try:
                        self._kill(pool)
                        fresh = self._new_pool(1) if clearance else self._new_pool(self.workers)
                        if clearance:
                            self.clear_pool = fresh
                        else:
                            self.pool = fresh
                    except Exception as e2:
                        why += f"; restarting the workers failed too ({type(e2).__name__}: {e2})"
            raise Rejected(f"planning unavailable ({why}); nothing was moved - send it again")

    def plan(self, spec, start, others, rest=False, report=False):
        return self._run("plan", spec, start, others, rest=rest, report=report)

    def check(self, paths, start, others, rest=False):
        return self._run("check", paths, start, others, rest=rest)

    def clearance(self, paths, current, others, window_s):
        T = max([p.duration for p in paths.values()] + [p.duration for p in (others or {}).values()], default=0.0)
        return self._run("clearance", paths, current, others, window_s, clearance=True,
                         timeout=self.CLEARANCE_TIMEOUT_S + 0.05 * T * (1.0 + window_s))

    def reach(self, arm, points, current):
        return self._run("reach", arm, points, current)

    def view_pose(self, arm, target, current, **kw):
        return self._run("view_pose", arm, target, current, **kw)

    def close(self):
        for p in (self.pool, self.clear_pool):
            if p is not None:
                self._kill(p)
        self.pool = self.clear_pool = None
