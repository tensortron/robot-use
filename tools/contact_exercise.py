#!/usr/bin/env python3
"""Free-air motion for rig check 13: collect the lag and current of normal motion, to set contact thresholds.

Drives ONE arm through robotd's API (it plans and safety-checks every move as usual): single-joint nudges of
10 deg around a raised pose, then a tour of larger poses, at speeds 0.2 / 0.5 / 1.0, repeated until the time
is up. Run robotd with `contact.mode: record_only` so each motion's per-tick lag and current land in the
event log as `contact_samples`; tools/contact_thresholds.py turns them into thresholds.

    .venv/bin/python tools/contact_exercise.py --arm right --check          # plan every leg offline, move nothing
    .venv/bin/python tools/contact_exercise.py --arm right --minutes 25     # run it (someone at the 12 V)

Stops at the first motion that ends in anything but `done` (a fault, a stop, a rejection it cannot route
around), and on Ctrl-C (robotd stops the arm when asked; the arm then holds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HOME = [0, -70, 70, 30, 0, 2]
# [pan (outward +), lift, elbow, wrist_flex, roll, gripper]: free air, tip well above the board
TOUR = [
    [30, -40, 40, 0, 60, 50],
    [-20, -20, 10, 40, -60, 2],
    [15, 10, -20, 30, 90, 30],
    [0, -90, 90, -30, -90, 2],
    [40, 20, -40, 20, 0, 2],
]
SPEEDS = (0.2, 0.5, 1.0)


def pose(arm: str, p) -> list[float]:
    """Outward pan is + for the right arm and - for the left (positive pan turns toward the robot's right)."""
    q = [float(v) for v in p]
    if arm == "left":
        q[0] = -q[0]
    return q


def legs(arm: str):
    """One cycle: (speed, label, target) for every move, starting and ending at HOME."""
    out = []
    for speed in SPEEDS:
        for i, j in enumerate(JOINTS[:5]):
            for d in (10, -10):
                q = list(HOME)
                q[i] += d                     # nudges are symmetric, so no outward flip needed
                out.append((speed, f"{j} {d:+d}", pose(arm, q)))
                out.append((speed, "home", pose(arm, HOME)))
        for k, p in enumerate(TOUR):
            out.append((speed, f"tour {k + 1}", pose(arm, p)))
        out.append((speed, "home", pose(arm, HOME)))
    return out


class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=45) as r:      # robotd caps wait_s at 30
            return json.loads(r.read())

    def post(self, path: str, body: dict):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"status": "error", "reason": e.read().decode(errors="replace")}


def check(arm: str, api: Api) -> int:
    """Plan and safety-check every leg in order, from the arm's real position, without moving anything."""
    from stack.rig.kinematics import ArmKinematics
    from stack.robotd.config import load_config
    from stack.robotd.planner import plan_joint_move
    from stack.robotd.safety import ArmModel, check_paths, load_arm_limits

    cfg = load_config()
    st = api.get("/state")["arms"]
    cur = {a: np.array([st[a]["joints"][j] for j in JOINTS], float) for a in ("left", "right")}
    model = ArmModel(arm, ArmKinematics(cfg.arms[arm].mount), load_arm_limits(cfg.arms[arm].calibration_id))
    q = cur[arm].copy()
    seq = [(0.3, "home (from where it is)", pose(arm, HOME))] + legs(arm)
    bad = 0
    total = 0.0
    for speed, label, target in seq:
        path = plan_joint_move(q, np.array(target, float), cfg.safety, speed)
        v = check_paths({arm: model}, cfg.safety, {arm: path}, {**cur, arm: q})
        total += path.duration
        if v:
            bad += 1
            print(f"REJECT {speed:.1f} {label:22s} {target}: {v[0].rule}: {v[0].message}")
        q = np.array(target, float)
    tips = [model.kin.pose(np.array(t[:5], float))["xyz"] for _, _, t in seq]
    print(f"{len(seq)} legs, {bad} rejected; one cycle {total:.0f} s of motion; "
          f"lowest tip z {min(p[2] for p in tips) * 100:.1f} cm")
    return 1 if bad else 0


def run(arm: str, api: Api, minutes: float) -> int:
    mode = (api.get("/config").get("contact") or {}).get("mode")
    if mode != "record_only":
        print(f"robotd's contact.mode is {mode!r}: start it with a config whose contact.mode is record_only, "
              "or this run records nothing")
        return 2
    t_end = time.time() + minutes * 60
    n = cycle = 0
    seq = [(0.3, "home", pose(arm, HOME))]
    try:
        while True:
            for speed, label, target in seq:
                out = api.post("/motion/move_joints", {"arm": arm, "targets": dict(zip(JOINTS, target)), "speed": speed})
                mid = out.get("motion_id")
                if out.get("status") == "pending_approval":
                    print("waiting for approval on the operator page (ARM) ...", flush=True)
                if not mid or out.get("status") in ("rejected", "error"):
                    print(f"STOP: {label} at speed {speed} was {out.get('status')}: {out.get('reason')}")
                    return 1
                res = api.get(f"/motion/{mid}?wait_s=60")
                while res.get("status") in ("pending_approval", "starting", "running"):
                    res = api.get(f"/motion/{mid}?wait_s=60")
                n += 1
                print(f"{time.strftime('%H:%M:%S')} c{cycle} #{n:4d} {speed:.1f} {label:14s} {res['status']}"
                      f" {res.get('duration_s', 0):5.2f}s", flush=True)
                if res["status"] != "done":
                    print(f"STOP: motion {mid} ended {res['status']}: {res.get('reason')}")
                    return 1
            if time.time() >= t_end:
                break
            cycle += 1
            seq = legs(arm)
    except KeyboardInterrupt:
        print("interrupted: stopping the arm")
        api.post("/motion/stop", {"arm": arm})
        return 130
    except Exception as e:                            # robotd unreachable, a timeout, a bad reply: stop the arm
        print(f"STOP: {type(e).__name__}: {e}")
        try:
            api.post("/motion/stop", {"arm": arm})
        except Exception:
            pass
        return 1
    print(f"done: {n} motions in {cycle} cycles")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--api", default="http://127.0.0.1:8770")
    ap.add_argument("--check", action="store_true", help="plan and safety-check every leg; move nothing")
    a = ap.parse_args()
    api = Api(a.api)
    return check(a.arm, api) if a.check else run(a.arm, api, a.minutes)


if __name__ == "__main__":
    sys.exit(main())
