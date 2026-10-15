#!/usr/bin/env python3
"""Generate the "gripper, jaws and wrist camera" section of stack/rig/embodiment.md from the model (one sentence,
the arm's sag under load, is a rig measurement written here)
(sprint 2026-10-05, task 6) - so the words the agent reads cannot drift from the geometry robotd checks.

From: the camera URDF (stack/assets/so101_new_calib_camera.urdf) through robot.yaml, the jaw collision pieces, and
robotd's safety rules (robotd.yaml). For each wrist roll (0, -90, +90) at a reference pose (tip 25 cm ahead,
pitch -45): which way the jaws open and where the moving jaw is; where the wrist camera sits and how far from
the wrist axis; which way the world appears in the wrist image. Then the lowest tip height the safety layer
accepts where the camera, not the tip, sets it.

Deliberately NOT stated (an experiment withholds them from the agent): which pitches are reachable where, and how far the
jaws reach below the tip point.

    .venv/bin/python tools/gen_embodiment.py           # rewrite the section in place
    .venv/bin/python tools/gen_embodiment.py --check   # exit 1 if it is out of date (a test runs this)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DOC = REPO / "stack" / "rig" / "embodiment.md"
BEGIN = "<!-- BEGIN GENERATED gripper-camera: tools/gen_embodiment.py from the model; do not edit by hand -->"
END = "<!-- END GENERATED gripper-camera -->"
ROLLS = (0.0, -90.0, 90.0)
PITCHES = (-60.0, -45.0, -30.0, -15.0, 0.0)
REF = (0.25, 0.12)                 # reference tip: 25 cm ahead, 12 cm to the arm's own side of the midline


def _quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _ik(P, arm, xyz, pitch, roll):
    m = P.models[arm]
    lo, hi = m.limits.usable(P.cfg.safety)
    r = m.kin.solve_ik(list(xyz), pitch, roll, seed_deg=np.array([0, 10, 20, 50, roll], float),
                       lower_deg=lo[:5], upper_deg=hi[:5])
    return np.r_[r.q, 30.0] if r.ok else None


def _side(v, ax):
    """Words for where vector v points, across the gripper axis ax: top / underneath / the arm's left / right."""
    up = np.array([0.0, 0.0, 1.0]) - ax[2] * ax
    up /= np.linalg.norm(up)
    left = np.cross(up, ax)                       # ax forward, up up: up x forward points to the left
    v = v - (v @ ax) * ax
    a, b = v @ up, v @ left
    if abs(a) >= abs(b):
        return "on top" if a > 0 else "underneath"
    return "on the arm's left" if b > 0 else "on the arm's right"


def _img_word(x, up, ways=8):
    """A direction in the image (x to the right, `up` upward) as a word (8 or 4 ways)."""
    ang = np.degrees(np.arctan2(up, x)) % 360
    if ways == 4:
        return ["right", "top", "left", "bottom"][int(((ang + 45) % 360) // 90)]
    return ["right", "upper right", "up", "upper left", "left", "lower left", "down", "lower right"][
        int(((ang + 22.5) % 360) // 45)]


def _jaw_axis(P, arm, q):
    """The jaws' closing direction seen from above (deg, mod 180): fixed jaw -> moving jaw, gripper half open."""
    from stack.robotd.compute import _hull_checker
    import mujoco
    hc, lock = _hull_checker(P.cfg)
    with lock:
        hc._set({arm: np.r_[q[:5], 60.0]})
        pts = {"moving": [], "fixed": []}
        for g, part in hc.geom_part.items():
            if hc.geom_arm[g] == arm and part == "jaws":
                body = mujoco.mj_id2name(hc.model, mujoco.mjtObj.mjOBJ_BODY, hc.model.geom_bodyid[g])
                pts["moving" if "moving_jaw" in body else "fixed"].append(hc.data.geom_xpos[g].copy())
    v = np.mean(pts["moving"], 0) - np.mean(pts["fixed"], 0)
    return float(np.degrees(np.arctan2(v[1], v[0]))), v


def _wrap(a, period=360.0):
    return (a + period / 2) % period - period / 2


def geometry(P, arm: str = "right") -> dict:
    """Sprint 2026-10-07 (C2, C3, C5): how the jaws close and turn with the roll, the wrist image's orientation at
    any roll, and the wrist camera's pose relative to the tip - all with the gripper pointing steeply down."""
    k = P.models[arm].kin
    cam = P.cfg.rig.camera(f"wrist_{arm}")
    Rc = _quat(*cam["rot_xyzw"])
    pan_y = P.cfg.rig.pan_axis_y if arm == "left" else -P.cfg.rig.pan_axis_y
    y = REF[1] if arm == "left" else -REF[1]
    tipw = (REF[0], y, 0.06)

    def ik(roll):
        q = _ik(P, arm, tipw, steep, roll)
        if q is None:
            raise RuntimeError(f"gen_embodiment: the reference pose {tipw} at pitch {steep} roll {roll} is out of reach "
                               "(robot.yaml or the URDF changed?) - pick another reference")
        return q
    heading = float(np.degrees(np.arctan2(tipw[1] - pan_y, tipw[0])))
    steep = -70.0
    out = {"heading": heading}
    # C2: the closing direction at roll -90 relative to the heading; how roll turns it
    a90, _ = _jaw_axis(P, arm, ik(-90.0))
    out["across_heading_deg"] = abs(_wrap(a90 - heading, 180.0))            # 90 = exactly across
    lo, hi = _jaw_axis(P, arm, ik(-100.0))[0], _jaw_axis(P, arm, ik(-80.0))[0]
    out["deg_per_roll_deg"] = _wrap(hi - lo, 180.0) / 20.0                  # > 0: CCW seen from above (slope at -90)
    a_lo, a_hi = _jaw_axis(P, arm, ik(-90.0))[0], _jaw_axis(P, arm, ik(90.0))[0]
    out["turn_over_180_deg"] = (a_hi - a_lo) % 360.0                         # total turn, roll -90 -> +90
    # C3: image up (projected on the table) vs heading + (roll + 90), over the rolls
    errs = {}                                   # signed: > 0 = the real image-up is CCW of the formula
    for roll in (-90.0, -45.0, 0.0, 45.0, 90.0):
        G = k.frames(ik(roll)[:5])[k._joint_pos[4]]
        up = -(G[:3, :3] @ Rc)[:, 1]
        errs[roll] = float(_wrap(np.degrees(np.arctan2(up[1], up[0])) - (heading + roll + 90.0)))
    out["image_formula_err_deg"] = errs
    # C5: the wrist camera relative to the tip and the fingers' direction (roll -90)
    q = ik(-90.0)
    G = k.frames(q[:5])[k._joint_pos[4]]
    tip = np.asarray(k.pose(q)["xyz"], float)
    ax = (tip - G[:3, 3]) / np.linalg.norm(tip - G[:3, 3])
    c = G[:3, 3] + G[:3, :3] @ np.asarray(cam["offset_m"], float)
    opt = (G[:3, :3] @ Rc)[:, 2]
    d = c - tip
    out["camera_behind_tip_mm"] = float(-(d @ ax) * 1000)
    out["camera_off_axis_mm"] = float(np.linalg.norm(d - (d @ ax) * ax) * 1000)
    out["camera_tilt_deg"] = float(np.degrees(np.arccos(np.clip(opt @ ax, -1, 1))))
    return out


def facts(P, arm: str = "right") -> dict:
    """The numbers, from the model (one arm: both arms are mirror-free copies of the same model)."""
    from stack.robotd.compute import _hull_checker
    import mujoco

    k = P.models[arm].kin
    cam = P.cfg.rig.camera(f"wrist_{arm}")
    Rc = _quat(*cam["rot_xyzw"])
    y = REF[1] if arm == "left" else -REF[1]
    hc, lock = _hull_checker(P.cfg)
    out = {"rolls": {}, "rest_roll": float(P.cfg.safety.rest_roll_deg)}
    for roll in ROLLS:
        q = _ik(P, arm, (REF[0], y, 0.10), -45.0, roll)
        G = k.frames(q[:5])[k._joint_pos[4]]
        tip = np.asarray(k.pose(q)["xyz"], float)
        wrist = G[:3, 3]
        ax = (tip - wrist) / np.linalg.norm(tip - wrist)
        c = wrist + G[:3, :3] @ np.asarray(cam["offset_m"], float)
        Ro = G[:3, :3] @ Rc                                     # optical frame: x right, y down, z forward
        off = c - wrist
        perp = off - (off @ ax) * ax
        img = {name: _img_word(v @ Ro[:, 0], -(v @ Ro[:, 1]))
               for name, v in (("world_up", np.array([0.0, 0.0, 1.0])), ("robot_left", np.array([0.0, 1.0, 0.0])))}
        t = tip - c
        jaws_in_image = _img_word(t @ Ro[:, 0], -(t @ Ro[:, 1]), ways=4)
        with lock:
            hc._set({arm: np.r_[q[:5], 60.0]})
            pts = {"moving": [], "fixed": []}
            for g, part in hc.geom_part.items():
                if hc.geom_arm[g] == arm and part == "jaws":
                    body = mujoco.mj_id2name(hc.model, mujoco.mjtObj.mjOBJ_BODY, hc.model.geom_bodyid[g])
                    pts["moving" if "moving_jaw" in body else "fixed"].append(hc.data.geom_xpos[g].copy())
        d = np.mean(pts["moving"], 0) - np.mean(pts["fixed"], 0)
        out["rolls"][roll] = {"camera": _side(perp, ax), "camera_mm": float(np.linalg.norm(perp) * 1000),
                              "moving_jaw": _side(d, ax), "image": img, "jaws_in_image": jaws_in_image}
    # the lowest tip height accepted, per roll x pitch, where something other than the tip rule sets it
    from stack.robotd.safety import Path as MPath
    cur = {a: P.rest[a].copy() for a in P.models}
    lows = {}
    for roll in ROLLS:
        for pitch in PITCHES:
            lo_z, hi_z = 0.0, 0.25
            q = _ik(P, arm, (REF[0], y, hi_z), pitch, roll)
            if q is None or P.full_check({arm: MPath(np.array([0.0]), q[None, :])}, {**cur, arm: q}):
                continue
            why = None
            for _ in range(10):                                 # bisect to 0.25 mm
                z = (lo_z + hi_z) / 2
                q = _ik(P, arm, (REF[0], y, z), pitch, roll)
                v = None if q is None else P.full_check({arm: MPath(np.array([0.0]), q[None, :])}, {**cur, arm: q})
                if q is not None and not v:
                    hi_z = z
                else:
                    lo_z = z
                    why = v[0].rule if v else why
            lows[(roll, pitch)] = (hi_z, why)
    tip_floor = P.cfg.safety.tip_clearance_m
    out["camera_limited"] = {k2: round(z * 1000) for k2, (z, why) in lows.items() if why == "camera_clearance"}
    # the configured tip rule, not the lowest accepted height: below 20 mm the jaw hulls set that height, and it
    # would give away how far the jaws reach below the tip point (withheld; sprint 2026-10-09 review)
    out["tip_rule_mm"] = round(tip_floor * 1000)
    out["geometry"] = geometry(P, arm)
    return out


def section(f: dict) -> str:
    r = f["rolls"]
    roll_word = {0.0: "0", -90.0: "-90", 90.0: "+90"}
    jaws_open = {"on top": "up and down", "underneath": "up and down",
                 "on the arm's left": "sideways", "on the arm's right": "sideways"}
    lines = [BEGIN, "",
             "**Gripper, jaws and wrist camera at each wrist roll.** Positive wrist_roll turns the gripper",
             "counter-clockwise as seen from behind it. The wrist camera is fixed to the gripper and turns with it.",
             "",
             "| wrist_roll | jaws open | moving jaw | wrist camera | in the wrist image |",
             "|---|---|---|---|---|"]
    for roll in ROLLS:
        x = r[roll]
        lines.append(f"| {roll_word[roll]} | {jaws_open[x['moving_jaw']]} | {x['moving_jaw']} | {x['camera']}, "
                     f"{x['camera_mm']:.0f} mm from the wrist axis | the robot's left is {x['image']['robot_left']}, "
                     f"world up is {x['image']['world_up']} |")
    jw = {x["jaws_in_image"] for x in r.values()}
    import textwrap
    para = (f"The jaws are always at the {'/'.join(sorted(jw))} of the wrist image, whatever the roll. Robot programs use "
            "`WORK_ROLL` = -90 for approaches and grasps: the camera is on top and the jaws open sideways. At roll "
            "+90 the camera hangs below the gripper, so the safety layer keeps the tip higher")
    cl = [z for (ro, p), z in f["camera_limited"].items() if ro == 90.0]
    if cl:                                       # one bound, not per pitch: which pitches are reachable is withheld
        para += (f" the flatter the gripper points: up to about {round(max(cl), -1) / 10:.0f} cm above the surface "
                 f"under it when level (otherwise the tip point keeps at least {f['tip_rule_mm']} mm, as at the other rolls; "
                 f"other parts of the gripper keep their own clearance, which can hold the tip higher).")
    else:
        para += "."
    para += (" `look_at` uses roll -90 where it can, else 0 or +90; its result says which. At rest the arm is folded"
             f" at roll {f['rest_roll']:.0f}; the rest motion turns the wrist only with the arm unfolded.")
    lines += [""] + textwrap.wrap(para, 110)
    g = f["geometry"]
    e = g["image_formula_err_deg"]
    worst = max(abs(v) for v in e.values())
    cw = all(v <= 0.5 for v in e.values())
    paras = [
        f"**How the jaws turn.** At wrist_roll -90 the jaws close across the arm's heading (the direction from the "
        f"arm's shoulder to the tip, seen from above), to within about {abs(90 - g['across_heading_deg']):.0f} deg. With "
        f"the gripper pointing steeply down (about -70), increasing wrist_roll turns the jaws' closing direction "
        f"{'counter-clockwise' if g['deg_per_roll_deg'] > 0 else 'clockwise'} as seen from above, roughly one degree per "
        f"degree of roll ({abs(g['deg_per_roll_deg']):.2f} near -90; {g['turn_over_180_deg']:.0f} deg from -90 to +90). As "
        "the gripper points flatter the roll turns them less and also tilts them.",
        "**The wrist image at any roll.** With the gripper pointing steeply down (about -70), \"up\" in the wrist "
        "image points along the arm's heading turned by (wrist_roll + 90) deg counter-clockwise, seen from above: "
        "along the heading at -90, to the heading's left at 0, back toward the robot at +90. That holds to within "
        f"about {max(abs(e[-90.0]), abs(e[90.0])):.0f} deg at -90 and +90; in between the real image-up is up to "
        f"about {worst:.0f} deg " + ("clockwise of it" if cw else "off it") + ", and more as the gripper points flatter.",
        f"**The wrist camera's pose.** Its optical centre is about {g['camera_behind_tip_mm']:.0f} mm behind the tip "
        f"along the gripper and {g['camera_off_axis_mm']:.0f} mm off the gripper's axis (on the side the table above "
        f"gives for each roll); it looks {g['camera_tilt_deg']:.0f} deg off the direction the fingers point.",
        "**Commanded and measured.** Under load, with the arm reaching out, the arm sags: the measured tip can sit "
        "1-2 cm below the commanded one (measured on the rig). `state()` gives both: `tip_xyz_m` is where the arm "
        "is, and while torque is on `command` is where robotd holds it.",
    ]
    for p in paras:
        lines += [""] + textwrap.wrap(p, 110)
    lines += ["", END]
    return "\n".join(lines)


def planner():
    from stack.robotd.compute import Planner
    from stack.robotd.config import load_config
    from stack.robotd.safety import load_arm_limits
    cfg = load_config()
    return Planner(cfg, {n: load_arm_limits(a.calibration_id) for n, a in cfg.arms.items()})


def render(doc: str, sec: str) -> str:
    if BEGIN in doc:
        i, j = doc.index(BEGIN), doc.index(END) + len(END)
        return doc[:i] + sec + doc[j:]
    return doc.rstrip("\n") + "\n\n" + sec + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = DOC.read_text()
    new = render(doc, section(facts(planner())))
    if a.check:
        if new != doc:
            print("stack/rig/embodiment.md: the generated section is out of date; run tools/gen_embodiment.py")
            return 1
        print("up to date")
        return 0
    DOC.write_text(new)
    print("wrote", DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main())
