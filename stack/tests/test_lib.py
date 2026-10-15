"""robot_use: the program-facing API. Unit tests against a fake runtime, then for real through the runner."""

import textwrap

import numpy as np
import pytest

import robot_use
from robot_use import arm, contact, elapsed, estimate, geo, gripper_stalled, joint_crosses, parallel, wait_until
from robot_use import runtime
from robot_use.arm import WaitTimeout



class Fake:
    def __init__(self):
        self.calls, self.status, self.report, self.refine = [], "done", {}, None

    def __call__(self, op, **kw):
        self.calls.append((op, kw))
        if op == "step":
            return {"status": self.status, "kind": kw["kind"], "motion_id": "m1", "report": self.report,
                    "reason": None if self.status == "done" else "because"}
        if op == "state":
            return {"arms": {"right": {"tip_xyz_m": [0.2, -0.18, 0.1], "pitch_deg": -60.0, "roll_deg": 5.0,
                                       "joints": {"gripper": 30.0}}}}
        if op == "checkpoint":
            est = {n: v["value"] for n, v in kw["estimates"].items()}
            if self.refine:
                est.update(self.refine)
            return {"decision": "refine" if self.refine else "continue", "estimates": est}
        return None


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(runtime, "_rpc", f)
    return f


def test_move_to_sends_one_checked_step(fake):
    r = arm("right").move_to([0.25, -0.15, 0.10], pitch=-90, speed=0.4, until=contact())
    op, kw = fake.calls[-1]
    assert op == "step" and kw["kind"] == "move_pose"
    a = kw["args"]
    assert a["arm"] == "right" and a["xyz"] == [0.25, -0.15, 0.10] and a["pitch"] == -90 and a["speed"] == 0.4
    assert a["until"] == {"type": "contact", "level": "light"} and a["linear"] is False
    assert r.status == "done" and r.condition_met is None
    with pytest.raises(TypeError, match="not the Estimate"):
        arm("right").move_to(estimate("b", [0.2, 0.0]))
    with pytest.raises(ValueError):
        arm("middle")


def test_move_by_keeps_pitch_and_roll(fake):
    arm("right").move_by(dz=0.05)
    a = fake.calls[-1][1]["args"]
    assert np.allclose(a["xyz"], [0.2, -0.18, 0.15]) and a["pitch"] == -60.0 and a["roll"] == 5.0 and a["linear"]


def test_close_reports_a_grasp_and_failures_raise(fake):
    fake.report = {"grasped": {"arm": "right", "gripper": 12.0, "current_ma": 300}}
    g = arm("right").close(effort=0.4)
    assert g.grasped["gripper"] == 12.0
    a = fake.calls[-1][1]["args"]
    assert a["value"] == 0.0 and a["max_effort"] == 0.4
    fake.status = "blocked"
    with pytest.raises(robot_use.StepFailed, match="blocked"):
        arm("right").move_joints(shoulder_pan=10)


def test_conditions_build_robotd_dicts():
    assert contact("firm", ["elbow_flex"]) == {"type": "contact", "level": "firm", "joints": ["elbow_flex"]}
    assert gripper_stalled(0.2) == {"type": "gripper_stalled", "effort": 0.2}
    assert joint_crosses("shoulder_pan", 20, 1)["direction"] == 1
    assert elapsed(0.5) == {"type": "time", "s": 0.5}


def test_estimate_points_and_checkpoint_refinement(fake):
    b = estimate("block", [0.25, -0.12], size=(0.04, 0.04, 0.04), tol=0.03, source="ego")
    assert b.top == [0.25, -0.12, 0.04] and b.centre == [0.25, -0.12, 0.02] and b.above(0.08)[2] == pytest.approx(0.12)
    fake.refine = {"block": [0.26, -0.13]}
    assert robot_use.checkpoint("hover", b, views=["wrist_right"], need="centred") == "refine"
    sent = fake.calls[-1][1]
    assert sent["estimates"]["block"]["tol"] == 0.03 and sent["views"] == ["wrist_right"]
    assert b.xy == [0.26, -0.13] and b.history[-1]["source"] == "checkpoint 'hover'"
    with pytest.raises(ValueError, match="distinct"):
        robot_use.checkpoint("x", b, estimate("block", [0, 0]))


def test_parallel_is_one_two_arm_motion(fake):
    parallel(arm("left").target(xyz=[0.2, 0.15, 0.1], pitch=-90), arm("right").target(shoulder_pan=-10, gripper=50))
    kw = fake.calls[-1][1]
    assert kw["kind"] == "trajectory" and kw["args"]["arm"] == "both" and kw["args"]["time_scale"] == "auto"
    wp = kw["args"]["waypoints"][0]
    assert wp["left"]["pose"]["xyz"] == [0.2, 0.15, 0.1] and wp["right"] == {"joints": {"shoulder_pan": -10}, "gripper": 50.0}


def test_wait_until():
    t = iter([False, False, True])
    assert wait_until(lambda: next(t), timeout_s=2, poll_s=0.01) >= 0
    with pytest.raises(WaitTimeout):
        wait_until(lambda: False, timeout_s=0.1, poll_s=0.02)
    with pytest.raises(ValueError):
        wait_until(lambda: True, timeout_s=0)


def test_geometry_round_trips_and_object_tops():
    for p in ([0.20, -0.15, 0.0], [0.35, 0.10, 0.0], [0.30, -0.15, 0.04]):
        u, v = geo.table_to_pixel(p)
        assert geo.distance(geo.pixel_to_table(u, v, z=p[2]), p) < 1e-6
    u, v = geo.table_to_pixel([0.30, -0.15, 0.04])           # a 4 cm block's top
    on_table = geo.pixel_to_table(u, v, z=0.0)
    assert 0.015 < geo.distance(on_table[:2], [0.30, -0.15]) < 0.035, "reading the top as the table is ~2 cm off"
    assert geo.frame_scale(640) == 2.0
    with pytest.raises(ValueError, match="horizon"):
        geo.pixel_to_table(640, -2000)


# ------------------------------------------------------------------------------------- in the sandbox
from .test_runner import approve, rig, run, trace  # noqa: E402,F401


def test_the_library_inside_a_sandboxed_program(rig):  # noqa: F811
    rd, runner = rig
    rd.arm_ios["right"].block["gripper"] = (14.0, -1)          # something 14% open between the jaws
    code = textwrap.dedent("""
        from robot_use import arm, estimate, contact, checkpoint, parallel, log, elapsed, geo
        r = arm("right")
        r.open()
        block = estimate("block", [0.22, -0.17], size=(0.04, 0.04, 0.04), tol=0.03, source="test")
        r.move_to(block.above(0.08), pitch=-60, speed=0.8)
        if checkpoint("hover", block, views=["wrist_right"], need="centred") == "refine":
            r.move_to(block.above(0.08), pitch=-60, speed=0.8)
        g = r.close(effort=0.3)
        log("grasped " + str(bool(g.grasped)))
        r.move_by(dz=0.03)
        res = r.move_joints(shoulder_pan=-30, speed=0.3, until=elapsed(0.3))
        log("until " + str(res.status))
        parallel(arm("left").target(shoulder_pan=10), arm("right").target(shoulder_pan=-10))
        log("pixel " + str([round(v) for v in geo.table_to_pixel(block.top)]))
    """)
    r = runner()
    st = run(rd, r, code)
    assert st["status"] == "checkpoint", st
    pid = st["program_id"]
    res = r.resume(pid, "refine", {"block": [0.225, -0.175]})
    assert res["ok"], res
    st = r.status(pid, wait_s=20)
    assert st["status"] == "done", (st.get("error"), st.get("summary"), st.get("outcome"))
    logs = [x["msg"] for x in trace(st) if x["type"] == "log"]
    assert "grasped True" in logs and "until condition_met" in logs, logs
    assert any(m.startswith("pixel [") for m in logs), "geometry works inside the sandbox"


def test_move_by_starts_from_the_commanded_pose_not_the_sagging_one(fake, monkeypatch):
    """Rig 2026-10-03: the elbow sags ~3 deg, so the measured tip sits ~1 cm below where robotd holds it; a
    relative move from the measured pose drifted the target."""
    measured = {"tip_xyz_m": [0.2, -0.18, 0.088], "pitch_deg": -63.0, "roll_deg": 5.0, "joints": {"gripper": 30.0},
                "command": {"tip_xyz_m": [0.2, -0.18, 0.1], "pitch_deg": -60.0, "roll_deg": 5.0}}
    orig = fake.__class__.__call__

    def call(self, op, **kw):
        if op == "state":
            return {"arms": {"right": measured}}
        return orig(self, op, **kw)
    monkeypatch.setattr(fake.__class__, "__call__", call)
    arm("right").move_by(dz=0.05)
    a = fake.calls[-1][1]["args"] if fake.calls else None
    assert a is not None and np.allclose(a["xyz"], [0.2, -0.18, 0.15]) and a["pitch"] == -60.0


def test_look_at_asks_for_a_viewpoint_then_makes_one_move(fake, monkeypatch):
    from robot_use import NoViewpoint
    orig = fake.__class__.__call__
    answers = {"view": {"ok": True, "camera": "wrist_left", "xyz": [0.18, -0.05, 0.08], "pitch": -70.0, "roll": -90.0,
                        "elevation": 44.9, "distance": 0.1502, "camera_xyz": [0.14, 0.0, 0.13], "axis_miss_mm": 0.0,
                        "requested": {"roll": -90.0, "elevation": 45, "distance": 0.15}, "candidate": 0,
                        "candidates": [[-90.0, 45.0], [90.0, 45.0]]}}

    def call(self, op, **kw):
        if op == "view_pose":
            self.calls.append((op, kw))
            return answers["view"]
        return orig(self, op, **kw)
    monkeypatch.setattr(fake.__class__, "__call__", call)
    v = arm("left").look_at(estimate("block", [0.2, -0.08], size=(0.04, 0.04, 0.04)), distance=0.15)
    (op1, kw1), (op2, kw2) = fake.calls[-2:]
    assert op1 == "view_pose" and kw1["target"] == [0.2, -0.08, 0.02] and kw1["distance"] == 0.15
    assert op2 == "step" and kw2["kind"] == "move_pose" and kw2["args"]["xyz"] == [0.18, -0.05, 0.08]
    assert v["camera"] == "wrist_left" and v["step"].status == "done"
    assert v["elevation"] == 44.9 and v["distance"] == 0.1502 and v["axis_miss_mm"] == 0.0    # as achieved
    answers["view"] = {"ok": False, "reason": "no reachable, clear viewpoint", "tried": [{"elevation": 45}]}
    n = len(fake.calls)
    with pytest.raises(NoViewpoint):
        arm("left").look_at([0.9, 0.0, 0.0])
    assert [c[0] for c in fake.calls[n:]] == ["view_pose"]          # no move was attempted


def test_look_at_tries_the_next_viewpoint_when_the_move_there_is_refused(fake, monkeypatch):
    """Review 2026-10-05: a refused path is not a failed step - try the next viewpoint, else NoViewpoint."""
    from robot_use import NoViewpoint, StepFailed
    orig = fake.__class__.__call__
    cands = [[-90.0, 45.0], [-90.0, 35.0], [90.0, 45.0]]
    starts = []

    def call(self, op, **kw):
        if op == "view_pose":
            self.calls.append((op, kw))
            k = kw["start"]
            starts.append(k)
            r, e = cands[k]
            return {"ok": True, "camera": "wrist_left", "xyz": [0.18, -0.05, 0.08], "pitch": -70.0, "roll": r,
                    "elevation": e, "distance": 0.15, "camera_xyz": [0.1, 0, 0.1], "axis_miss_mm": 0.0,
                    "requested": {"roll": r, "elevation": e, "distance": 0.15}, "candidate": k, "candidates": cands}
        return orig(self, op, **kw)
    monkeypatch.setattr(fake.__class__, "__call__", call)
    fake.status = "rejected"
    with pytest.raises(NoViewpoint) as e:
        arm("left").look_at([0.2, -0.08, 0.02])
    assert starts == [0, 1, 2]
    assert [(t["roll"], t["elevation"]) for t in e.value.tried] == [tuple(c) for c in cands]
    assert all("the move there" in t["reason"] for t in e.value.tried)
    fake.status = "stopped"                          # it moved, then failed: that IS a failed step
    with pytest.raises(StepFailed):
        arm("left").look_at([0.2, -0.08, 0.02])
