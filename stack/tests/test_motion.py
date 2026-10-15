"""robotd step 3: checked motion, approval, contact, grasp, stop, faults, idle rest -- on mock arms."""

import time

import numpy as np
import pytest

from stack.robotd.config import load_config
from stack.robotd.daemon import EventLog, RobotD
from stack.robotd.hardware import MockArmIO
from stack.rig.kinematics import ARM_JOINTS
from stack.robotd.motion import Motion
from stack.robotd.safety import ArmLimits, Path

from .conftest import CALIBRATED
from .test_robotd import REPO_CONFIG

J = {j: i for i, j in enumerate(ARM_JOINTS)}
START = [0.0, -60.0, 60.0, 40.0, 0.0, 20.0]


def make(tmp_path, compute_workers=0, **over):
    cfg = load_config(REPO_CONFIG)
    cfg.log_dir, cfg.overlay, cfg.cameras = tmp_path, None, {}
    cfg.motion_enabled = True
    cfg.idle_rest_s = 1e9
    for k, v in over.items():
        if k.startswith("contact_"):
            setattr(cfg.contact, k[8:], v)
        else:
            setattr(cfg, k, v)
    arms = {n: MockArmIO(n, q=START) for n in cfg.arms}
    rd = RobotD(cfg, arms, {}, limits={n: ArmLimits(dict(CALIBRATED[n])) for n in cfg.arms}, event_log=EventLog(tmp_path),
                compute_workers=compute_workers)
    rd.supervisor.mode = cfg.approval_mode
    rd.start()
    wait(lambda: all(st.reading is not None for st in rd.arms.values()))
    rd.settle()                                    # the background pose checks (tests share the hull checker)
    return rd


def wait(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    raise AssertionError("timed out")


def done(rd, out, timeout=15.0):
    assert out.get("status") != "rejected", out
    return rd.motion_status(out["motion_id"], wait_s=timeout)


@pytest.fixture
def rd(tmp_path):
    r = make(tmp_path)
    yield r
    r.stop()


def q(rd, arm):
    return rd.arms[arm].reading.q.copy()


def test_disabled_rejects_everything(tmp_path):
    r = make(tmp_path, motion_enabled=False)
    try:
        out = r.move_joints("left", {"elbow_flex": 50})
        assert out["status"] == "rejected" and "read-only" in out["reason"]
        assert r.arm_ios["left"].writes == 0
    finally:
        r.stop()


def test_first_motion_needs_approval_then_runs_free(rd):
    out = rd.move_joints("left", {"elbow_flex": 45, "shoulder_pan": 10}, speed=1.0)
    assert out["status"] == "pending_approval"
    assert rd.arm_ios["left"].writes == 0, "nothing may move before approval"
    assert rd.approve()["started"] == [out["motion_id"]]
    st = done(rd, out)
    assert st["status"] == "done", st
    assert abs(q(rd, "left")[J["elbow_flex"]] - 45) < 3 and abs(q(rd, "left")[J["shoulder_pan"]] - 10) < 3
    out2 = rd.move_joints("left", {"elbow_flex": 55}, speed=1.0)
    assert out2["status"] == "running", "armed session: no second approval"
    assert done(rd, out2)["status"] == "done"


def test_every_motion_mode(tmp_path):
    r = make(tmp_path, approval_mode="every_motion")
    try:
        for target in (50, 55):
            out = r.move_joints("right", {"elbow_flex": target}, speed=1.0)
            assert out["status"] == "pending_approval"
            r.approve()
            assert done(r, out)["status"] == "done"
    finally:
        r.stop()


def test_out_of_range_is_rejected_with_the_rule(rd):
    before = rd.arm_ios["left"].writes
    out = rd.move_joints("left", {"elbow_flex": 120})
    assert out["status"] == "rejected" and out["violations"][0]["rule"] == "joint_range"
    assert "outside usable range" in out["reason"]
    out = rd.move_joints("left", {"elbw": 10})
    assert out["status"] == "rejected" and "unknown joints" in out["reason"]
    assert rd.arm_ios["left"].writes == before


def test_busy_arm_rejects_a_second_motion(rd):
    rd.supervisor.arm()
    a = rd.move_joints("left", {"shoulder_pan": 30}, speed=0.2)
    b = rd.move_joints("left", {"shoulder_pan": -10})
    assert b["status"] == "rejected" and "busy" in b["reason"]
    c = rd.move_joints("right", {"shoulder_pan": -10}, speed=1.0)        # the other arm is free
    assert c["status"] == "running"
    rd.halt("both")
    done(rd, a)


def test_obstruction_is_detected_backed_off_and_needs_reapproval(rd):
    rd.supervisor.arm()
    rd.arm_ios["right"].block["elbow_flex"] = (40.0, -1)          # can't go below 40 deg
    out = rd.move_joints("right", {"elbow_flex": 10}, speed=0.5)
    st = done(rd, out)
    assert st["status"] == "blocked", st
    assert st["report"]["blocked"]["joint"] == "elbow_flex" and "blocked" in st["reason"]
    wait(lambda: any(m.kind == "backoff" for m in rd.motions.values()))     # planned off the loop thread
    bo = next(m for m in rd.motions.values() if m.kind == "backoff")
    assert rd.motion_status(bo.id, wait_s=10)["status"] == "done"
    assert q(rd, "right")[J["elbow_flex"]] > 41.0, "backed off away from the obstacle"
    goal = rd.arm_ios["right"].goal[J["elbow_flex"]]
    assert abs(goal - q(rd, "right")[J["elbow_flex"]]) < 1.0, "no longer pushing"
    nxt = rd.move_joints("right", {"elbow_flex": 60}, speed=1.0)
    assert nxt["status"] == "pending_approval", "a blocked arm needs approval again, even when armed"
    other = rd.move_joints("left", {"elbow_flex": 50}, speed=1.0)
    assert other["status"] == "running", "the other arm is unaffected"
    done(rd, other)


def test_record_only_mode_logs_contact_but_does_not_act(tmp_path):
    r = make(tmp_path, contact_mode="record_only")
    try:
        r.supervisor.arm()
        r.arm_ios["left"].block["elbow_flex"] = (40.0, -1)
        st = done(r, r.move_joints("left", {"elbow_flex": 10}, speed=1.0))
        assert st["status"] == "done" and st["report"]["contact_recorded"]["joint"] == "elbow_flex"
        assert "not_settled" in st["report"]
    finally:
        r.stop()


def test_gripper_grasp_stops_at_effort(rd):
    rd.supervisor.arm()
    rd.arm_ios["left"].block["gripper"] = (12.0, -1)               # an object 12% open
    st = done(rd, rd.set_gripper("left", 0.0, speed=1.0, max_effort=0.3))
    assert st["status"] == "done" and st["report"]["grasped"]["arm"] == "left", st
    assert rd.arm_ios["left"].goal[J["gripper"]] > 10.0, "gripper holds at the object, not squeezing to 0"
    # an empty close just closes
    rd.arm_ios["right"].block.clear()
    st = done(rd, rd.set_gripper("right", 0.0, speed=1.0, max_effort=0.3))
    assert st["status"] == "done" and "grasped" not in st.get("report", {})


def test_a_full_close_goes_to_0_and_below_is_refused(rd):
    """The jaws touch at 1-2 on the rig (2026-10-11): a full close aims at 0 so the servo presses them together."""
    rd.supervisor.arm()
    assert rd.cfg.safety.gripper_usable == (0.0, 98.0)
    done(rd, rd.set_gripper("right", 50.0, speed=1.0))
    st = done(rd, rd.set_gripper("right", 0.0, speed=1.0, max_effort=0.5))
    assert st["status"] == "done" and rd.arm_ios["right"].goal[J["gripper"]] < 0.5
    before = rd.arm_ios["right"].writes
    out = rd.set_gripper("right", -0.5, speed=1.0)
    assert out["status"] == "rejected" and out["violations"][0]["rule"] == "joint_range", out
    assert rd.arm_ios["right"].writes == before


def test_stop_decelerates_and_disarms(rd):
    rd.supervisor.arm()
    out = rd.move_joints("left", {"shoulder_pan": 60}, speed=1.0)
    time.sleep(0.6)
    pan_at_stop = q(rd, "left")[J["shoulder_pan"]]
    res = rd.halt("both")
    assert res["ok"] and res["stopped"] == ["left"]
    assert rd.motion_status(out["motion_id"])["status"] == "stopped"
    wait(lambda: not rd._active("left"))
    time.sleep(0.2)
    final = q(rd, "left")[J["shoulder_pan"]]
    assert final < 55 and final - pan_at_stop < 20, "stopped well short of the target, within the braking distance"
    assert rd.supervisor.armed is False
    assert rd.move_joints("left", {"shoulder_pan": 0})["status"] == "pending_approval"


def test_bus_failure_during_motion_is_sticky_until_reset(rd):
    rd.supervisor.arm()
    out = rd.move_joints("right", {"shoulder_pan": 40}, speed=0.3)
    time.sleep(0.3)
    rd.arm_ios["right"].fail_reads = 5
    st = done(rd, out)
    assert st["status"] == "faulted"
    wait(lambda: rd.arms["right"].failed_reads == 0)
    assert rd.arms["right"].fault and rd.arms["right"].sticky_fault, "stays faulted after reads recover"
    assert rd.move_joints("right", {"shoulder_pan": 0})["status"] == "rejected"
    assert rd.reset_arm("right")["ok"]
    assert rd.arms["right"].fault is None
    nxt = rd.move_joints("right", {"shoulder_pan": 0}, speed=1.0)
    assert nxt["status"] == "pending_approval"
    rd.approve()
    assert done(rd, nxt)["status"] == "done"


def test_move_pose_and_linear(rd):
    rd.supervisor.arm()
    m = rd.models["left"]
    target_q = np.array([15.0, -40.0, 50.0, 40.0, 0.0, 20.0])
    p = m.kin.pose(target_q)
    st = done(rd, rd.move_pose("left", p["xyz"].tolist(), pitch=p["pitch"], speed=1.0))
    assert st["status"] == "done", st
    assert np.linalg.norm(m.kin.fk(q(rd, "left"))[:3, 3] - p["xyz"]) < 0.01
    tgt = (p["xyz"] + np.array([0.03, 0.0, 0.02])).tolist()
    out = rd.move_pose("left", tgt, linear=True, speed=1.0)
    assert out["status"] == "running" and "joint_solution" in out
    assert done(rd, out)["status"] == "done"
    assert np.linalg.norm(m.kin.fk(q(rd, "left"))[:3, 3] - tgt) < 0.01
    bad = rd.move_pose("left", [0.9, 0.2, 0.1])
    assert bad["status"] == "rejected" and "unreachable" in bad["reason"]
    assert rd.solve_ik("left", tgt)["ok"] and not rd.solve_ik("left", [0.9, 0.2, 0.1])["ok"]


def test_trajectory_both_arms_and_validation(rd):
    rd.supervisor.arm()
    wps = [{"t": 1.0, "left": {"joints": {"elbow_flex": 50}}, "right": {"joints": {"elbow_flex": 50}}},
           {"t": 2.0, "left": {"joints": {"elbow_flex": 60}, "gripper": 40}, "right": {"joints": {"elbow_flex": 65}}}]
    rep = rd.validate_trajectory("both", wps)
    assert rep["ok"] and rep["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert set(rep["arms"]) == {"left", "right"} and rep["closest_arm_approach_mm"] > 40
    fast = [{"t": 0.2, "left": {"joints": {"shoulder_pan": 40}}, "right": {"joints": {"shoulder_pan": -40}}}]
    assert not rd.validate_trajectory("both", fast)["ok"]
    auto = rd.validate_trajectory("both", fast, time_scale="auto")
    assert auto["ok"] and auto["time_scale_applied"] > 1
    st = done(rd, rd.run_trajectory("both", wps))
    assert st["status"] == "done", st
    assert abs(q(rd, "right")[J["elbow_flex"]] - 65) < 3 and abs(q(rd, "left")[J["gripper"]] - 40) < 3
    assert rd.run_trajectory("both", [{"t": 1.0, "left": {"joints": {}}}])["status"] == "rejected"


def test_go_rest_and_idle_rest(tmp_path):
    r = make(tmp_path, idle_rest_s=0.5)
    try:
        r.supervisor.arm()
        assert done(r, r.move_joints("left", {"elbow_flex": 70}, speed=1.0))["status"] == "done"
        try:
            wait(lambda: any(m.kind == "idle_rest" for m in r.motions.values()), timeout=10)
        except AssertionError:                        # say why: a rejected or failed idle rest is in the event log
            log = "".join(f.read_text() for f in tmp_path.glob("*.jsonl"))
            raise AssertionError("no idle rest within 10 s; event log: "
                                 + " | ".join(ln for ln in log.splitlines() if "idle_rest" in ln)[-2000:]) from None
        m = next(m for m in r.motions.values() if m.kind == "idle_rest")
        assert r.motion_status(m.id, wait_s=30)["status"] == "done"
        assert np.max(np.abs(q(r, "left")[:5] - r.rest["left"][:5])) < 3
        # one idle rest per idle period: its planning took longer than the period, and planning threads used to
        # pile up (one more every period) and lose to each other as "busy" rejections
        log = "".join(f.read_text() for f in tmp_path.glob("*.jsonl"))
        assert "idle_rest_rejected" not in log and sum(m.kind == "idle_rest" for m in r.motions.values()) == 1, log
    finally:
        r.stop()


def test_approval_rejects_if_the_arm_moved_while_pending(rd):
    out = rd.move_joints("left", {"elbow_flex": 50})
    rd.arm_ios["left"].torque_off()
    rd.arms["left"].torque = [0] * 6
    rd.arm_ios["left"].q[J["shoulder_pan"]] += 10                 # someone moved it by hand
    time.sleep(0.1)
    res = rd.approve()
    assert res["started"] == []
    assert rd.motion_status(out["motion_id"])["status"] == "rejected"


def test_torque_off_arm_is_configured_and_stiffened_before_moving(rd):
    io = rd.arm_ios["right"]
    io.torque_off()
    rd.arms["right"].torque = [0] * 6
    rd.arms["right"].q_cmd = None
    rd.supervisor.arm()
    st = done(rd, rd.move_joints("right", {"elbow_flex": 50}, speed=1.0))
    assert st["status"] == "done" and io.configured == 1 and all(io.torque)


def test_step_check_aborts_a_jumping_path(rd):
    rd.supervisor.arm()
    q0 = rd._start_q("left")
    q1 = q0.copy()
    q1[J["shoulder_pan"]] += 30
    bad = Motion("test", ["left"], {"left": Path(np.array([0.0, 0.02, 0.04, 1.0]), np.array([q0, q0, q1, q1]))})
    with rd._lock:
        rd.motions[bad.id] = bad
        bad.status = "starting"
    rd._prepare_and_launch(bad)                    # the real path: stiffen outside the lock, then launch
    st = rd.motion_status(bad.id, wait_s=5)
    assert st["status"] == "stopped" and "step check" in st["reason"]
    # the 30 deg jump was never sent. Under load a tick can land just after t = 0.02 and send one goal part of
    # the way (less than a tick's limit), which is allowed; the next tick's step check aborts the motion.
    sent = abs(rd.arm_ios["left"].goal[J["shoulder_pan"]] - q0[J["shoulder_pan"]])
    cap = rd.cfg.safety.vmax["shoulder_pan"] * rd._period * 2.0 + 0.5
    assert sent <= cap, f"the jump was sent: {sent:.1f} deg (one tick allows {cap:.1f})"


def test_none_mode_needs_localhost(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(REPO_CONFIG.read_text().replace("approval_mode: first_motion", "approval_mode: none"))
    with pytest.raises(ValueError, match="mcp_localhost_only"):
        load_config(p)
    p.write_text(p.read_text().replace("mcp_localhost_only: false", "mcp_localhost_only: true"))
    assert load_config(p).approval_mode == "none"


# ---------------------------------------------------------------------------------------- step 2
def test_clearance_is_checked_against_the_other_arms_path(rd):
    rd.supervisor.arm()
    left = rd.move_joints("left", {"shoulder_pan": 40, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0}, speed=0.15)
    assert left["status"] == "running"
    right = rd.move_joints("right", {"shoulder_pan": -40, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0}, speed=1.0)
    arm_rules = {"arm_clearance", "camera_clearance", "jaws_clearance", "gripper_clearance", "wrist_clearance"}
    assert right["status"] == "rejected" and arm_rules & {v["rule"] for v in right["violations"]}, right
    assert "the left arm's" in right["reason"] or "arm_clearance" in {v["rule"] for v in right["violations"]}, right
    rd.halt("both")
    done(rd, left)


def test_registration_refuses_if_the_other_arm_started_moving_meanwhile(rd):
    from stack.robotd.daemon import Rejected
    rd.supervisor.arm()
    with rd._lock:
        snap = rd._preflight(["right"])
    m = rd._plan("move_joints", ["right"], ("to", "right", snap.start["right"] + np.array([5, 0, 0, 0, 0, 0]), 1.0),
                 snap)
    other = rd.move_joints("left", {"shoulder_pan": 20}, speed=0.2)        # starts while "right" is being planned
    with rd._lock:
        with pytest.raises(Rejected, match="started or stopped moving"):
            rd._register(m, snap)
    rd.halt("both")
    done(rd, other)


def test_approval_rechecks_against_an_arm_that_started_moving(rd):
    rd.supervisor.arm()
    rd.supervisor.rearm_required.add("right")                 # right's next motion must wait for ARM
    right = rd.move_joints("right", {"shoulder_pan": -40, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0}, speed=1.0)
    assert right["status"] == "pending_approval"
    left = rd.move_joints("left", {"shoulder_pan": 40, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0}, speed=0.15)
    assert left["status"] == "running"
    res = rd.approve()
    assert right["motion_id"] not in res["started"]
    st = rd.motion_status(right["motion_id"])
    assert st["status"] == "rejected" and "other arm is moving" in st["reason"]
    rd.halt("both")
    done(rd, left)


def test_lease_gives_exclusive_command_and_expiry_stops_the_arm(rd):
    rd.supervisor.arm()
    lease = rd.acquire_lease("program p1", ttl_s=0.6)
    assert lease["ok"]
    assert not rd.acquire_lease("someone else")["ok"]
    out = rd.move_joints("left", {"elbow_flex": 50})
    assert out["status"] == "rejected" and "leased by 'program p1'" in out["reason"]
    mine = rd.move_joints("left", {"shoulder_pan": 45}, speed=0.15, lease=lease["token"])
    assert mine["status"] == "running"
    assert rd.renew_lease(lease["token"])["ok"]
    st = rd.motion_status(mine["motion_id"], wait_s=5)        # stop renewing: the "program" died
    assert st["status"] == "stopped" and "lease" in st["reason"]
    assert rd.lease is None and rd.supervisor.armed is False
    events = [__import__("json").loads(l)["event"] for l in rd.events.path.read_text().splitlines()]
    assert "lease_expired" in events
    assert not rd.renew_lease(lease["token"])["ok"]
    out = rd.move_joints("left", {"elbow_flex": 50}, lease=lease["token"])
    assert out["status"] == "rejected" and "expired" in out["reason"]


def test_stop_needs_no_lease_and_release_frees_the_robot(rd):
    rd.supervisor.arm()
    lease = rd.acquire_lease("p2", ttl_s=10)
    mine = rd.move_joints("right", {"shoulder_pan": -45}, speed=0.15, lease=lease["token"])
    assert rd.halt("both")["ok"]                               # the operator / anyone can always stop
    assert rd.motion_status(mine["motion_id"])["status"] == "stopped"
    assert rd.release_lease(lease["token"])["ok"]
    rd.supervisor.arm()
    wait(lambda: not rd._active("right"))
    assert rd.move_joints("right", {"shoulder_pan": 0}, speed=1.0)["status"] == "running"


def test_until_contact_ends_the_step_before_it_counts_as_blocked(rd):
    rd.supervisor.arm()
    rd.arm_ios["right"].block["elbow_flex"] = (40.0, -1)
    st = done(rd, rd.move_joints("right", {"elbow_flex": 10}, speed=0.5, until={"type": "contact", "level": "light"}))
    assert st["status"] == "condition_met", st
    hit = st["report"]["condition_met"]["right"]
    assert hit["type"] == "contact" and hit["joint"] == "elbow_flex"
    assert "right" not in rd.supervisor.rearm_required, "an expected touch is not an obstruction"
    goal = rd.arm_ios["right"].goal[J["elbow_flex"]]
    assert abs(goal - q(rd, "right")[J["elbow_flex"]]) < 1.0, "holds where it touched, not pushing"


def test_until_joint_crosses_and_time_decelerate(rd):
    rd.supervisor.arm()
    st = done(rd, rd.move_joints("left", {"shoulder_pan": 60}, speed=0.5,
                                 until={"type": "joint_crosses", "joint": "shoulder_pan", "value": 20, "direction": 1}))
    assert st["status"] == "condition_met" and st["report"]["condition_met"]["left"]["type"] == "joint_crosses"
    wait(lambda: not rd._active("left"))
    time.sleep(0.2)
    assert 20 <= q(rd, "left")[J["shoulder_pan"]] < 40, "stopped soon after crossing, decelerating, well short of 60"
    st = done(rd, rd.move_joints("right", {"shoulder_pan": -50}, speed=0.3, until={"type": "time", "s": 0.4}))
    assert st["status"] == "condition_met"


def test_until_not_met_is_reported_and_bad_conditions_rejected(rd):
    rd.supervisor.arm()
    st = done(rd, rd.move_joints("left", {"elbow_flex": 55}, speed=1.0, until={"type": "contact"}))
    assert st["status"] == "done" and st["report"]["condition_met"] is None
    for bad in ({"type": "touch"}, {"type": "contact", "level": "hard"},
                {"type": "joint_crosses", "joint": "elbow_flex", "value": 3}, {"type": "time", "s": 0}):
        out = rd.move_joints("left", {"elbow_flex": 50}, until=bad)
        assert out["status"] == "rejected" and "until" in out["reason"], bad


def test_lease_and_until_over_http(rd):
    import json, socket, urllib.request
    from stack.robotd.http import serve
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = serve(rd, "127.0.0.1", port, operator=False)

    def post(path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        rd.supervisor.arm()
        code, lease = post("/lease/acquire", {"holder": "http test", "ttl_s": 5})
        assert code == 200 and lease["ok"]
        code, out = post("/motion/move_joints", {"arm": "left", "targets": {"elbow_flex": 55}, "speed": 1.0,
                                                 "until": {"type": "time", "s": 0.2}, "lease": lease["token"]})
        assert code == 200 and out["status"] == "running" and out["until"][0]["type"] == "time"
        assert post("/lease/renew", {"token": lease["token"]})[1]["ok"]
        assert post("/lease/release", {"token": lease["token"]})[1]["ok"]
        assert post("/lease/steal", {})[0] == 404
        assert post("/lease/acquire", {"holder": "x", "bogus": 1})[0] == 400
    finally:
        srv.shutdown()


def test_stiffening_an_arm_never_stalls_the_loop(rd):
    """Regression, rig 2026-10-03: preparing a torque-off arm under the main lock overran the 50 Hz loop."""
    io = rd.arm_ios["right"]
    io.torque_off()
    rd.arms["right"].torque = [0] * 6
    rd.arms["right"].q_cmd = None
    rd.arms["right"].prepared = False
    slow = io.configure_for_motion

    def slow_configure():
        time.sleep(0.2)                              # a real arm's configuration is ~100 ms of bus traffic
        slow()
    io.configure_for_motion = slow_configure
    before = rd.overruns
    out = rd.move_joints("right", {"elbow_flex": 50}, speed=1.0)
    assert out["status"] == "pending_approval"
    res = rd.approve()
    assert res["status"][out["motion_id"]] == "running", res
    assert done(rd, out)["status"] == "done"
    assert rd.overruns == before, "the loop overran while the arm was being stiffened"


def test_planning_a_long_move_never_stalls_the_loop(rd):
    """Regression, rig 2026-10-03: a ~150 ms move_pose plan starved the loop's bus reads of the GIL (77 ms tick)."""
    for io in rd.arm_ios.values():
        read = io.read

        def slow_read(read=read):
            for _ in range(3):                       # sync reads that release the GIL while waiting, like serial I/O
                time.sleep(0.0012)
            return read()
        io.read = slow_read
    time.sleep(0.2)
    ticks = rd.ticks
    out = rd.move_pose("right", [0.28, -0.186, 0.05], pitch=-60, speed=0.3)
    assert out["status"] == "pending_approval", out
    time.sleep(0.1)
    worst = max(rd._tick_ms[-(rd.ticks - ticks):])
    assert worst < 25, f"a loop tick took {worst:.0f} ms while the move was being planned"


def test_record_only_logs_every_tick_of_lag_and_current(tmp_path):
    """A bench check sets the contact thresholds from these samples (the 99.9th percentile of free motion)."""
    import json
    r = make(tmp_path, contact_mode="record_only", approval_mode="none")
    try:
        out = r.move_joints("right", {"elbow_flex": 40}, speed=1.0)
        res = done(r, out)
        assert res["status"] == "done"
        time.sleep(0.2)                                # written off the loop thread
        recs = [json.loads(line) for p in tmp_path.glob("*.jsonl") for line in p.read_text().splitlines()]
        s = [x for x in recs if x["event"] == "contact_samples"]
        assert len(s) == 1 and s[0]["motion_id"] == out["motion_id"] and s[0]["arm"] == "right"
        rows = s[0]["rows"]
        assert len(rows) >= 0.8 * res["duration_s"] * 50 and all(len(row) == 10 for row in rows)
        assert s[0]["joints"][2] == "elbow_flex" and max(row[2] for row in rows) > 0
    finally:
        r.stop()


def test_act_mode_logs_no_samples(rd, tmp_path):
    import json
    rd.supervisor.mode = "none"
    done(rd, rd.move_joints("right", {"elbow_flex": 40}, speed=1.0))
    recs = [json.loads(line) for p in rd.events.dir.glob("*.jsonl") for line in p.read_text().splitlines()]
    assert not [x for x in recs if x["event"] == "contact_samples"]


# ---- independent review 2026-10-03 (findings 1, 2, 4, 5, 6, 7)

def _colliding_pair():
    """Each fine alone from START; together the gripper ends come within ~3 mm. (Pan +/-30, found by the reviewer,
    met only the old wrist+gripper capsules: its real parts stay 25+ mm apart, and since sprint 2026-10-08 the hulls
    judge that pair.)"""
    return ({"shoulder_pan": 40, "shoulder_lift": -20, "elbow_flex": 0, "wrist_flex": 40},
            {"shoulder_pan": -40, "shoulder_lift": -20, "elbow_flex": 0, "wrist_flex": 40})


def test_two_motions_approved_together_are_checked_against_each_other(rd):
    left, right = _colliding_pair()
    ol, orr = rd.move_joints("left", left, speed=1.0), rd.move_joints("right", right, speed=1.0)
    assert ol["status"] == orr["status"] == "pending_approval"
    res = rd.approve()
    assert res["started"] == [ol["motion_id"]], res
    assert res["status"][orr["motion_id"]] == "rejected" and "other arm" in res["reasons"][orr["motion_id"]]


def test_a_motion_launching_after_the_other_arm_started_is_rechecked(rd):
    """The left arm is stiffened (slow) while the right arm's colliding motion is planned and launched."""
    rd.supervisor.mode = "none"
    io = rd.arm_ios["left"]
    io.torque_off()
    rd.arms["left"].torque, rd.arms["left"].q_cmd, rd.arms["left"].prepared = [0] * 6, None, False
    slow = io.configure_for_motion
    io.configure_for_motion = lambda: (time.sleep(0.3), slow())
    left, right = _colliding_pair()
    import threading
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("l", rd.move_joints("left", left, speed=1.0)))
    t.start()
    wait(lambda: any(m.status == "starting" for m in rd.motions.values()))
    o_r = rd.move_joints("right", right, speed=1.0)
    t.join()
    statuses = sorted([box["l"]["status"], o_r["status"]])
    assert statuses == ["rejected", "running"], (box["l"], o_r)


def test_torque_off_needs_approval_before_the_arm_is_stiffened_again(rd):
    out = rd.move_joints("right", {"elbow_flex": 50})
    rd.approve()
    done(rd, out)
    lease = rd.acquire_lease("prog", ttl_s=30)["token"]
    assert rd.torque_off("right")["ok"]
    assert rd.lease is None, "a program holding the lease must end when the operator drops an arm"
    assert not rd.supervisor.armed and "right" in rd.supervisor.rearm_required
    nxt = rd.move_joints("right", {"elbow_flex": 40})
    assert nxt["status"] == "pending_approval"
    assert not any(rd.arm_ios["right"].torque), "nothing re-stiffens the arm until the operator approves"
    del lease


def test_arm_with_nothing_pending_does_not_clear_re_approval(rd):
    rd.supervisor.rearm_required.add("right")
    rd.approve()
    assert "right" in rd.supervisor.rearm_required
    out = rd.move_joints("right", {"elbow_flex": 40})
    assert out["status"] == "pending_approval"
    rd.approve()                                       # approving that motion is what clears it
    assert "right" not in rd.supervisor.rearm_required


def test_reset_arm_does_its_bus_io_outside_the_main_lock(rd):
    io = rd.arm_ios["right"]
    ts = io.torque_state
    io.torque_state = lambda: (time.sleep(0.3), ts())[1]
    before = rd.overruns
    assert rd.reset_arm("right")["ok"]
    assert rd.overruns == before


def test_no_idle_rest_while_a_program_holds_the_lease(tmp_path):
    r = make(tmp_path, idle_rest_s=0.3)
    try:
        r.supervisor.mode = "none"
        done(r, r.move_joints("right", {"elbow_flex": 30}))
        r.acquire_lease("prog at a checkpoint", ttl_s=30)
        time.sleep(1.0)
        assert not [m for m in r.motions.values() if m.kind == "idle_rest"]
    finally:
        r.stop()


# ---- second review of the fixes, 2026-10-03

def test_launch_is_checked_against_an_idle_arm_that_moved_into_the_way(rd):
    left, right = _colliding_pair()
    rd.supervisor.armed = True
    rd.supervisor.rearm_required.add("left")
    ol = rd.move_joints("left", left, speed=1.0)                 # pends (re-approval), planned vs right at START
    assert ol["status"] == "pending_approval"
    assert done(rd, rd.move_joints("right", right, speed=1.0))["status"] == "done"   # right moves in and stops
    res = rd.approve()
    assert res["started"] == [] and res["status"][ol["motion_id"]] == "rejected", res
    assert "left" in rd.supervisor.rearm_required, "a motion that never ran does not settle re-approval"


def test_torque_off_during_a_batch_launch_is_not_undone(rd):
    import threading
    for a in ("left", "right"):
        rd.arm_ios[a].torque_off()
        st = rd.arms[a]
        st.torque, st.q_cmd, st.prepared = [0] * 6, None, False
    slow = rd.arm_ios["left"].configure_for_motion
    rd.arm_ios["left"].configure_for_motion = lambda: (time.sleep(0.5), slow())
    ol = rd.move_joints("left", {"elbow_flex": 50}, speed=1.0)
    orr = rd.move_joints("right", {"elbow_flex": 50}, speed=1.0)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", rd.approve()))
    t.start()
    time.sleep(0.2)                                              # left is being stiffened
    assert rd.torque_off("right")["ok"]
    t.join()
    time.sleep(0.2)
    assert not any(rd.arm_ios["right"].torque) and rd.arms["right"].q_cmd is None
    assert rd.motions[orr["motion_id"]].status == "stopped"
    del ol


def test_a_slow_motion_launches_while_the_other_arm_moves(rd):
    rd.supervisor.mode = "none"
    a = rd.move_joints("left", {"shoulder_pan": -20, "elbow_flex": 30}, speed=0.1)
    b = rd.move_joints("right", {"shoulder_pan": 20, "elbow_flex": 30}, speed=0.1)   # a ~5 s path: slow to check
    assert a["status"] == "running" and b["status"] == "running", b
    done(rd, a), done(rd, b)


def test_arm_after_a_long_idle_stays_armed(rd):
    rd.supervisor.last_activity -= rd.supervisor.armed_timeout_s + 10
    rd.approve()
    time.sleep(0.1)
    assert rd.supervisor.armed


def test_a_slow_launch_check_is_fine_while_the_other_arm_is_idle(rd, monkeypatch):
    """Review 2026-10-03: the launch window applied even with the other arm still, so a long path (slow to
    check) was rejected with "the other arm kept changing" in the twin."""
    import stack.robotd.compute as C
    orig = C.check_clearance
    monkeypatch.setattr(C, "check_clearance", lambda *a, **k: (time.sleep(0.3), orig(*a, **k))[1])
    rd.supervisor.mode = "none"
    out = rd.move_joints("right", {"elbow_flex": 40}, speed=1.0)
    assert out["status"] == "running", out


def test_check_reach_answers_where_and_at_which_pitches(rd):
    out = rd.check_reach("right", [{"xyz": [0.2, -0.08, 0.04]}, {"xyz": [0.30, -0.12, 0.13], "pitch": -90},
                                   {"xyz": [0.9, 0.0, 0.1]}])
    a, b, c = out["points"]
    assert a["reachable"] and a["pose_ok"] and a["pitch_ranges"] and set(a["joints"]) >= {"shoulder_pan", "wrist_flex"}
    assert not b["reachable"] and b["closest"]["xyz"] and b["pitch_ranges"] and b["pitch_ranges"][0][0] > -90
    assert not c["reachable"] and c["pitch_ranges"] == []
    assert not rd.check_reach("right", [])["ok"] and not rd.check_reach("middle", [{"xyz": [0, 0, 0]}])["ok"]


def test_state_reports_the_commanded_pose(rd):
    rd.supervisor.mode = "none"
    done(rd, rd.move_joints("right", {"elbow_flex": 40}, speed=1.0))
    cmd = rd.state()["arms"]["right"]["command"]
    assert abs(cmd["joints"]["elbow_flex"] - 40) < 1e-6 and len(cmd["tip_xyz_m"]) == 3


def test_check_reach_refuses_non_finite_input(rd):
    assert not rd.check_reach("right", [{"xyz": [float("nan"), 0, 0.1]}])["ok"]
    assert not rd.check_reach("right", [{"xyz": [0.2, -0.1, 0.1], "pitch": float("inf")}])["ok"]



def test_planning_in_worker_processes_behaves_like_inline(tmp_path):
    """The real daemon plans in worker processes (compute.py) so planning never stalls its 50 Hz loop."""
    r = make(tmp_path, compute_workers=2)
    try:
        assert r.compute.pool is not None
        r.supervisor.mode = "none"
        assert done(r, r.move_joints("right", {"elbow_flex": 40}, speed=1.0))["status"] == "done"
        out = r.move_pose("right", [0.25, -0.15, 0.08], pitch=-60, linear=False, speed=0.8)
        assert done(r, out)["status"] == "done", out
        bad = r.move_joints("right", {"elbow_flex": 999})
        assert bad["status"] == "rejected" and bad["violations"] and bad["violations"][0]["rule"] == "joint_range"
        reach = r.check_reach("left", [{"xyz": [0.2, 0.08, 0.05]}])
        assert reach["ok"] and reach["points"][0]["reachable"]
        v = r.validate_trajectory("right", [{"t": 1.0, "joints": {"elbow_flex": 30}}])
        assert v["ok"] and "duration_s" in v
        assert done(r, r.go_rest("right"))["status"] == "done"
    finally:
        r.stop()


def test_a_crashed_planning_worker_is_refused_cleanly_and_replaced(tmp_path):
    """Review 2026-10-03: a dead worker broke the pool for good - every later plan raised, and a launch could
    be left 'starting'. Now the request is refused with the reason, the pool is rebuilt, the next one works."""
    import os
    import signal
    r = make(tmp_path, compute_workers=1)
    try:
        r.supervisor.mode = "none"
        for pid in list(r.compute.pool._processes):
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        out = r.move_joints("right", {"elbow_flex": 40}, speed=1.0)
        assert out["status"] == "rejected" and "planning unavailable" in out["reason"], out
        out = r.move_joints("right", {"elbow_flex": 40}, speed=1.0)
        assert done(r, out)["status"] == "done", out
        for pid in list(r.compute.clear_pool._processes):     # the launch check's own worker
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        out = r.move_joints("right", {"elbow_flex": 30}, speed=1.0)
        st = done(r, out) if out.get("motion_id") else out
        assert st["status"] in ("faulted", "rejected") and "unavailable" in (st.get("reason") or ""), st
        assert not [m for m in r.motions.values() if m.status == "starting"]
        out = r.move_joints("right", {"elbow_flex": 30}, speed=1.0)
        assert done(r, out)["status"] == "done", out
    finally:
        r.stop()



def test_a_pool_rebuilt_from_a_short_lived_thread_survives_it(tmp_path):
    """Review 2026-10-03: PR_SET_PDEATHSIG killed workers when the request THREAD that rebuilt the pool ended."""
    import os
    import signal
    import threading
    r = make(tmp_path, compute_workers=1)
    try:
        for pid in list(r.compute.pool._processes):
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("r", r.check_reach("right", [{"xyz": [0.2, -0.08, 0.04]}])))
        t.start()
        t.join()
        assert not box["r"]["ok"] and "planning unavailable" in box["r"]["reason"]
        time.sleep(2.5)                                  # the rebuilding thread is gone; its workers must not be
        again = r.check_reach("right", [{"xyz": [0.2, -0.08, 0.04]}])
        assert again["ok"], again
    finally:
        r.stop()


def test_view_pose_aims_the_wrist_camera_not_the_fingers(rd):
    import numpy as np
    cur = {a: rd._start_q(a) for a in rd.arms}
    P = rd.compute.local
    for arm, tgt in (("left", [0.20, -0.08, 0.02]), ("right", [0.20, 0.02, 0.03])):
        v = P.view_pose(arm, tgt, cur, distance=0.15)
        assert v["ok"], v
        q = np.array([v["joints"][j] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                              "wrist_roll", "gripper")])
        cam, axis = P._wrist_cam(arm, q)
        d = np.asarray(tgt) - cam
        # looking straight at it from 15 cm (task 5: the camera's sideways offset is solved, not ignored)
        assert abs(d @ axis - 0.15) < 0.002 and np.linalg.norm(d - (d @ axis) * axis) < 0.002
        assert abs(v["distance"] - d @ axis) < 0.0005 and v["axis_miss_mm"] < 2.0
        assert abs(v["elevation"] - np.degrees(np.arcsin(-axis[2]))) < 0.2
    # the same request gives the same viewpoint wherever the arm is now (review 2026-10-05)
    moved = {a: q.copy() for a, q in cur.items()}
    moved["left"][:4] = [25.0, 15.0, 35.0, -25.0]
    assert P.view_pose("left", [0.20, -0.08, 0.02], cur)["joints"] == P.view_pose("left", [0.20, -0.08, 0.02], moved)["joints"]
    far = P.view_pose("left", [0.9, -0.5, 0.0], cur, distance=0.1)
    assert not far["ok"] and far["tried"]
    assert not rd.view_pose("left", [0.2, 0.0, float("nan")])["ok"]
    assert not rd.view_pose("left", [0.2, 0.0, 0.0], distance=2.0)["ok"]


# ---- sprint 2026-10-05 task 4: convex-hull checks of the gripper end ----------------------------------------

def _grasp_q(rd, roll, z=0.022, pitch=-35.0, xy=(0.30, -0.11)):
    m = rd.models["right"]
    lo, hi = m.limits.usable(rd.cfg.safety)
    r = m.kin.solve_ik([xy[0], xy[1], z], pitch, roll, seed_deg=np.array([0, 10, 20, 50, roll], float),
                       lower_deg=lo[:5], upper_deg=hi[:5])
    assert r.ok, r.message
    return np.r_[r.q, 80.0]


def test_the_camera_underneath_a_low_gripper_is_rejected_and_named(rd):
    """The agent's successful table grasp (roll +90, pitch -35, tip 22 mm) had the wrist camera ~5 mm above
    the table in the model - with the arm's ~1 cm sag, at or into it. Same grasp at roll -90: camera on top."""
    rd.supervisor.mode = "none"
    q = _grasp_q(rd, 90)
    out = rd.move_joints("right", dict(zip(J, q[:5])), speed=1.0)
    assert out["status"] == "rejected", out
    v = [x for x in out["violations"] if x["rule"] == "camera_clearance"]
    assert v and "right wrist camera" in v[0]["message"] and "the floor" in v[0]["message"]
    q2 = _grasp_q(rd, -90)
    out2 = rd.move_joints("right", dict(zip(J, q2[:5])), speed=1.0)
    assert out2["status"] != "rejected", out2


def test_a_part_inside_its_margin_may_leave_but_not_go_closer(tmp_path):
    import numpy as np
    low = None
    r = make(tmp_path)
    try:
        low = _grasp_q(r, 90, z=0.030)                     # camera inside its 15 mm floor margin, not touching
        hc_d = r.compute.local.full_check({"right": Path(np.array([0.0]), low[None, :])},
                                          {a: r._start_q(a) for a in r.arms})
        assert any(x.rule == "camera_clearance" for x in hc_d)
    finally:
        r.stop()
    r = make(tmp_path / "b")
    try:
        r.arm_ios["right"].q[:] = low
        r.arm_ios["right"].goal[:] = low
        time.sleep(0.2)
        r.arms["right"].q_cmd = None
        r.supervisor.mode = "none"
        up = r.move_joints("right", {"shoulder_lift": float(low[1]) - 15}, speed=0.5)    # lifts away
        assert up["status"] != "rejected", up
        done(r, up)
        r.arm_ios["right"].q[:] = low
        r.arm_ios["right"].goal[:] = low
        time.sleep(0.2)
        r.arms["right"].q_cmd = None
        down = r.move_joints("right", {"shoulder_lift": float(low[1]) + 6}, speed=0.5)   # gets closer
        assert down["status"] == "rejected" and any(x["rule"] == "camera_clearance" for x in down["violations"]), down
    finally:
        r.stop()


def test_link_clearances_are_reported_never_enforced(rd):
    """Task 4b: wrist/forearm/upper-arm hulls are measured and returned for the log, never as violations."""
    import numpy as np
    q = np.array([0.0, 35.0, 50.0, 0.0, -90.0, 2.0])          # wrist and gripper parts down on the table
    P = rd.compute.local
    v, report = P.full_check_report({"right": Path(np.array([0.0]), q[None, :])}, {a: rd._start_q(a) for a in rd.arms})
    assert any(k[1] == "wrist" for k in report), report
    assert not any(x.rule.startswith(("wrist", "forearm", "upper_arm")) for x in v)
    assert any(x.rule in ("camera_clearance", "jaws_clearance", "gripper_clearance") for x in v)


def test_jaw_decomposition_covers_the_jaws_and_is_used(rd):
    """Task 3: the jaws' convex pieces (CoACD) cover every jaw vertex, the checker collides them, and against
    the table they give the single hull's distance (a hull's lowest point is the part's) - which also catches
    MuJoCo's mid-phase silently dropping pieces (found 2026-10-05)."""
    import json
    import sys
    import trimesh
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[2] / "tools"))
    from build_jaw_collision import outside_distance
    from stack.twin_mujoco.scene import assets_dir
    from stack.robotd.compute import _hull_checker
    from stack.robotd.hulls import HullChecker
    data = json.loads((P(__file__).resolve().parents[1] / "assets" / "gripper_collision" / "so101_jaws.json").read_text())
    for name, part in data["parts"].items():
        mesh = trimesh.load(assets_dir() / "assets" / f"{name}.stl")
        assert outside_distance(mesh.vertices, part["hulls"]).max() < 1e-6, name
        assert part["volume_ratio"]["pieces_over_mesh"] < 1.5
    hc, _ = _hull_checker(rd.cfg)
    assert sum(1 for g, p in hc.geom_part.items() if p == "jaws" and hc.geom_arm[g] == "right") == 64
    q = _grasp_q(rd, -90)
    cur = {"left": rd._start_q("left"), "right": q}
    pieces = min(v for k, v in hc.distances(cur).items() if k[0] == "right" and k[1] == "jaws" and k[2] == "floor")
    from stack.twin_mujoco import scene as S
    orig = S.load_jaw_hulls
    S.load_jaw_hulls = lambda path=None: None
    try:
        one = HullChecker(rd.cfg.rig, rd.cfg.safety)
    finally:
        S.load_jaw_hulls = orig
    hull = min(v for k, v in one.distances(cur).items() if k[0] == "right" and k[1] == "jaws" and k[2] == "floor")
    assert abs(pieces - hull) < 0.0005, (pieces, hull)


def test_hulls_one_floor_missing_arms_left_out_and_a_rest_may_dip(rd):
    """Review 2026-10-05: the table and the board top are one target ("the floor"), an arm with no reading is left
    out rather than kept at a stale pose, and a rest path starting inside a margin may dip REST_DIP_M first."""
    from stack.robotd.compute import _hull_checker
    from stack.robotd.hulls import REST_DIP_M
    hc, _ = _hull_checker(rd.cfg)
    q = _grasp_q(rd, 90, z=0.030)
    d = hc.distances({"left": rd._start_q("left"), "right": q})
    floor = [k for k in d if k[2] == "floor"]
    assert floor and all(k[3] == "the floor" for k in floor)
    # the left arm parked against the right one: present it is seen, missing it is not
    alone = hc.distances({"right": q})
    assert not any(k[2] == "other_arm" for k in alone) and not any(k[0] == "left" for k in alone)
    # a path that dips 1 mm before lifting: rejected as a motion, allowed as a rest
    cam = min(v for k, v in d.items() if k[:3] == ("right", "camera", "floor"))
    m = rd.models["right"]
    lo, hi = m.limits.usable(rd.cfg.safety)
    dip = None
    for dl in np.arange(0.2, 6.0, 0.2):
        q2 = q.copy(); q2[1] += dl
        c2 = min(v for k, v in hc.distances({"left": rd._start_q("left"), "right": q2}).items()
                 if k[:3] == ("right", "camera", "floor"))
        if 0.0005 < cam - c2 < REST_DIP_M - 0.0003:
            dip = q2
            break
    assert dip is not None
    up = q.copy(); up[1] -= 25
    path = Path(np.array([0.0, 0.5, 1.0]), np.stack([q, dip, up]))
    cur = {"left": rd._start_q("left"), "right": q}
    v, _ = hc.check({"right": path}, cur)
    assert any(x.rule == "camera_clearance" for x in v)
    v, _ = hc.check({"right": path}, cur, rest=True)
    assert not any(x.rule == "camera_clearance" for x in v), v


def test_motions_approved_together_are_checked_together(tmp_path):
    """Review 2026-10-05: two pending motions on different arms, each planned with the other arm still, start
    together at ARM; their combined motion is checked (hulls included) and the later one is rejected if they meet."""
    r = make(tmp_path, approval_mode="every_motion")
    try:
        # both grippers to the same spot in front of the board centre
        outs = []
        for arm in ("left", "right"):
            outs.append(r.move_pose(arm, [0.25, 0.0, 0.08], pitch=-60.0, speed=1.0))
        assert all(o["status"] == "pending_approval" for o in outs), outs
        r.approve()
        st = [r.motion_status(o["motion_id"], wait_s=0.0) for o in outs]
        assert st[0]["status"] != "rejected", st[0]
        assert st[1]["status"] == "rejected" and "approved together with" in st[1]["reason"], st[1]
    finally:
        r.stop()


# ---- follow-up review 2026-10-05 ------------------------------------------------------------------------------

_CAMS_MEET = (np.array([16.5, 22.6, -19.8, -47.1, -130.3, 20.0]),     # found by search: link capsules clear,
              np.array([-41.2, 25.3, -20.0, -57.8, 49.6, 20.0]))      # the two wrist cameras ~7 mm apart


def test_the_launch_recheck_includes_the_gripper_hulls(rd):
    """Capsules alone keep the wrist axes ~90 mm apart; the cameras stick out 73 mm each. The launch-time
    arm-to-arm check must see them. (The cameras are ~7 mm apart: the 20 mm camera margin of the time is set here,
    since robotd.yaml has 5 mm since 2026-10-08.)"""
    from stack.robotd.safety import check_clearance
    m = rd.cfg.safety.hull_margins_m
    rd.cfg.safety.hull_margins_m = {**m, "other_arm": {**m["other_arm"], "camera": 0.020}}
    ql, qr = _CAMS_MEET
    P = rd.compute.local
    assert P.cfg.safety.hull_margins_m["other_arm"]["camera"] == 0.020
    paths = {"right": Path(np.array([0.0]), qr[None, :])}
    cur = {"left": ql, "right": qr}
    assert not check_clearance(P.models, rd.cfg.safety, paths, cur, {}, 0.0)        # the old launch check passed
    v = P.clearance(paths, cur, {}, 0.1)
    assert v and v[0]["rule"] == "camera_clearance" and "arm's camera" in v[0]["message"], v
    assert "minimum 20 mm" in v[0]["message"], v


def test_a_motion_needs_a_reading_of_both_arms_a_query_does_not(rd):
    from stack.robotd.compute import Rejected
    q = _grasp_q(rd, -90)
    P = rd.compute.local
    path = {"right": Path(np.array([0.0]), q[None, :])}
    with pytest.raises(Rejected, match="no reading for the left arm"):
        P.full_check(path, {"left": None, "right": q})
    assert not P.full_check(path, {"left": None, "right": q}, query=True)


def test_a_full_contact_buffer_fails_closed(rd, monkeypatch):
    import mujoco
    from stack.robotd.compute import _hull_checker
    hc, _ = _hull_checker(rd.cfg)
    real = hc._set

    def overflowing(q):
        real(q)
        hc.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number = 1
    monkeypatch.setattr(hc, "_set", overflowing)
    q = _grasp_q(rd, -90)
    v, _ = hc.check({"right": Path(np.array([0.0]), q[None, :])}, {"left": rd._start_q("left"), "right": q})
    assert [x.rule for x in v] == ["hull_check_incomplete"]
    assert [x.rule for x in hc.arm_check({"right": Path(np.array([0.0]), q[None, :])},
                                         {"left": rd._start_q("left"), "right": q})] == ["hull_check_incomplete"]


def test_every_gripper_part_lies_within_grip_reach_of_its_segment(rd):
    """The launch hull check skips samples where the wrist+gripper segments are > 2 * GRIP_REACH_M + margin apart."""
    from stack.robotd.compute import _hull_checker
    from stack.robotd.hulls import GRIP_REACH_M, OTHER_ARM_ENFORCED
    from stack.robotd.safety import _segments, segment_distance
    hc, _ = _hull_checker(rd.cfg)
    m, d = hc.model, hc.data
    geoms = [g for g, p in hc.geom_part.items() if p in OTHER_ARM_ENFORCED and hc.geom_arm[g] == "right"]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        q = np.r_[rng.uniform(-90, 90, 4), rng.uniform(-180, 180), rng.uniform(0, 100)]
        hc._set({"right": q})
        s = _segments(rd.models["right"], q[None, :])[0, 3]
        for g in geoms:
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            V = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            worst = max(worst, float(segment_distance(V, V, s[0][None], s[1][None]).max()))
    assert 0.09 < worst < GRIP_REACH_M - 0.005, worst


# ---- sprint 2026-10-08: the gripper ends are kept apart by the hulls, not the capsules -------------------------

# Episode e20261008-075921, program p20261008-120239-4321: the right arm holds a backstop low in front of the cup
# (pitch -20), the left arm comes in beside it. The wrist+gripper capsules came within 33 mm (40 required) while the
# real parts kept 28.5 mm (the left jaws to the right wrist).
_BACKSTOP = (np.array([7.9, -26.5, 81.1, -29.6, -90.0, 98.0]),       # left, at its target
             np.array([-44.0, 62.3, -22.9, -7.2, -90.0, 2.0]))       # right, holding
# Found by search: the two wrist bodies 2 mm apart, every camera, jaw and gripper 37+ mm from the other arm. (The
# remaining capsules refuse it too - a forearm is near - so it tests the hull layer's own wrist rule.)
_WRISTS_MEET = (np.array([44.0, 6.8, -4.5, 99.3, 83.7, 31.3]),
                np.array([-45.8, 22.1, -17.2, -57.6, 124.4, 46.9]))
# Found by the reviewer: the gripper ends overlap, and no remaining capsule pair is near - only the hulls can refuse.
_GRIPPERS_CROSS = (np.array([45.6, 47.4, -29.1, 36.7, 87.3, 78.1]),
                   np.array([-19.3, 0.7, 83.8, -55.6, 80.9, 54.5]))


def _still_paths(ql, qr):
    return {"left": Path(np.array([0.0]), ql[None, :])}, {"left": ql, "right": qr}


def test_the_gripper_end_capsules_no_longer_block_what_the_hulls_clear(rd):
    from stack.robotd.safety import check_paths
    P = rd.compute.local
    paths, cur = _still_paths(*_BACKSTOP)
    assert "arm_clearance" in {x.rule for x in check_paths(P.models, rd.cfg.safety, paths, cur)}   # capsules alone
    assert P.full_check(paths, cur) == []                                   # with the hulls: clear (28.5 mm)
    assert P.clearance(paths, cur, {}, 0.1) == []                           # and the launch check agrees


def test_without_hull_checks_the_capsules_are_unchanged(rd, monkeypatch):
    P = rd.compute.local
    monkeypatch.setattr(rd.cfg.safety, "hull_checks", False)
    assert not P._hulls_on()
    paths, cur = _still_paths(*_BACKSTOP)
    assert [x.rule for x in P.full_check(paths, cur)] == ["arm_clearance"]
    assert [v["rule"] for v in P.clearance(paths, cur, {}, 0.1)] == ["arm_clearance"]


def test_the_wrist_is_enforced_against_the_other_arm(rd):
    """With the wrist+gripper capsule pair gone, the wrist body must be checked by the hulls: two wrists 2 mm
    apart, with no camera, jaw or gripper near, are refused by the hull layer for the wrist alone."""
    from stack.robotd.compute import _hull_checker
    P = rd.compute.local
    paths, cur = _still_paths(*_WRISTS_MEET)
    hc, lock = _hull_checker(rd.cfg)
    with lock:
        v, _ = hc.check(paths, cur)
    assert v and {x.rule for x in v} == {"wrist_clearance"}, v
    assert " wrist would be " in v[0].message and "arm's " in v[0].message, v[0].message
    assert "wrist_clearance" in {x.rule for x in P.full_check(paths, cur)}


def test_where_only_the_hulls_see_it_the_launch_check_refuses(rd):
    """The remaining capsule pairs are clear here; the launch re-check must still refuse, through the hulls
    (wrist included)."""
    from stack.robotd.safety import check_clearance
    P = rd.compute.local
    paths, cur = _still_paths(*_GRIPPERS_CROSS)
    assert check_clearance(P.models, rd.cfg.safety, paths, cur, {}, 0.0, hulls=True) == []
    assert check_clearance(P.models, rd.cfg.safety, paths, cur, {}, 0.0)          # the old pair saw it
    rules = {v["rule"] for v in P.clearance(paths, cur, {}, 0.1)}
    assert "jaws_clearance" in rules and rules <= {"camera_clearance", "jaws_clearance", "gripper_clearance",
                                                    "wrist_clearance"}, rules     # the hulls alone refuse it


def test_the_wrist_is_not_enforced_against_the_floor_or_itself(rd):
    from stack.robotd.hulls import ENFORCED, OTHER_ARM_ENFORCED
    assert "wrist" not in ENFORCED and set(OTHER_ARM_ENFORCED) == set(ENFORCED) | {"wrist"}
    assert set(rd.cfg.safety.hull_margins_m["other_arm"]) == {"camera", "jaws", "gripper", "wrist"}
    assert "wrist" not in rd.cfg.safety.hull_margins_m["floor"]


def _mixed_margins(rd, jaws=0.005, rest=0.020):
    """A hull checker whose other-arm margins differ per part (jaws 5 mm, the rest 20 mm, as on 2026-10-08), so the
    pair rule is tested whatever robotd.yaml sets."""
    import copy
    from stack.robotd.compute import _hull_checker
    cfg = copy.copy(rd.cfg)
    cfg.safety = copy.deepcopy(rd.cfg.safety)
    cfg.safety.hull_margins_m["other_arm"] = {"camera": rest, "jaws": jaws, "gripper": rest, "wrist": rest}
    return _hull_checker(cfg)


def test_robotd_yaml_sets_every_other_arm_margin_to_5_mm(rd):
    assert rd.cfg.safety.hull_margins_m["other_arm"] == {"camera": 0.005, "jaws": 0.005, "gripper": 0.005,
                                                          "wrist": 0.005}          # user, 2026-10-08


def test_a_pair_between_the_arms_takes_the_larger_of_its_two_parts_margins(rd):
    """Against the other arm a pair takes the larger of its two parts' margins, and a report-only part (forearm,
    upper arm) counts as the largest: with jaws 5 mm and the rest 20 mm, only jaws against jaws gets 5 mm."""
    hc, lock = _mixed_margins(rd)
    grid = np.array([0.0])
    jaws_jaws = {("left", "jaws", "other_arm", "the right arm's jaws"): np.array([0.008]),
                 ("right", "jaws", "other_arm", "the left arm's jaws"): np.array([0.008])}
    jaws_camera = {("left", "jaws", "other_arm", "the right arm's camera"): np.array([0.008]),
                   ("right", "camera", "other_arm", "the left arm's jaws"): np.array([0.008])}
    with lock:
        assert hc._judge(jaws_jaws, grid, False)[0] == []
        v, _ = hc._judge(jaws_camera, grid, False)
        too_close = {("left", "jaws", "other_arm", "the right arm's jaws"): np.array([0.004])}
        v2, _ = hc._judge(too_close, grid, False)
    assert {x.rule for x in v} == {"camera_clearance", "jaws_clearance"}, v        # both sides, each at 20 mm
    assert all("minimum 20 mm" in x.message for x in v), v
    assert [x.rule for x in v2] == ["jaws_clearance"] and "minimum 5 mm" in v2[0].message, v2
    # the forearm and upper arm are report-only, so their side never refuses; the jaws' side must still use 20 mm
    for link in ("forearm", "upper arm"):
        with lock:
            v3, _ = hc._judge({("left", "jaws", "other_arm", f"the right arm's {link}"): np.array([0.008])}, grid, False)
        assert [x.rule for x in v3] == ["jaws_clearance"] and "minimum 20 mm" in v3[0].message, (link, v3)


# Found by the reviewer of the 5 mm jaws margin: the right jaws ~15 mm from the left forearm, the capsules ~73 mm apart.
_JAWS_AT_FOREARM = (np.array([9.7, 50.5, -2.2, -89.8, 153.9, 93.6]),
                    np.array([-48.3, 44.8, -5.0, -75.1, 105.0, 81.7]))


def test_the_jaws_to_forearm_pair_follows_the_larger_margin_end_to_end(rd):
    from stack.robotd.safety import check_paths
    P = rd.compute.local
    paths, cur = _still_paths(*_JAWS_AT_FOREARM)
    capsules = {x.rule for x in check_paths(P.models, rd.cfg.safety, paths, cur, hulls=True)}
    assert "arm_clearance" not in capsules, capsules                                # the capsules do not see it
    hc, lock = _mixed_margins(rd)                                                   # jaws 5 mm, the rest 20 mm
    with lock:
        d = hc.distances(cur)
        v, _ = hc.check(paths, cur)
    near = {k: x for k, x in d.items() if k[0] == "right" and k[1] == "jaws" and "left arm's" in k[3]}
    assert near and 0.005 < min(near.values()) < 0.020, near
    assert any(x.rule == "jaws_clearance" and "minimum 20 mm" in x.message for x in v), v
    hc5, lock5 = _mixed_margins(rd, rest=0.005)                                     # all 5 mm: robotd.yaml today
    with lock5:
        assert hc5.check(paths, cur)[0] == []


def test_look_at_prefers_the_camera_on_top_then_beside_then_underneath(rd):
    """D9 amendment (2026-10-05/06): roll -90 first (camera on top), then 0 and +90 where -90 has no clear
    viewpoint. Over table targets they find far more viewpoints than -90 alone."""
    P = rd.compute.local
    cur = {a: rd._start_q(a) for a in rd.arms}
    pts = [[x, y, 0.02] for x in (0.15, 0.25) for y in (-0.15, -0.05, 0.05, 0.15)]
    only_top = sum(P.view_pose("right", p, cur, rolls=(-90.0,))["ok"] for p in pts)
    res = [P.view_pose("right", p, cur) for p in pts]
    assert sum(v["ok"] for v in res) >= only_top + 2, (only_top, [v["ok"] for v in res])
    rolls = {v["roll"] for v in res if v["ok"]}
    assert -90.0 in rolls and rolls - {-90.0}, rolls
    n = len(P.VIEW_ELEVATIONS)
    for v in res:                                    # in order: every elevation at -90, then at 0, then at +90
        if v["ok"]:
            assert v["candidate"] // n == P.VIEW_ROLLS.index(v["roll"]) and v["axis_miss_mm"] < 2.0


def test_the_generated_gripper_section_of_embodiment_md_is_up_to_date(rd):
    """Sprint 2026-10-05 task 6: the gripper/camera words the agent reads come from the model
    (tools/gen_embodiment.py); a geometry or rule change without regenerating fails here."""
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[2] / "tools"))
    import gen_embodiment as G
    f = G.facts(rd.compute.local)
    assert f["tip_rule_mm"] == round(rd.cfg.safety.tip_clearance_m * 1000)   # the rule, not a height the jaw
                                                                           # hulls set (that is the jaw depth)
    assert f["rolls"][0.0]["camera"] == "on the arm's left"          # confirmed at the rig 2026-10-05
    assert f["rolls"][0.0]["moving_jaw"] == "on top"                 # ditto: the fixed jaw underneath at roll 0
    assert f["rolls"][-90.0]["camera"] == "on top" and f["rolls"][90.0]["camera"] == "underneath"
    assert f["rolls"][-90.0]["moving_jaw"] == "on the arm's right"
    g = f["geometry"]                                   # sprint 2026-10-07: C2, C3, C5 checked against the model
    assert 80 < g["across_heading_deg"] <= 90                       # roll -90: jaws close across the heading
    assert 0.85 < g["deg_per_roll_deg"] < 1.0                       # + roll turns them CCW from above, ~1:1
    assert max(abs(v) for v in g["image_formula_err_deg"].values()) < 15 and abs(g["image_formula_err_deg"][-90.0]) < 6
    assert 160 < g["turn_over_180_deg"] < 200
    assert 20 < g["camera_tilt_deg"] < 30 and 60 < g["camera_off_axis_mm"] < 85
    doc = G.DOC.read_text()
    assert G.render(doc, G.section(f)) == doc, "run tools/gen_embodiment.py"
    # C1: the tip's jaw and the offset stay withheld (rung 2) in everything the agent reads
    import re
    from pathlib import Path as P_
    root = P_(__file__).resolve().parents[2]
    for f_ in (G.DOC, root / "agent_ws" / "AGENTS.md", root / "lib" / "robot_use" / "README.md"):
        t = f_.read_text().lower()
        assert "between the jaw tips" not in t and "jaw mouth" not in t, f_
        assert not re.search(r"3\.5 cm|35 mm|23\.6|0\.035|tip (is|sits) (at|on) the fixed jaw", t), f_


def test_the_rest_roll_setting_and_the_folded_rest_at_minus_90(rd):
    """Task 6b (model part): rest_roll_deg -90 folds the arm with the camera on top; that rest is clear of the
    table, board and the other arm by the hull margins, and keeps the camera and jaws further from the arm's own
    shoulder than the roll-0 rest does (robotd does not check an arm against itself)."""
    import dataclasses
    import mujoco
    from stack.robotd.compute import _hull_checker
    from stack.robotd.safety import rest_pose
    cfg0 = dataclasses.replace(rd.cfg.safety, rest_roll_deg=0.0)
    cfg90 = dataclasses.replace(rd.cfg.safety, rest_roll_deg=-90.0)
    hc, _ = _hull_checker(rd.cfg)
    m, d = hc.model, hc.data

    def own_min(q):
        hc._set(q)
        out = {}
        for arm in q:
            mine = [g for g, p in hc.geom_part.items() if hc.geom_arm[g] == arm and p in ("camera", "jaws")]
            own = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                   and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) == f"{arm}_shoulder"]
            out[arm] = min(mujoco.mj_geomDistance(m, d, g, o, 0.1, None) for g in mine for o in own)
        return out
    r0 = {a: rest_pose(mm, cfg0) for a, mm in rd.models.items()}
    r90 = {a: rest_pose(mm, cfg90) for a, mm in rd.models.items()}
    assert all(q[4] == -90.0 and np.allclose(q[[0, 1, 2, 3, 5]], r0[a][[0, 1, 2, 3, 5]]) for a, q in r90.items())
    P = rd.compute.local
    assert not P.full_check({a: Path(np.array([0.0]), q[None, :]) for a, q in r90.items()}, r90)
    dist = hc.distances(r90)
    assert not [k for k in dist if k[2] != "own_arm"]       # nothing within the detection margin of the world
    assert min(v for k, v in dist.items() if k[2] == "own_arm") >= 0.005   # and the own-arm margin kept
    s0, s90 = own_min(r0), own_min(r90)
    assert all(s90[a] > s0[a] + 0.003 and s90[a] > 0.005 for a in s0), (s0, s90)


def test_the_rest_motion_never_turns_the_wrist_while_folded(rd):
    """Task 6b: folded at roll +90 the camera is inside the shoulder, and rolling in place grazes it. A rest motion
    that changes the roll goes out to the pre-rest pose, turns there, then folds; along the way the camera and jaws
    keep clear of the arm's own shoulder, base and upper arm (which robotd does not check)."""
    import mujoco
    from stack.robotd.compute import PRE_REST_DELTA, _hull_checker
    P = rd.compute.local
    hc, _ = _hull_checker(rd.cfg)
    m, d = hc.model, hc.data
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    assert rd.cfg.safety.rest_roll_deg == -90.0                  # robotd.yaml (task 6b)
    for arm in ("left", "right"):
        rest = P.rest[arm]
        own = [g for g in range(m.ngeom) if m.geom_group[g] == 3 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and body(g) in (f"{arm}_shoulder", f"{arm}_base", f"{arm}_upper_arm")]
        mine = [g for g, p in hc.geom_part.items() if hc.geom_arm[g] == arm and p in ("camera", "jaws", "gripper")]
        y = 0.12 if arm == "left" else -0.12
        for start in (_grasp_q_for(rd, arm, (0.18, y, 0.15), -45, 90), _grasp_q_for(rd, arm, (0.18, y, 0.15), -45, 0)):
            p = P.rest_path(start, rest)
            qs = np.array([p.at(t) for t in np.linspace(0, p.duration, 80)])
            pre = rest + PRE_REST_DELTA
            # the roll changes only while lift and elbow are at the pre-rest pose (out of the fold)
            turning = np.abs(np.diff(p.q[:, 4])) > 1e-6          # between consecutive path knots
            assert turning.any()
            for k in np.nonzero(turning)[0]:
                assert np.allclose(p.q[[k, k + 1], 1:3], pre[1:3], atol=1e-6), "the wrist turned outside the pre-rest pose"
            closest = 1.0
            for q in qs[::4]:
                hc._set({**{a: P.rest[a] for a in P.models}, arm: q})
                closest = min(closest, min(mujoco.mj_geomDistance(m, d, g, o, 0.1, None) for g in mine for o in own))
            assert closest > 0.005, (arm, closest)
            assert np.allclose(p.at(p.duration), rest)
    # no roll change: the old, direct fold (unchanged)
    s = P.rest["right"].copy(); s[1] += 20
    assert not np.any(np.abs(np.diff(np.array([P.rest_path(s, P.rest["right"]).at(t) for t in np.linspace(0, 1, 20)])[:, 4])) > 1e-9)


def _grasp_q_for(rd, arm, xyz, pitch, roll):
    m = rd.models[arm]
    lo, hi = m.limits.usable(rd.cfg.safety)
    r = m.kin.solve_ik(list(xyz), pitch, roll, seed_deg=np.array([0, 10, 20, 50, roll], float),
                       lower_deg=lo[:5], upper_deg=hi[:5])
    assert r.ok, r.message
    return np.r_[r.q, 30.0]


# ---- review 2026-10-06 (tasks 5-6b) and task 12 (own-arm check, pulled in) ------------------------------------

def test_the_gripper_end_is_checked_against_its_own_arm(rd):
    """Folded at roll +90 the camera is inside the shoulder; robotd refuses it, naming the part. The -90 rest and
    the paths to it from the old roll-0 rest pass (start inside a margin may leave, never get closer)."""
    P = rd.compute.local
    rest = {a: P.rest[a].copy() for a in P.models}
    q = rest["right"].copy()
    q[4] = 90.0
    v = P.full_check({"right": Path(np.array([0.0]), q[None, :])}, {**rest, "right": q})
    assert any(x.rule == "camera_clearance" and "its own shoulder" in x.message for x in v), [x.message for x in v]
    assert not P.full_check({a: Path(np.array([0.0]), r[None, :]) for a, r in rest.items()}, rest)
    for arm in ("left", "right"):
        r0 = rest[arm].copy()
        r0[4] = 0.0
        assert not P.full_check({arm: P.rest_path(r0, rest[arm])}, {**rest, arm: r0}, rest=True)


def test_no_path_turns_the_wrist_while_folded(rd):
    """Moves out of the folded rest that change the roll go out to the pre-rest pose first; anything else that
    would turn the wrist while folded (a trajectory) is refused."""
    from stack.robotd.compute import PRE_REST_DELTA, Rejected
    P = rd.compute.local
    rest = {a: P.rest[a].copy() for a in P.models}
    paths, _ = P.plan(("joints", "right", {"wrist_roll": 0.0}, 0.5), rest, {})
    p = paths["right"]
    pre = rest["right"] + PRE_REST_DELTA
    for k in np.nonzero(np.abs(np.diff(p.q[:, 4])) > 1e-6)[0]:
        assert np.allclose(p.q[[k, k + 1], 1:3], pre[1:3], atol=1e-6)
    assert not P._folded_roll_violations(paths)
    with pytest.raises(Rejected, match="while the arm is folded"):
        P.plan(("trajectory", "right", [{"t": 4.0, "joints": {"wrist_roll": 0.0}}], "reject"), rest, {})


def test_every_viewpoint_and_the_move_there_is_clear_of_the_arm_itself(rd):
    """Review 2026-10-06: a +90 viewpoint had the camera 16 mm inside the shoulder, and look_at's move there could
    be refused (a failed step). view_pose now checks the move from where the arm is, own arm included."""
    from stack.robotd.compute import _hull_checker
    P = rd.compute.local
    hc, _ = _hull_checker(rd.cfg)
    cur = {a: P.rest[a].copy() for a in P.models}
    for arm, ys in (("right", (-0.20, -0.15, -0.05)), ("left", (0.20, 0.15, 0.05))):
        for x in (0.15, 0.22):
            for y in ys:
                v = P.view_pose(arm, [x, y, 0.02], cur)
                if not v["ok"]:
                    continue
                q = np.array([v["joints"][j] for j in J])
                assert hc.own_clearance(arm, {**cur, arm: q})[0] >= 0.005, (arm, x, y, v["roll"])
                paths, _ = P.build(("pose", arm, v["xyz"], v["pitch"], v["roll"], False, P.VIEW_SPEED), cur)
                assert not P.full_check(paths, cur), (arm, x, y)
    bad = P.view_pose("right", [0.2, -0.1, 0.02], cur, start=99)
    assert not bad["ok"] and "start must be" in bad["reason"]
    assert not rd.view_pose("right", [0.2, -0.1, 0.02], elevations=[90])["ok"]


def test_rest_routes_and_startup_checks(rd):
    """Re-review 2026-10-06: robotd refuses to start with a rest pose that fails the checks; a rest motion whose
    way out to the pre-rest pose is refused turns the wrist where the arm is instead; the pre-rest pose is outside
    the fold zone; a move whose middle (not its ends) is folded is routed too."""
    import dataclasses
    from stack.robotd.compute import PRE_REST_DELTA
    P = rd.compute.local
    assert P.check_rest_poses() == []
    for a in P.models:
        assert P._folded(a, P.rest[a]) and not P._folded(a, P.rest[a] + PRE_REST_DELTA)
    # the reviewer's stranded pose: clear, but the straight way out to the pre-rest pose sweeps the shoulder at +90
    start = {a: P.rest[a].copy() for a in P.models}
    start["left"] = np.array([-75.0, -58.0, 79.0, 32.0, 90.0, 76.0])
    assert P.full_check({"left": P.rest_path(start["left"], P.rest["left"])}, start, rest=True), \
        "the default route should fail here (else this test no longer covers the alternative)"
    paths, _ = P.build(("rest", ["left"]), start)
    assert not P.full_check(paths, start, rest=True), "go_rest must find a route home"
    # a move through the fold (neither end folded) is routed, not refused
    r = P.rest["right"]
    s = r.copy(); s[1] += 30; s[2] -= 10; s[4] = -90.0
    t = r.copy(); t[1] += 5; t[2] -= 45; t[4] = 0.0
    p = P._joint_move("right", s, t, 0.5)
    assert not P._folded_roll_violations({"right": p})
    # robotd will not start with a rest that fails (here: a huge own-arm margin)
    cfg = dataclasses.replace(rd.cfg, safety=dataclasses.replace(
        rd.cfg.safety, hull_margins_m={**rd.cfg.safety.hull_margins_m,
                                       "own_arm": {"camera": 0.03, "jaws": 0.03, "gripper": 0.03}}))
    from stack.robotd.daemon import RobotD
    from stack.robotd.hardware import MockArmIO
    lims = {a: P.models[a].limits for a in P.models}
    with pytest.raises(RuntimeError, match="rest pose fails"):
        RobotD(dataclasses.replace(cfg, motion_enabled=True), {a: MockArmIO(a) for a in cfg.arms}, {}, limits=lims)
    RobotD(dataclasses.replace(cfg, motion_enabled=False), {a: MockArmIO(a) for a in cfg.arms}, {},
           limits=lims).compute.close()                 # without motion: starts (with a warning)


# ---- sprint 2026-10-07: gripper-only moves inside a margin; rest / as-it-sits warnings --------------------------

def test_a_gripper_only_move_inside_a_margin_passes_but_nothing_may_get_closer(rd):
    """Setup v2 trial: open() on the torque-off left arm, settled 4.5 mm from its own shoulder, was refused for
    "ending" inside the 5 mm margin although it moved the arm nowhere."""
    from stack.robotd.compute import _hull_checker
    P = rd.compute.local
    hc, _ = _hull_checker(rd.cfg)
    rest = {a: P.rest[a].copy() for a in P.models}
    q = rest["left"].copy()
    q[1] += 2.5
    q[2] += 2.5                                    # settled: the jaws ~3 mm from the shoulder (inside 5 mm)
    cur = {**rest, "left": q}
    assert hc.own_clearance("left", cur)[0] < 0.005
    opened = q.copy()
    opened[5] = 90.0
    grip = Path(np.array([0.0, 1.0]), np.stack([q, opened]))
    assert not P.full_check({"left": grip}, cur), "a gripper-only move at the same pose must pass"
    # an arm move that ends inside the margin is still refused (it starts outside, at the rest)
    into = Path(np.array([0.0, 1.0]), np.stack([rest["left"], q]))
    assert any(x.rule == "jaws_clearance" for x in P.full_check({"left": into}, rest))
    # and a gripper-only move whose part gets closer is still refused (synthetic: shrink the start distance)
    v, _ = hc.check({"left": grip}, cur)
    assert not v
    series = {("left", "jaws", "own_arm", "its own shoulder"): np.array([0.004, 0.003])}
    bad, _ = hc._judge(series, np.array([0.0, 1.0]), rest=False, moving=set())
    assert bad and bad[0].rule == "jaws_clearance"


def test_rest_and_as_it_sits_warnings(rd):
    """The thin left-arm rest margin is reported as a warning (not a refusal); an arm sitting inside a margin
    is reported until it moves."""
    import time
    P = rd.compute.local
    assert P.check_rest_poses() == []
    w = P.rest_pose_warnings()
    assert any(x.startswith("left rest settled") and "own shoulder" in x for x in w), w
    assert rd.state()["warnings"] == [f"rest pose: {x}" for x in rd.rest_warnings]
    q = P.rest["left"].copy()
    q[1] += 2.5
    q[2] += 2.5
    assert P.pose_warnings("left", {"left": q, "right": P.rest["right"]})
    rd.arm_ios["left"].q[:] = q                    # the mock arm sits there, torque off
    rd.torque_off("left")
    t0 = time.time()
    while time.time() - t0 < 5 and not rd.state()["arms"]["left"]["pose_warnings"]:
        time.sleep(0.05)
    assert any("own shoulder" in x for x in rd.state()["arms"]["left"]["pose_warnings"])


def test_gripper_only_relaxation_edges(rd):
    """Re-review 2026-10-07: a still arm gets no rest dip; the launch re-check relaxes like planning; NaN moves."""
    from stack.robotd.compute import _hull_checker
    from stack.robotd.hulls import _moving
    hc, _ = _hull_checker(rd.cfg)
    key = ("left", "jaws", "own_arm", "its own shoulder")
    grid = np.array([0.0, 0.5, 1.0])
    v, _ = hc._judge({key: np.array([0.0045, 0.0030, 0.0026])}, grid, rest=True, moving=set())
    assert v, "a still arm's rest motion must not dip 2 mm closer and stay there"
    v, _ = hc._judge({key: np.array([0.0045, 0.0045, 0.0045])}, grid, rest=True, moving=set())
    assert not v
    still = Path(np.array([0.0, 1.0]), np.array([[0, 0, 0, 0, 0, 2.0], [0, 0, 0, 0, 0, 90.0]]))
    nan = Path(np.array([0.0, 1.0]), np.array([[0, 0, 0, 0, 0, 2.0], [np.nan, 0, 0, 0, 0, 2.0]]))
    assert _moving({"left": still}) == set() and _moving({"left": nan}) == {"left"}
