import numpy as np
import pytest

from stack.robotd.planner import plan_joint_move
from stack.robotd.safety import Path, check_paths, load_arm_limits, rest_pose


def _rules(v):
    return {x.rule for x in v}


def _still(q):
    return Path(np.array([0.0]), np.asarray(q, float)[None, :])


@pytest.fixture
def home(models, cfg):
    return {a: rest_pose(m, cfg) for a, m in models.items()}


def test_rest_pose_is_derived_and_valid(models, cfg, home):
    for a, m in models.items():
        q = home[a]
        assert q[0] == 0 and q[4] == 0 and q[5] == cfg.rest_gripper == 2.0
        assert abs(q[1] - (m.limits.hard["shoulder_lift"][0] + 4)) < 1e-9
        assert abs(q[2] - (m.limits.hard["elbow_flex"][1] - 4)) < 1e-9
        assert q[3] <= 70, "rest prefers the recording plan's 70 deg wrist, less only if the tip would hit the board"
        assert check_paths({a: m}, cfg, {a: _still(q)}, {a: q}, rest=True) == []
        # ...and it is inside the 5 deg margin, so only go_rest may move INTO it (holding there is fine)
        from stack.robotd.planner import plan_joint_move
        q_mid = q.copy(); q_mid[:4] = [0, -60, 60, 40]
        into = plan_joint_move(q_mid, q, cfg)
        assert "joint_range" in _rules(check_paths({a: m}, cfg, {a: into}, {a: q_mid}))
        assert check_paths({a: m}, cfg, {a: into}, {a: q_mid}, rest=True) == []


def test_safe_move_has_no_violations(models, cfg, home):
    q1 = home["right"].copy()
    q1[:5] = [20, -40, 60, 20, 10]
    p = plan_joint_move(home["right"], q1, cfg)
    assert check_paths(models, cfg, {"right": p}, home, rest=True) == []


def test_joint_range(models, cfg, home):
    q1 = home["left"].copy()
    q1[2] = 99.0                                  # elbow past 97.2 - 5
    v = check_paths(models, cfg, {"left": plan_joint_move(home["left"], q1, cfg)}, home, rest=True)
    v = [x for x in v if x.rule == "joint_range"]
    assert v and v[0].joint == "elbow_flex" and "outside usable range" in v[0].message and "hard limit" in v[0].message


def test_speed_and_acceleration(models, cfg, home):
    q0 = home["right"]
    q1 = q0.copy()
    q1[0] += 30.0
    fast = Path(np.array([0.0, 0.02, 0.04, 0.3]), np.array([q0, q0, q1, q1]))    # 30 deg in one tick
    assert {"speed", "acceleration"} <= _rules(check_paths(models, cfg, {"right": fast}, home, rest=True))


def test_floor(models, cfg, home):
    q1 = home["right"].copy()
    q1[:5] = [0, 60, 30, 30, 0]                  # reach forward and down into the table
    v = check_paths(models, cfg, {"right": plan_joint_move(home["right"], q1, cfg)}, home, rest=True)
    assert "floor" in _rules(v)
    assert any("mm above the board top" in x.message or "mm above the work surface" in x.message for x in v)


def test_workspace(models, cfg, home):
    import dataclasses
    tight = dataclasses.replace(cfg, workspace_max=(0.20, 0.55, 0.45))
    q1 = home["right"].copy()
    q1[:5] = [0, 0, 0, 0, 0]                     # tip at x ~ 0.35
    v = check_paths(models, tight, {"right": plan_joint_move(home["right"], q1, cfg)}, home, rest=True)
    assert "workspace" in _rules(v)


def test_arm_clearance(models, cfg, home):
    # both arms swing toward the middle, reaching out horizontally
    ql, qr = home["left"].copy(), home["right"].copy()
    ql[:5] = [40, 0, 0, 0, 0]
    qr[:5] = [-40, 0, 0, 0, 0]
    paths = {"left": plan_joint_move(home["left"], ql, cfg), "right": plan_joint_move(home["right"], qr, cfg)}
    v = check_paths(models, cfg, paths, home, rest=True)
    assert "arm_clearance" in _rules(v)
    # one arm alone into the middle is fine while the other rests
    assert "arm_clearance" not in _rules(check_paths(models, cfg, {"left": paths["left"]}, home, rest=True))


def test_start_and_not_finite(models, cfg, home):
    q = home["left"].copy()
    q[0] += 10
    assert "start" in _rules(check_paths(models, cfg, {"left": _still(q)}, home, rest=True))
    q = home["left"].copy()
    q[3] = np.nan
    assert _rules(check_paths(models, cfg, {"left": _still(q)}, home, rest=True)) == {"not_finite"}


def test_load_arm_limits_from_calibration_files():
    try:
        lim = load_arm_limits("bi_follower_left")
    except RuntimeError:
        pytest.skip("no calibration files on this machine")
    assert set(lim.hard) >= {"shoulder_pan", "gripper"}
    with pytest.raises(RuntimeError):
        load_arm_limits("no_such_arm_id")


def test_calibration_limits_are_read_without_lerobot(tmp_path, monkeypatch):
    """The stack reads lerobot's calibration JSON itself (packaging sprint 2026-10-13): same path, same numbers."""
    import json
    from stack.robotd import calibration
    cal = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 700, "range_max": 3400},
           "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 0, "range_min": 1417, "range_max": 2908},
           "broken": {"id": 7}}
    for i, j in enumerate(("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), 2):
        cal[j] = {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 900, "range_max": 3100}
    d = tmp_path / "robots" / "so_follower"
    d.mkdir(parents=True)
    (d / "arm_x.json").write_text(json.dumps(cal))
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))
    assert calibration.calibration_path("arm_x") == d / "arm_x.json"
    lim = calibration.read_joint_limits("arm_x")
    assert lim["gripper"] == (0.0, 100.0) and "broken" not in lim
    assert lim["shoulder_pan"] == pytest.approx((-118.68, 118.68), abs=0.01)
    assert load_arm_limits("arm_x").hard["shoulder_pan"] == lim["shoulder_pan"]
    assert calibration.read_joint_limits("no_such_arm") is None
    monkeypatch.delenv("HF_LEROBOT_CALIBRATION")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lr"))
    assert calibration.calibration_dir() == tmp_path / "lr" / "calibration"
    monkeypatch.delenv("HF_LEROBOT_HOME")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert calibration.calibration_dir() == tmp_path / "hf" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_HOME", "$HOME/hfx")                   # huggingface_hub expands variables
    monkeypatch.setenv("HOME", str(tmp_path))
    assert calibration.calibration_dir() == tmp_path / "hfx" / "lerobot" / "calibration"
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", "")             # empty counts as set, as in lerobot
    assert str(calibration.calibration_dir()) == "."
    try:                                                          # where webslider is installed: the same numbers
        from lerobot_teleoperator_webslider.limits import read_joint_limits as webslider_limits
    except ImportError:
        return
    assert webslider_limits("arm_x", d) == lim


def test_board_outline_sets_the_floor(models, cfg):
    import dataclasses
    rig = dataclasses.replace(cfg, board_front_x=0.025, board_half_width_y=0.24975)
    assert rig.floor_z(0.20, -0.18) == 0.0                      # out on the table
    assert rig.floor_z(0.00, -0.18) == 0.018                    # over the board
    assert rig.floor_z(0.00, -0.30) == 0.0                      # beside the board
    assert cfg.floor_z(0.20, -0.18) == 0.018                    # unmeasured outline: board everywhere


def test_low_grasp_on_the_table_is_allowed_only_past_the_board(models, cfg):
    import dataclasses
    from stack.robotd.planner import plan_linear
    rig = dataclasses.replace(cfg, board_front_x=0.025, board_half_width_y=0.24975)
    m = models["right"]
    lo, hi = m.limits.usable(rig)
    start = np.array([0.0, -30.0, 50.0, 40.0, 0.0, 50.0])
    for target, ok in (([0.22, -0.186, 0.025], True),           # 25 mm above the table: fine
                       ([0.22, -0.186, 0.012], False)):         # 12 mm: under the 20 mm tip clearance
        r = m.kin.solve_ik(target, -90.0, 0.0, seed_deg=start[:5], lower_deg=lo[:5], upper_deg=hi[:5])
        assert r.ok, r.message
        q = start.copy(); q[:5] = r.q
        v = check_paths({"right": m}, rig, {"right": _still(q)}, {"right": q})
        assert (not v) == ok, [x.message for x in v]
        if not ok:
            assert v[0].rule == "floor" and "work surface" in v[0].message
    # a tip 25 mm up is only 7 mm above the board top: refused over the board, as the floor model says
    assert rig.floor_z(0.0, -0.186) == 0.018 and 0.025 - rig.floor_z(0.0, -0.186) < rig.tip_clearance_m


def test_floor_checks_points_along_links_not_only_joints(models, cfg):
    from stack.robotd.safety import _floor_points, _keypoints
    m = models["left"]
    q = np.array([0.0, -30.0, 50.0, 40.0, 0.0, 50.0])
    pts, names = _floor_points(_keypoints(m, q[None]), cfg.floor_samples_per_link)
    assert pts.shape[1] == 5 + 4 * cfg.floor_samples_per_link and names.count("forearm") == cfg.floor_samples_per_link


STAND = ({"name": "camera stand", "min": [-0.115, -0.0325, 0.018], "max": [-0.055, 0.0125, 0.700]},)


def test_camera_stand_is_a_fixed_obstacle(models, cfg, home):
    import dataclasses
    rig = dataclasses.replace(cfg, obstacles=STAND)
    m = models["right"]
    q = home["right"].copy()
    q[:5] = [-99.0, -86.0, 68.0, 26.0, -1.0]                   # swung to the centre, folded back: into the stand
    v = check_paths({"right": m}, rig, {"right": _still(q)}, {"right": q})
    assert v and v[0].rule == "obstacle" and "camera stand" in v[0].message
    assert check_paths({"right": m}, cfg, {"right": _still(q)}, {"right": q}) == [] or \
        all(x.rule != "obstacle" for x in check_paths({"right": m}, cfg, {"right": _still(q)}, {"right": q}))
    for a in ("left", "right"):                                 # the rest poses clear it
        assert check_paths({a: models[a]}, rig, {a: _still(home[a])}, {a: home[a]}, rest=True) == []


def test_repo_config_carries_the_stand_and_board(tmp_path):
    from stack.robotd.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "robotd.yaml")
    names = [o["name"] for o in cfg.safety.obstacles]
    assert "camera stand" in names and cfg.safety.board_front_x == 0.025


def test_a_path_may_start_outside_the_usable_range_if_it_does_not_go_further_out(models, cfg, home):
    """A gripper resting a hair under its minimum (or an arm at the edge) must still be able to move."""
    from stack.robotd.planner import plan_joint_move
    m = models["left"]
    q0 = home["left"].copy()
    q0[5] = 98.1                                                  # gripper just over the usable 0..98
    q_open = q0.copy(); q_open[5] = 20.0
    assert check_paths({"left": m}, cfg, {"left": plan_joint_move(q0, q_open, cfg)}, {"left": q0}, rest=True) == []
    q_pan = q0.copy(); q_pan[0] = 20.0                            # the gripper holds at 98.1 while the arm moves
    assert check_paths({"left": m}, cfg, {"left": plan_joint_move(q0, q_pan, cfg)}, {"left": q0}, rest=True) == []
    q_out = q0.copy(); q_out[5] = 99.0                            # further out: still refused
    v = check_paths({"left": m}, cfg, {"left": plan_joint_move(q0, q_out, cfg)}, {"left": q0}, rest=True)
    assert v and v[0].rule == "joint_range" and v[0].joint == "gripper"


def test_a_path_may_lift_off_from_resting_on_the_table_but_not_go_lower(models, cfg):
    """An arm resting with its jaws on the table (below the tip clearance) must be able to leave.

    Found on the rig 2026-10-03 when the uncorrected model put the resting tip ~7 mm up; with the measured
    offsets the real rest is clear, so the pose here is one the current model puts below the clearance."""
    from stack.robotd.planner import plan_joint_move
    m = models["left"]
    resting = np.array([-4.26, -85.0, 96.35, 70.0, -7.6, 1.14])
    assert m.kin.fk(resting[:5])[2, 3] < cfg.tip_clearance_m
    up = resting.copy(); up[:5] = [0, -70, 70, 30, 0]
    assert check_paths({"left": m}, cfg, {"left": plan_joint_move(resting, up, cfg, 0.2)}, {"left": resting},
                       rest=True) == []
    down = resting.copy(); down[3] += 8                                  # tips the gripper further down
    v = check_paths({"left": m}, cfg, {"left": plan_joint_move(resting, down, cfg, 0.2)}, {"left": resting}, rest=True)
    assert v and v[0].rule in ("floor", "joint_range")


def test_robotd_yaml_lowers_the_tip_clearance_to_5_mm_and_a_12_mm_grasp_passes(models):
    """robotd.yaml: tip_clearance_m 5 mm (user, 2026-10-09, for cloth grasps; the default stays 20 mm). A tip 12 mm
    above the table, refused at 20 mm, now passes the point rule; 3 mm is still refused."""
    from stack.robotd.config import load_config
    from stack.robotd.planner import plan_linear
    from .test_robotd import REPO_CONFIG
    cfg = load_config(REPO_CONFIG).safety
    assert cfg.tip_clearance_m == 0.005
    m = models["right"]
    lo, hi = m.limits.usable(cfg)
    start = np.array([0.0, -30.0, 50.0, 40.0, 0.0, 50.0])
    for z, ok in ((0.012, True), (0.003, False)):
        r = m.kin.solve_ik([0.22, -0.186, z], -90.0, 0.0, seed_deg=start[:5], lower_deg=lo[:5], upper_deg=hi[:5])
        assert r.ok, r.message
        q = start.copy(); q[:5] = r.q
        v = [x for x in check_paths({"right": m}, cfg, {"right": _still(q)}, {"right": q}) if x.rule == "floor"]
        assert (not v) == ok, (z, [x.message for x in v])


@pytest.mark.hardware
def test_calibration_dir_is_lerobots():
    """Where lerobot is installed, the stack's resolver and lerobot's agree (the bus reads the same file)."""
    import subprocess
    import sys
    code = ("import lerobot.utils.constants as c; from stack.robotd.calibration import calibration_dir; "
            "assert calibration_dir() == c.HF_LEROBOT_CALIBRATION, (calibration_dir(), c.HF_LEROBOT_CALIBRATION)")
    repo = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    import os
    for extra in ({}, {"HF_HOME": "$HOME/hfx"}, {"HF_LEROBOT_HOME": "/tmp/lrh"}, {"XDG_CACHE_HOME": "/tmp/xdg"}):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "XDG_CACHE"))} | extra
        r = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (extra, r.stderr[-500:])
