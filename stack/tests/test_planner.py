import math

import numpy as np
import pytest

from stack.robotd.planner import PlanError, plan_joint_move, plan_linear, plan_waypoints
from stack.robotd.safety import _rate_violations, check_paths, rest_pose

from .conftest import reachable_sample


def _caps_ok(model, cfg, p):
    return _rate_violations(model, cfg, p.t, p.q) == []


def test_joint_move_caps_endpoints_and_sync(models, cfg, rng):
    m = models["left"]
    qs = reachable_sample(m, cfg, rng, 200)
    for a, b in zip(qs[::2], qs[1::2]):
        for speed in (0.2, 0.5, 1.0):
            p = plan_joint_move(a, b, cfg, speed)
            assert _caps_ok(m, cfg, p)
            assert np.allclose(p.q[0], a) and np.allclose(p.q[-1], b)
            # synchronised: every joint is the same fraction of the way at every sample
            d = b - a
            k = int(np.argmax(np.abs(d)))
            frac = (p.q[:, k] - a[k]) / d[k]
            assert np.allclose(p.q, a + frac[:, None] * d, atol=1e-9)
    # speed 0.5 takes longer than 1.0
    a, b = qs[0], qs[1]
    assert plan_joint_move(a, b, cfg, 0.5).duration > plan_joint_move(a, b, cfg, 1.0).duration


def test_joint_move_rejects_bad_speed(cfg):
    for s in (0, -1, 1.5, math.nan):
        with pytest.raises(PlanError):
            plan_joint_move(np.zeros(6), np.ones(6), cfg, s)


def test_zero_move(cfg):
    p = plan_joint_move(np.ones(6), np.ones(6), cfg)
    assert p.duration == 0 and len(p.t) == 1


def test_waypoints_reject_vs_auto(models, cfg):
    m = models["right"]
    q0 = rest_pose(m, cfg)
    w1 = q0.copy(); w1[:5] = [30, -30, 40, 20, 0]
    w2 = q0.copy(); w2[:5] = [-20, -10, 20, 40, 45]
    fast = [(0.3, w1), (0.6, w2)]
    p, scale = plan_waypoints(q0, fast, cfg, "reject")
    assert scale == 1.0 and not _caps_ok(m, cfg, p)
    p, scale = plan_waypoints(q0, fast, cfg, "auto")
    assert scale > 1.0 and _caps_ok(m, cfg, p)
    assert np.allclose(p.q[-1], w2, atol=1e-6)


def test_waypoints_bad_times(cfg):
    q = np.zeros(6)
    for w in ([], [(0.0, q)], [(1.0, q), (0.5, q)], [(math.nan, q)]):
        with pytest.raises(PlanError):
            plan_waypoints(q, w, cfg)


def test_linear_is_straight_and_capped(models, cfg):
    m = models["right"]
    q0 = rest_pose(m, cfg)
    start = q0.copy(); start[:5] = [0, -30, 50, 40, 0]
    p0 = m.kin.pose(start)["xyz"]
    target = p0 + np.array([0.08, -0.05, 0.03])
    p = plan_linear(m, cfg, start, target, speed=0.8)
    tips = np.array([m.kin.fk(q)[:3, 3] for q in p.q])
    u = (target - p0) / np.linalg.norm(target - p0)
    off = tips - p0
    dev = np.linalg.norm(off - (off @ u)[:, None] * u, axis=1)
    assert np.max(dev) < 0.003, f"strays {np.max(dev) * 1000:.1f} mm off the line"
    assert np.linalg.norm(tips[-1] - target) < 0.002
    assert _caps_ok(m, cfg, p)
    assert check_paths({"right": m}, cfg, {"right": p}, {"right": start}) == []


def test_linear_rejects_line_leaving_reach(models, cfg):
    m = models["right"]
    start = rest_pose(m, cfg); start[:5] = [0, -30, 50, 40, 0]
    with pytest.raises(PlanError, match="reachable"):
        plan_linear(m, cfg, start, [0.9, -0.2, 0.1])


def test_a_straight_lift_near_the_wrist_limit_is_not_a_crawl():
    """Review 2026-10-03: a 9 cm lift after a grasp, ending near the wrist limit, was retimed uniformly to 198 s
    because one sharp bend in joint space set the pace for the whole line. Timing along the path keeps it short."""
    import numpy as np
    from stack.rig.kinematics import ArmKinematics, default_mount
    from stack.robotd.config import load_config
    from stack.robotd.planner import plan_linear
    from stack.robotd.safety import ArmLimits, ArmModel, check_paths
    from .conftest import CALIBRATED
    cfg = load_config().safety
    m = ArmModel("right", ArmKinematics(default_mount("right")), ArmLimits(dict(CALIBRATED["right"])))
    lo, hi = m.limits.usable(cfg)
    r = m.kin.solve_ik([0.297, -0.112, 0.022], -70, 0, seed_deg=np.array([0, 10, 20, 60, 0.0]),
                       lower_deg=lo[:5], upper_deg=hi[:5])
    assert r.ok
    q0 = np.r_[r.q, 2.0]
    for target, pitch in (([0.3035, -0.1104, 0.1083], -66.6), ([0.297, -0.112, 0.05], -70.0)):
        p = plan_linear(m, cfg, q0, target, pitch, 0.0, 0.3)
        assert p.duration < 8.0, p.duration
        assert check_paths({"right": m}, cfg, {"right": p}, {"right": q0}) == []
        assert np.allclose(m.kin.pose(p.q[-1])["xyz"], target, atol=0.003)
