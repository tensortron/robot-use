import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from stack.rig.kinematics import JOINTS, ArmKinematics, ArmMount, default_mount

from .conftest import reachable_sample

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _mujoco_model():
    """The upstream MJCF with meshes stripped (kinematics only)."""
    mujoco = pytest.importorskip("mujoco")
    root = ET.parse(ASSETS / "so101_new_calib_camera.xml").getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom" and child.get("type") == "mesh"):
                parent.remove(child)
    for c in root.iter("compiler"):
        c.attrib.pop("meshdir", None)
    return mujoco, mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_fk_matches_mujoco(rng):
    mujoco, m = _mujoco_model()
    d = mujoco.MjData(m)
    kin = ArmKinematics(ArmMount(xyz=(0.0, 0.0, 0.0)))
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    worst = 0.0
    for _ in range(1000):
        q = rng.uniform(-100, 100, 5)
        for j, v in zip(JOINTS, q):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = math.radians(v)
        mujoco.mj_kinematics(m, d)
        worst = max(worst, float(np.linalg.norm(d.site_xpos[site] - kin.fk(q)[:3, 3])))
    assert worst < 0.005, f"FK differs from MuJoCo by up to {worst * 1000:.2f} mm"


def test_mount_places_pan_axes_371_5mm_apart():
    l, r = (ArmKinematics(default_mount(a)).keypoints(np.zeros(5))[0] for a in ("left", "right"))
    assert abs(l[1] - r[1] - 0.3715) < 1e-6 and abs(l[0]) < 1e-6 and abs(r[0]) < 1e-6
    assert l[1] > 0 > r[1], "left arm must be on +y (the robot's left)"


def test_pitch_convention():
    from dataclasses import replace
    kin = ArmKinematics(replace(default_mount("right"), offsets_deg=(0.0,) * 5))   # the URDF's own zero
    assert abs(kin.pitch_deg(np.zeros(5))) < 0.5            # zero pose reaches forward horizontally
    tip0 = kin.fk(np.zeros(5))[:3, 3]
    assert tip0[0] > 0.25                                    # ... in front of the base (+x)
    # wrist_flex alone changes pitch one for one; wrist_roll never changes it
    assert abs(kin.pitch_deg([0, 0, 0, 30, 0]) - kin.pitch_deg([0, 0, 0, 0, 0]) + 30) < 0.5
    assert abs(kin.pitch_deg([0, 0, 0, 0, 77]) - kin.pitch_deg([0, 0, 0, 0, 0])) < 1e-3


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("with_pitch", [True, False])
def test_ik_round_trip(models, cfg, rng, arm, with_pitch):
    m = models[arm]
    lo, hi = m.limits.usable(cfg)
    targets = reachable_sample(m, cfg, rng, 2500)
    seed = np.array([0.0, -95.0, 90.0, 30.0, 0.0])
    ok, times = 0, []
    for q in targets:
        p = m.kin.pose(q)
        t0 = time.perf_counter()
        r = m.kin.solve_ik(p["xyz"], p["pitch"] if with_pitch else None, q[4], seed_deg=seed,
                           lower_deg=lo[:5], upper_deg=hi[:5])
        times.append(time.perf_counter() - t0)
        if r.ok:
            got = m.kin.pose(r.q)
            assert np.linalg.norm(got["xyz"] - p["xyz"]) <= 0.002
            if with_pitch:
                assert abs(got["pitch"] - p["pitch"]) <= 2.0
            assert np.all(r.q >= lo[:5] - 1e-6) and np.all(r.q <= hi[:5] + 1e-6)
            ok += 1
    assert ok / len(targets) >= 0.99, f"only {ok}/{len(targets)} reachable targets solved"
    assert np.median(times) < 0.002, f"median IK {np.median(times) * 1000:.2f} ms"


def test_ik_rejects_unreachable_with_closest_point(models, cfg):
    m = models["right"]
    lo, hi = m.limits.usable(cfg)
    for target in ([0.9, -0.2, 0.1], [0.1, -0.2, 0.9], [-0.6, -0.2, 0.05]):
        r = m.kin.solve_ik(target, None, 0.0, seed_deg=np.zeros(5), lower_deg=lo[:5], upper_deg=hi[:5])
        assert not r.ok
        assert "unreachable" in r.message and "mm from the target" in r.message
        assert r.pos_err_m > 0.05


def test_ik_rejects_roll_outside_range(models, cfg):
    m = models["left"]
    lo, hi = m.limits.usable(cfg)
    r = m.kin.solve_ik([0.25, 0.18, 0.1], None, 179.0, seed_deg=np.zeros(5), lower_deg=lo[:5], upper_deg=hi[:5])
    assert not r.ok and "roll" in r.message


def test_ik_prefers_solution_near_seed(models, cfg, rng):
    """A target reachable elbow-up and elbow-down comes back in the seed's branch."""
    m = models["right"]
    lo, hi = m.limits.usable(cfg)
    tested = 0
    for q in reachable_sample(m, cfg, rng, 200):
        p = m.kin.pose(q)
        branches = m.kin._closed_form(p["xyz"], np.deg2rad([p["pitch"]]), q[4])
        if len(branches) > 1 and np.sort(np.max(np.abs(branches - q[:5]), axis=1))[1] < 15.0:
            continue                                 # near the straight-elbow singularity both branches are "near"
        tested += 1
        r = m.kin.solve_ik(p["xyz"], p["pitch"], q[4], seed_deg=q[:5] + rng.normal(0, 3, 5) * [1, 1, 1, 1, 0],
                           lower_deg=lo[:5], upper_deg=hi[:5])
        assert r.ok and np.max(np.abs(r.q - q[:5])) < 5.0
    assert tested > 150
