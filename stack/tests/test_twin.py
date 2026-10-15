"""The MuJoCo twin (headless): scene from robot.yaml, simulated servos, and program previews."""

import json
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from stack.rig.kinematics import ArmKinematics
from stack.rig.rig import load_rig
from stack.robotd.safety import ArmLimits

from .conftest import CALIBRATED

mujoco = pytest.importorskip("mujoco")
try:
    from stack.twin_mujoco.scene import assets_dir
    assets_dir()
except FileNotFoundError:
    pytest.skip("SO-101 MuJoCo assets not fetched (tools/fetch_so101_assets.py)", allow_module_level=True)

from stack.twin_mujoco.preview import preview_program  # noqa: E402
from stack.twin_mujoco.scene import build_scene  # noqa: E402
from stack.twin_mujoco.sim import SimArmIO, SimWorld  # noqa: E402

LIMITS = {a: ArmLimits(dict(CALIBRATED[a])) for a in ("left", "right")}
BLOCK = [{"name": "block", "xy": [0.22, -0.17], "size": [0.04, 0.04, 0.04]}]
PICK = textwrap.dedent("""
    from robot_use import arm, estimate, contact, log
    r = arm("right")
    block = estimate("block", [0.22, -0.17], size=(0.04, 0.04, 0.04), tol=0.03, source="test")
    dy = 0.023                                     # the tip is the fixed jaw's inner face: 3 mm off the block's side
    r.open(80)
    r.move_to([block.xy[0], block.xy[1] + dy, 0.09], pitch=-70, roll=-90, speed=0.8)    # jaws square to its sides
    r.move_to([block.xy[0], block.xy[1] + dy, 0.015], pitch=-70, roll=-90, linear=True, speed=0.3, until=contact())
    g = r.close(effort=0.5)
    log("grasped " + str(bool(g.grasped)))
    r.move_by(dz=0.06)
""")


def test_twin_arms_match_robotd_kinematics_in_the_rig_frame(rng):
    rig = load_rig()
    sc = build_scene(rig, BLOCK)
    d = mujoco.MjData(sc.model)
    worst = 0.0
    for arm in ("left", "right"):
        kin = ArmKinematics(rig.arm_mount(arm))
        for _ in range(200):
            q = rng.uniform(-80, 80, 5)
            for i, a in enumerate(sc.arms[arm]["qpos"][:5]):
                d.qpos[a] = kin.to_urdf_rad(q)[i]
            mujoco.mj_kinematics(sc.model, d)
            worst = max(worst, float(np.linalg.norm(d.site_xpos[sc.arms[arm]["tip_site"]] - kin.fk(q)[:3, 3])))
    assert worst < 5e-5, f"{worst * 1000:.3f} mm"
    assert set(sc.objects) == {"block"}


def test_twin_current_reads_load_not_motion():
    rig = load_rig()
    q = [0.0, -40.0, 50.0, 40.0, 0.0, 80.0]
    w = SimWorld(build_scene(rig), start={"left": q, "right": q})
    w.start()
    try:
        time.sleep(0.4)
        r = SimArmIO(w, "right")
        target, peak = np.array(q), 0.0
        for k in range(60):                                   # close on nothing at ~150 %/s
            target[5] = max(2.0, 80 - 3.0 * k)
            r.write_goal(target)
            time.sleep(0.02)
            peak = max(peak, r.read().current_ma[5])
        assert peak < 320, f"free closing reads {peak:.0f} mA (the grasp threshold at effort 0.4 is 320)"
    finally:
        w.stop()


def test_preview_picks_a_block_for_real(tmp_path):
    rep = preview_program(PICK, BLOCK, limits=LIMITS, work_dir=tmp_path)
    assert rep["ok"] and rep["status"] == "done", rep
    b = rep["objects"]["block"]
    assert b["final"][2] - b["initial"][2] > 0.04, f"the block was not lifted: {b}"
    from stack.twin_mujoco.preview import PREVIEW_NOTE
    assert rep["note"] == PREVIEW_NOTE and "grasp physics are not" in PREVIEW_NOTE    # C6 (sprint 2026-10-07)
    logs = [json.loads(l).get("msg") for l in (Path(rep["trace_dir"]) / "trace.jsonl").read_text().splitlines()]
    assert "grasped True" in logs


def test_a_held_block_stays_in_the_jaws_through_a_carry(tmp_path):
    # the jaws collide as their convex pieces (stack/assets/gripper_collision), as in robotd's safety model; as one
    # convex hull each, the block tilted at grasp height and fell out in the carry (2026-10-15)
    carry = PICK + "r.move_to([0.27, -0.05, 0.09], pitch=-65, roll=-90, speed=0.2)\n"
    rep = preview_program(carry, BLOCK, limits=LIMITS, work_dir=tmp_path)
    assert rep["ok"] and rep["status"] == "done", rep
    b = rep["objects"]["block"]
    assert b["final"][2] - b["initial"][2] > 0.04 and np.hypot(*np.subtract(b["final"][:2], b["initial"][:2])) > 0.1, b


def test_preview_reports_what_failed(tmp_path):
    bad = PICK.replace("[block.xy[0], block.xy[1] + dy, 0.09]", "[block.xy[0] + 0.3, block.xy[1] + dy, 0.09]")   # beyond reach
    rep = preview_program(bad, BLOCK, limits=LIMITS, work_dir=tmp_path)
    assert not rep["ok"] and rep["step_failures"], rep
    assert "unreachable" in rep["step_failures"][0]["reason"]
    assert rep["objects"]["block"]["final"] == pytest.approx(rep["objects"]["block"]["initial"], abs=0.003)


def test_runner_rejects_a_program_whose_preview_fails(tmp_path):
    from stack.runner.runner import ProgramRunner
    from stack.twin_mujoco.preview import preview_program as pp
    r = ProgramRunner("http://127.0.0.1:9", tmp_path / "ep",
                      preview=lambda code, scene=None: pp(code, scene, limits=LIMITS, work_dir=tmp_path / "twin"))
    out = r.submit(PICK.replace("[block.xy[0], block.xy[1] + dy, 0.09]", "[block.xy[0] + 0.3, block.xy[1] + dy, 0.09]"), scene=BLOCK)
    st = r.status(out["program_id"], wait_s=30)
    assert st["status"] == "rejected" and "preview" in st["reason"], st
    assert st["preview"]["step_failures"]
    skipped = ProgramRunner("http://127.0.0.1:9", tmp_path / "ep2").submit("x = 1\n", skip_preview=True)
    assert skipped["ok"]


# ---------------------------------------------------------------- scene files (the simulated rig's objects)
from stack.twin_mujoco.scene_file import footprint_radius, load_scene_file  # noqa: E402


def _settle(scene, start=None, seconds=1.5):
    """Step a world with the arms held at `start` (no thread); return the data."""
    w = SimWorld(scene, start=start)
    for _ in range(int(seconds / scene.model.opt.timestep)):
        mujoco.mj_step(w.model, w.data)
    return w


def test_a_scene_file_draws_the_same_layout_from_the_same_seed():
    rig = load_rig()
    a, b, c = (load_scene_file("block_cup", s) for s in (7, 7, 8))
    assert a.seed == 7 and [o.xyz for o in a.objects] == [o.xyz for o in b.objects]
    assert [o.xyz for o in a.objects] != [o.xyz for o in c.objects]
    drawn = load_scene_file("block_cup", random=True)               # random: a seed is drawn and reported,
    again = load_scene_file("block_cup", drawn.seed)                # so the run can be repeated
    assert [o.xyz for o in drawn.objects] == [o.xyz for o in again.objects]
    for seed in range(50):
        objs = load_scene_file("block_cup", seed).objects
        assert {o.name for o in objs} == {"blue_block", "pink_cup", "cream_cup"}
        for i, o in enumerate(objs):
            assert o.xyz[0] - footprint_radius(o) >= rig.board.front_x_m        # off the board
            assert o.xyz[2] == pytest.approx(o.size[2] / 2)                     # standing on the table
            for p in objs[i + 1:]:                                              # min_gap_m apart (0.03)
                gap = np.hypot(o.xyz[0] - p.xyz[0], o.xyz[1] - p.xyz[1]) - footprint_radius(o) - footprint_radius(p)
                assert gap >= 0.03, (seed, o.name, p.name, gap)


def test_a_scene_file_starts_from_its_start_layout_unless_asked_for_another(tmp_path):
    rig = load_rig()
    a, b = load_scene_file("block_cup"), load_scene_file("block_cup")      # the canned first task: every run alike
    assert a.seed is None and [(o.xyz, o.yaw_deg) for o in a.objects] == [(o.xyz, o.yaw_deg) for o in b.objects]
    pose = {o.name: (o.xyz[:2], o.yaw_deg) for o in a.objects}
    assert pose["blue_block"] == ((0.22, -0.15), 10.0) and pose["pink_cup"][0] == (0.27, -0.03)
    for i, o in enumerate(a.objects):                                      # the same rules as a drawn layout
        assert o.xyz[0] - footprint_radius(o) >= rig.board.front_x_m and o.xyz[2] == pytest.approx(o.size[2] / 2)
        for p in a.objects[i + 1:]:
            assert np.hypot(o.xyz[0] - p.xyz[0], o.xyz[1] - p.xyz[1]) - footprint_radius(o) - footprint_radius(p) >= 0.03
    r = load_scene_file("block_cup", random=True)                         # --random: drawn, with its seed
    assert r.seed is not None and [o.xyz for o in r.objects] == [o.xyz for o in load_scene_file("block_cup", r.seed).objects]
    assert [o.xyz for o in load_scene_file("block_cup", 7).objects] != [o.xyz for o in a.objects]
    f = tmp_path / "s.yaml"
    obj = "{{name: {n}, size: [0.04, 0.04, 0.04], x: [0.2, 0.3], y: [-0.2, 0.2]{st}}}"
    f.write_text("objects:\n  - " + obj.format(n="a", st="") + "\n")
    assert load_scene_file(f).seed is not None                             # no start layout: one is drawn
    for st, other, match in [(", start: {xy: [0.25, 0.0]}", "", "on every object"),
                             (", start: {xy: [0.25, 0.0]}", ", start: {xy: [0.26, 0.01]}", "within min_gap_m"),
                             (", start: {xy: [0.0, 0.0]}", ", start: {xy: [0.25, 0.1]}", "on the board"),
                             (", start: {xy: [0.25, 0.0], z: 1}", ", start: {xy: [0.25, 0.1]}", "start is"),
                             (", start: {xy: [0.25]}", ", start: {xy: [0.25, 0.1]}", "start xy")]:
        f.write_text("objects:\n  - " + obj.format(n="a", st=st) + "\n  - " + obj.format(n="b", st=other) + "\n")
        with pytest.raises(ValueError, match=match):
            load_scene_file(f)
    f.write_text("objects:\n  - " + obj.format(n="a", st=", start: {xy: [0.0, 0.0]}") + "\n")
    assert load_scene_file(f, 3).seed == 3                                 # a drawn layout ignores the start poses


def test_a_scene_file_refuses_what_it_cannot_honour(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("objects:\n  - {name: a, size: [0.04, 0.04, 0.04], xy: [0.2, 0.0], colour: red}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_scene_file(bad, 0)
    bad.write_text("objects:\n  - {name: a, shape: ball, xy: [0.2, 0.0]}\n")
    with pytest.raises(ValueError, match="shape"):
        load_scene_file(bad, 0)
    for obj, match in [("{name: a, shape: cup, size: [0.08, 0.06, 0.05], xy: [0.2, 0.0]}", "round"),
                       ("{name: a, x: [0.2, 0.25]}", "give xy"),
                       ("{name: a, x: '0.2', y: 0.0}", "a number"),
                       ("{size: [0.04, 0.04, 0.04], xy: [0.2, 0.0]}", "with a name")]:
        bad.write_text(f"objects:\n  - {obj}\n")
        with pytest.raises(ValueError, match=match):
            load_scene_file(bad, 0)
    bad.write_text("- just a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_scene_file(bad, 0)
    bad.write_text("objects:\n" + "".join(f"  - {{name: c{i}, shape: cup, size: [0.08, 0.08, 0.05], "
                                          f"x: [0.2, 0.25], y: [0.0, 0.05]}}\n" for i in range(4)))
    with pytest.raises(ValueError, match="could not place"):        # too crowded: an error, not overlapping cups
        load_scene_file(bad, 0)


def test_the_scene_objects_rest_on_the_table_clear_of_the_arms_at_rest():
    from stack.robotd.config import load_config
    from stack.robotd.safety import ArmModel, rest_pose
    rig, safety = load_rig(), load_config().safety
    start = {a: rest_pose(ArmModel(a, ArmKinematics(rig.arm_mount(a)), LIMITS[a]), safety) for a in LIMITS}
    for seed in (0, 1, 2):
        layout = load_scene_file("block_cup", seed)
        sc = build_scene(rig, layout.objects)
        w = _settle(sc, start)
        for o in layout.objects:
            pos = w.object_pose(o.name)
            assert np.linalg.norm(np.subtract(pos, o.xyz)) < 0.002, (seed, o.name, pos, o.xyz)
            quat = w.data.qpos[sc.objects[o.name]["qpos"] + 3:sc.objects[o.name]["qpos"] + 7]
            up = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)                 # the body's z axis . world z
            assert up > 0.999, (seed, o.name, "tipped")
        obj_bodies = {v["body"] for v in sc.objects.values()}
        for i in range(w.data.ncon):
            b1, b2 = (int(sc.model.geom_bodyid[g]) for g in (w.data.contact[i].geom1, w.data.contact[i].geom2))
            names = {mujoco.mj_id2name(sc.model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in (b1, b2)}
            if obj_bodies & {b1, b2}:
                assert not any(n.startswith(("left_", "right_")) for n in names), (seed, names)


def test_a_cups_pieces_stay_inside_its_footprint():
    from stack.twin_mujoco.scene_file import footprint_radius as fr
    from stack.twin_mujoco.scene import SceneObject
    cup = SceneObject.from_dict({"name": "cup", "shape": "cup", "size": [0.085, 0.085, 0.05], "xy": [0.25, 0.0]})
    sc = build_scene(load_rig(), [cup])
    m, d = sc.model, mujoco.MjData(sc.model)
    mujoco.mj_forward(m, d)
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_cup")
    for g in (g for g in range(m.ngeom) if m.geom_bodyid[g] == body):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:                  # the disc: its radius
            assert np.hypot(d.geom_xpos[g][0] - 0.25, d.geom_xpos[g][1]) + m.geom_size[g][0] <= fr(cup) + 1e-6
            continue
        R, half = d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            p = d.geom_xpos[g] + R @ (corner * half)
            assert np.hypot(p[0] - 0.25, p[1]) <= fr(cup) + 1e-6, (g, p)


def test_a_block_dropped_into_a_cup_stays_in_it():
    rig = load_rig()
    cup = {"name": "cup", "shape": "cup", "size": [0.085, 0.085, 0.05], "xy": [0.25, 0.0], "wall_m": 0.004}
    inner_r = 0.085 / 2 - 0.004
    # centred: it lands flat on the cup's base (4 mm thick)
    block = {"name": "block", "size": [0.04, 0.04, 0.04], "xyz": [0.25, 0.0, 0.09]}
    w = _settle(build_scene(rig, [cup, block]), seconds=2.0)
    b = np.array(w.object_pose("block"))
    assert np.hypot(b[0] - 0.25, b[1]) < 0.002 and b[2] == pytest.approx(0.004 + 0.02, abs=0.001), b
    # off-centre and turned: it may tip against the wall, but it stays inside, below the rim, and the cup stands
    block = {"name": "block", "size": [0.04, 0.04, 0.04], "xyz": [0.25, 0.012, 0.09], "yaw_deg": 20}
    w = _settle(build_scene(rig, [cup, block]), seconds=2.0)
    b, c = np.array(w.object_pose("block")), np.array(w.object_pose("cup"))
    assert np.hypot(*(b - c)[:2]) < inner_r and 0.004 < b[2] < 0.05, (b, c)
    assert np.linalg.norm(c - [0.25, 0.0, 0.025]) < 0.005, c
    # near the wall, on an edge: it rests on the base, never through it, and the cup stays down (2026-10-15: a
    # 4 mm cylinder base let such a block through, and the cup rose round it)
    for dy, yaw in ((0.015, 45), (-0.016, 30), (0.0, 45)):
        block = {"name": "block", "size": [0.04, 0.04, 0.04], "xyz": [0.25, dy, 0.08], "yaw_deg": yaw}
        w = _settle(build_scene(rig, [cup, block]), seconds=2.0)
        b, c = np.array(w.object_pose("block")), np.array(w.object_pose("cup"))
        assert b[2] > 0.004 + 0.015 and np.hypot(*(b - c)[:2]) < inner_r, (dy, yaw, b, c)
        assert np.linalg.norm(c - [0.25, 0.0, 0.025]) < 0.005, (dy, yaw, c)
    # beside the cup: it lands on the table, not through the wall
    block = {"name": "block", "size": [0.04, 0.04, 0.04], "xyz": [0.25, 0.07, 0.09]}
    w = _settle(build_scene(rig, [cup, block]), seconds=2.0)
    assert np.allclose(w.object_pose("block"), [0.25, 0.07, 0.02], atol=0.002)
