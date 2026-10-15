"""robot.yaml: the single measured description, and what is derived from it."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from stack.rig.kinematics import ArmKinematics
from stack.rig.rig import DEFAULT_RIG, Rig, RigError, load_rig


@pytest.fixture
def raw():
    return yaml.safe_load(DEFAULT_RIG.read_text())


def test_mounts_put_the_pan_axes_on_the_rig_frame(raw):
    rig = load_rig()
    for arm, y in (("left", 0.18575), ("right", -0.18575)):
        pan = ArmKinematics(rig.arm_mount(arm)).keypoints(np.zeros(5))[0]
        assert np.allclose(pan[:2], [0.0, y], atol=1e-6), pan
        dz = raw["arms"][arm].get("base_z_offset_m", 0.0)
        assert abs(rig.arm_mount(arm).xyz[2] - (0.018 + dz)) < 1e-9   # base plates on the board top (+ measured)


def test_base_z_offset_raises_the_whole_arm(raw, tmp_path):
    r = copy.deepcopy(raw)
    r["arms"]["left"]["base_z_offset_m"] = 0.005
    k0 = ArmKinematics(Rig(dict(raw, arms={**raw["arms"], "left": {**raw["arms"]["left"], "base_z_offset_m": 0.0}}),
                           tmp_path / "robot.yaml").arm_mount("left"))
    k1 = ArmKinematics(Rig(r, tmp_path / "robot.yaml").arm_mount("left"))
    q = np.array([10.0, 20, -30, 15, 0])
    assert np.allclose(k1.keypoints(q) - k0.keypoints(q), [0, 0, 0.005])


def test_board_stand_and_twin_offset():
    rig = load_rig()
    b = rig.board
    assert (b.thickness_m, b.front_x_m) == (0.018, 0.025) and abs(b.half_width_y_m - 0.24975) < 1e-9
    stand = next(o for o in rig.obstacles() if o["name"] == "camera stand")
    assert np.allclose(stand["min"], [-0.115, -0.0125, 0.018]) and np.allclose(stand["max"], [-0.075, 0.0125, 0.700])
    ox, oy, oz = rig.sim_offset()
    assert abs(ox - 0.0388) < 1e-4 and oy == 0.0 and oz == -0.018    # matches the twin's rig_frame block


def test_every_section_states_its_sources(raw):
    for section in ("arms", "board", "cameras", "obstacles", "gripper", "grasp", "tcp"):
        assert raw[section].get("sources"), section


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["arms"]["left"].__setitem__("joint_signs", [1, 1, 2, 1, 1]), "joint_signs"),
    (lambda r: r["arms"]["right"].__setitem__("joint_offsets_deg", [0, 0]), "joint_offsets_deg"),
    (lambda r: r["board"].pop("sources"), "sources"),
    (lambda r: r["arms"]["right"].__setitem__("base_z_offset_m", 0.05), "base_z_offset_m"),
    (lambda r: r.__setitem__("version", 2), "version"),
])
def test_bad_rig_files_are_refused(raw, tmp_path, mutate, msg):
    r = copy.deepcopy(raw)
    mutate(r)
    with pytest.raises(RigError, match=msg):
        Rig(r, tmp_path / "robot.yaml")


def test_a_sign_flip_reaches_the_kinematics(raw, tmp_path):
    r = copy.deepcopy(raw)
    r["arms"]["left"]["joint_signs"] = [-1, 1, 1, 1, 1]
    flipped = Rig(r, tmp_path / "robot.yaml").arm_mount("left")
    k0, k1 = ArmKinematics(load_rig().arm_mount("left")), ArmKinematics(flipped)
    q = np.array([30.0, 0, 0, 0, 0])
    assert np.allclose(k0.fk(q)[:3, 3] * [1, 1, 1], k1.fk(-q)[:3, 3])


def test_description_is_available():
    assert "SO-101" in load_rig().description


def test_twin_export_derives_and_detects_drift(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    twin_export = pytest.importorskip("twin_export", reason="the Digital Twin's tool is not in the public repo")

    ours = twin_export.derive(load_rig())
    assert ours["rig_frame"] == {"pan_axis_x_m": 0.0388, "work_surface_z_m": -0.018}
    twin = {"arm_sep_m": 0.3715, "rig_frame": dict(ours["rig_frame"]), "rig_geometry": {},
            "calibration": copy.deepcopy(ours["calibration"]),
            **{k: copy.deepcopy(ours[k]) for k in ("calibration_per_arm", "arm_base_z_offset_m") if k in ours},
            "cameras": [{"name": n, **{k: v for k, v in f.items()}} for n, f in ours["cameras"].items()]}
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(twin))
    assert twin_export.check(ours, p) == []
    twin["rig_frame"]["work_surface_z_m"] = 0.0                    # the pre-78f6194 table height
    twin["calibration"]["shoulder_pan"]["scale"] = -1.0
    p.write_text(yaml.safe_dump(twin))
    problems = twin_export.check(ours, p)
    assert any("work_surface_z_m" in x for x in problems) and any("shoulder_pan.scale" in x for x in problems)


def test_twin_export_checks_each_arm_and_its_base_height(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    twin_export = pytest.importorskip("twin_export", reason="the Digital Twin's tool is not in the public repo")

    raw = yaml.safe_load(DEFAULT_RIG.read_text())
    raw["arms"]["right"]["joint_offsets_deg"] = [0, 0, -9.0, -3.0, 0]
    raw["arms"]["right"]["base_z_offset_m"] = 0.006
    ours = twin_export.derive(Rig(raw, DEFAULT_RIG))
    assert ours["calibration_per_arm"]["right"]["elbow_flex"]["offset"] == -9.0
    assert ours["arm_base_z_offset_m"]["right"] == 0.006
    twin = {"arm_sep_m": 0.3715, "rig_frame": dict(ours["rig_frame"]), "rig_geometry": {},
            "calibration": copy.deepcopy(ours["calibration"]),
            "cameras": [{"name": n, **f} for n, f in ours["cameras"].items()]}
    p = tmp_path / "so101.yaml"
    p.write_text(yaml.safe_dump(twin))
    problems = twin_export.check(ours, p)
    assert any("calibration_per_arm.right.elbow_flex.offset" in x for x in problems), problems
    assert any("arm_base_z_offset_m.right" in x for x in problems), problems


def test_the_camera_urdf_leaves_the_arm_kinematics_unchanged():
    """Sprint 2026-10-05 task 3a: the upstream camera variant adds only the wrist camera links."""
    from dataclasses import replace
    from stack.rig.kinematics import ArmKinematics
    rig = load_rig()
    assert rig.urdf_path.name == "so101_new_calib_camera.urdf"
    old = rig.urdf_path.with_name("so101_new_calib.urdf")
    rng = np.random.default_rng(3)
    for arm in ("left", "right"):
        m = rig.arm_mount(arm)
        k_new, k_old = ArmKinematics(m, urdf_path=rig.urdf_path), ArmKinematics(m, urdf_path=old)
        for _ in range(200):
            q = rng.uniform(-90, 90, 5)
            assert np.allclose(k_new.keypoints(q), k_old.keypoints(q), atol=1e-12)


def test_wrist_camera_pose_comes_from_the_urdf():
    rig = load_rig()
    for name in ("wrist_left", "wrist_right"):
        c = rig.camera(name)
        assert c["parent"] == "gripper_link" and c["urdf_link"] == "wrist_camera_link"
        x, y, z, w = c["rot_xyzw"]
        axis = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        # the optical axis validated against the agent's wrist images and the operator's view (2026-10-05)
        assert np.degrees(np.arccos(np.clip(axis @ np.array([0.015, -0.422, -0.907]), -1, 1))) < 2.0
        assert 0.06 < c["offset_m"][1] < 0.09             # on the arm's own left of the wrist axis at roll 0
    raw = yaml.safe_load(DEFAULT_RIG.read_text())
    assert "offset_m" not in raw["cameras"]["wrist_left"]  # no hand-copied mount numbers


def test_the_twin_model_carries_the_wrist_camera():
    from stack.rig.rig import REPO
    xml = (REPO / "stack" / "assets" / "so101_new_calib_camera.xml").read_text()
    assert 'body name="wrist_camera_mount"' in xml and 'body name="wrist_camera"' in xml


def test_camera_files_resolve_relative_to_the_repo(tmp_path):
    """The ego calibration lives in the repo (public-repo sprint, task 2); absolute and ~ paths still work."""
    from stack.rig.rig import REPO
    rig = load_rig()
    assert rig.camera_file("ego", "intrinsics_file") == REPO / "calib/ego_intrinsics.json"
    assert rig.camera_file("ego", "extrinsics_file").is_file()
    raw = yaml.safe_load((REPO / "robot.yaml").read_text())
    raw["cameras"]["ego"]["intrinsics_file"] = str(tmp_path / "i.json")
    raw["cameras"]["ego"]["extrinsics_file"] = "~/e.json"
    other = Rig(raw, tmp_path / "robot.yaml")
    assert other.camera_file("ego", "intrinsics_file") == tmp_path / "i.json"
    assert other.camera_file("ego", "extrinsics_file") == Path("~/e.json").expanduser()
    with pytest.raises(RigError):
        rig.camera_file("wrist_left", "intrinsics_file")


def _table_grid():
    return np.array([[x, y, 0.0] for x in np.linspace(0.12, 0.42, 7) for y in np.linspace(-0.25, 0.25, 7)])


def test_the_camera_comes_from_its_files_and_otherwise_from_rig_yamls_estimates(tmp_path):
    """Calibration sprint 2026-10-14, D2: without calibration files the stack builds the ego camera from robot.yaml's
    numbers, so a new rig runs on tape-measured estimates. The files win whenever they are named."""
    from stack.rig.camera_model import CameraModel
    from stack.rig.rig import REPO
    rig = load_rig()
    cal, src = rig.camera_model("ego")
    assert src == {"lens": "calibrated", "pose": "calibrated"}
    ref = CameraModel.from_files(rig.camera_file("ego", "intrinsics_file"), rig.camera_file("ego", "extrinsics_file"))
    assert np.allclose(cal.K, ref.K) and np.allclose(cal.R_cw, ref.R_cw) and np.allclose(cal.t_cw, ref.t_cw)

    # the inverse is exact: an estimate made from the calibrated camera's own pose is that camera
    R_cw, t_cw = CameraModel.pose_estimate(**cal.pose_angles())
    assert np.allclose(R_cw, cal.R_cw, atol=1e-9) and np.allclose(t_cw, cal.t_cw, atol=1e-9)

    raw = yaml.safe_load((REPO / "robot.yaml").read_text())
    ego = raw["cameras"]["ego"]
    for k in ("intrinsics_file", "extrinsics_file"):
        ego.pop(k)
    est, src = Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    assert src == {"lens": "estimated", "pose": "estimated"}
    px_cal, _ = cal.project(_table_grid())
    px_est, _ = est.project(_table_grid())
    assert np.abs(px_est - px_cal).max() < 4.0         # robot.yaml's rounded summary: within a few pixels

    ego.pop("distortion")                             # a datasheet's field of view alone: a pinhole, no distortion
    rough, src = Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    assert src["lens"] == "estimated" and rough.dist.tolist() == [0.0] * 5
    centre = np.array([[0.27, 0.0, 0.0]])          # this wide lens's barrel distortion: ~50 px, ~5 cm on the table
    assert np.abs(rough.project(centre)[0] - cal.project(centre)[0]).max() < 60.0

    ego["intrinsics_file"] = "calib/ego_intrinsics.json"    # a calibrated lens with an estimated pose
    mixed, src = Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    assert src == {"lens": "calibrated", "pose": "estimated"} and np.allclose(mixed.K, cal.K)


def test_a_camera_without_files_or_numbers_says_what_is_missing(tmp_path):
    from stack.rig.rig import REPO
    raw = yaml.safe_load((REPO / "robot.yaml").read_text())
    ego = raw["cameras"]["ego"]
    for k in ("intrinsics_file", "extrinsics_file", "eye_m"):
        ego.pop(k)
    with pytest.raises(RigError, match="eye_m and tilt_down_deg"):
        Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    for k in ("distortion", "fov_h_deg"):
        ego.pop(k)
    with pytest.raises(RigError, match="focal_px nor fov_h_deg"):
        Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    ego["extrinsics_file"] = "calib/ego_extrinsics.json"      # a pose without the lens it was solved with
    with pytest.raises(RigError, match="needs its intrinsics_file"):
        Rig(raw, tmp_path / "robot.yaml").camera_model("ego")
    with pytest.raises(RigError, match="rides on an arm"):
        load_rig().camera_model("wrist_left")


def test_the_operating_agent_cannot_read_the_rig_geometry_or_camera_calibration():
    """robot.yaml, the ego calibration and the generated twin geometry (which repeats the camera pose and lens) stay
    out of the agent's file tools, as the deny rules for robot.yaml and ~/calib intended (public-repo sprint,
    2026-10-13). pi/omp gates grep, find and external reads on the search root only, so no such file may sit under
    any directory those tools allow; Claude Code's agent likewise gets none under its additionalDirectories, and a
    Read deny for each. Codex has no read rules yet (D7)."""
    import json
    import re
    import sys
    from stack.rig.rig import REPO
    sys.path.insert(0, str(REPO / "tools"))
    import setup_agent_ws                                 # the settings as rendered for this machine, from the templates
    rig = load_rig()
    files = [REPO / "robot.yaml", rig.camera_file("ego", "intrinsics_file"), rig.camera_file("ego", "extrinsics_file"),
             REPO / "tools/twin_geometry.generated.yaml"]
    embodiment = REPO / "stack/rig/embodiment.md"

    # Claude Code: deny rules are gitignore-style (`**` crosses `/`, `*` does not)
    def claude_rx(p):
        return re.compile(re.escape(p).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*") + "$")

    # the settings as rendered now, and the file the agent actually loads (if written; it may be stale, e.g. a new
    # folder in the home since: then the doctor warns, but the protection checked here must hold in both)
    sources = {"rendered": setup_agent_ws.render_claude(REPO, Path.home().resolve())["permissions"]}
    if setup_agent_ws.CLAUDE_OUT.exists():
        on_disk = json.loads(setup_agent_ws.CLAUDE_OUT.read_text()).get("permissions", {})
        assert {"deny", "additionalDirectories"} <= set(on_disk), \
            "agent_ws/.agents/settings.json is not a rendered file: run python tools/setup_agent_ws.py"
        sources["on disk"] = on_disk
    for label, settings in sources.items():
        claude = [claude_rx(d[len("Read(/"):-1]) for d in settings["deny"] if d.startswith("Read(//")]
        extra_dirs = [Path(d) for d in settings["additionalDirectories"]] + [REPO / "agent_ws"]   # and the cwd
        assert not any(rx.match(str(embodiment)) for rx in claude), f"{label}: the agent must still read embodiment.md"
        assert any(embodiment.is_relative_to(d) for d in extra_dirs), label
        for f in files:
            assert any(rx.match(str(f)) for rx in claude), f"{label}: Claude Code's agent may read {f}"
            assert not any(f.is_relative_to(d) for d in extra_dirs), f"{label}: {f} is inside an additionalDirectory"

    # pi/omp: per tool, the last matching rule wins; `*` matches anything (also `/`), `?` one character
    pi_sources = {"rendered": json.loads(setup_agent_ws.render_pi(REPO))["permission"]}
    if setup_agent_ws.PI_OUT.exists():
        pi_sources["on disk"] = json.loads(setup_agent_ws.PI_OUT.read_text())["permission"]

    def pi_allows(pi, tool, path):
        verdict = None
        for pat, v in pi[tool].items():
            rx = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pat)
            if re.fullmatch(rx, str(path), re.S):
                verdict = v
        return verdict == "allow"

    for label, pi in pi_sources.items():
        for tool in ("read", "grep", "find", "external_directory_read"):
            assert pi_allows(pi, tool, embodiment.parent if tool != "read" else embodiment), \
                f"{label}: pi {tool} lost embodiment.md"
            for f in files:
                assert not pi_allows(pi, tool, f), f"{label}: pi {tool} may reach {f}"
                if tool != "read":                   # searched from any allowed ancestor directory
                    for d in f.parents:
                        assert not pi_allows(pi, tool, d), f"{label}: pi {tool} rooted at {d} would reach {f}"


def test_the_calibration_files_carry_no_private_paths():
    from stack.rig.rig import REPO
    for f in (REPO / "calib").glob("*.json"):
        text = f.read_text()
        assert "/home/" not in text and "/Users/" not in text, f


def test_a_leftover_rig_yaml_beside_robot_yaml_is_refused(tmp_path):
    """D16: rig.yaml was renamed robot.yaml; two descriptions side by side could disagree, so loading refuses."""
    import shutil

    from stack.rig.rig import REPO, _load
    shutil.copy2(REPO / "robot.yaml", tmp_path / "robot.yaml")
    assert _load.__wrapped__(str(tmp_path / "robot.yaml")).path.name == "robot.yaml"     # uncached
    (tmp_path / "rig.yaml").write_text("version: 1\n")
    with pytest.raises(RigError, match="renamed robot.yaml"):
        _load.__wrapped__(str(tmp_path / "robot.yaml"))
