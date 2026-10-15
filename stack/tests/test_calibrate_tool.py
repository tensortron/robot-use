"""tools/calibrate.py and its steps (calibration sprint 2026-10-14, D7): the robot.yaml editor, the arm step against a
pretend bus, and the lens and camera-pose steps end to end on synthetic checkerboard images from a known camera."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import calib_steps as cs  # noqa: E402
from calib_demo import DemoBus  # noqa: E402

cv2 = pytest.importorskip("cv2")


@pytest.fixture(autouse=True)
def _data_folder(tmp_path, monkeypatch):
    """Never the real data folder: the tool keeps what the servos held, and backups, there."""
    monkeypatch.setenv("ROBOT_USE_HOME", str(tmp_path / "robot-use"))
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "lerobot-calibration"))


# ------------------------------------------------------------------------------------------- robot.yaml editor
@pytest.fixture
def rig_copy(tmp_path):
    p = tmp_path / "robot.yaml"
    shutil.copy2(REPO / "robot.yaml", p)
    return p


def _changed(a: str, b: str) -> list[int]:
    la, lb = a.splitlines(), b.splitlines()
    return [i for i in range(max(len(la), len(lb))) if i >= len(la) or i >= len(lb) or la[i] != lb[i]]


def test_the_rig_editor_changes_only_the_lines_it_sets(rig_copy):
    before = rig_copy.read_text()
    ed = cs.RigEditor(rig_copy)
    ed.set(("arms", "pan_axis_separation_m"), 0.372, "measured 2026-10-14 (tools/calibrate.py, tape)")
    ed.save(backup=False)
    after = rig_copy.read_text()
    changed = _changed(before, after)
    assert len(changed) == 2                                 # the value and its sources line, nothing else
    line = after.splitlines()[changed[0]]
    old = before.splitlines()[changed[0]]
    assert line.startswith("  pan_axis_separation_m: 0.372 ") and line.index("#") == old.index("#")
    raw = yaml.safe_load(after)
    assert raw["arms"]["pan_axis_separation_m"] == 0.372
    assert raw["arms"]["sources"]["pan_axis_separation_m"].startswith("measured 2026-10-14")
    assert raw["arms"]["sources"]["joint_offsets_deg"] == yaml.safe_load(before)["arms"]["sources"]["joint_offsets_deg"]


def test_the_rig_editor_adds_replaces_and_deletes_keys_and_reads_them_back(rig_copy):
    ed = cs.RigEditor(rig_copy)
    cam = ("cameras", "ego")
    ed.delete(cam + ("intrinsics_file",))
    ed.delete(cam + ("extrinsics_file",))
    ed.set(cam + ("eye_m",), [-0.05, 0.0, 0.69])
    ed.set(cam + ("tilt_down_deg",), 30.0, comment="from straight down (60.0 below horizontal)")
    ed.set(cam + ("distortion",), {"model": "opencv_pinhole"})
    ed.set(cam + ("distortion", "focal_px"), [800.0, 801.5])
    ed.set(cam + ("new_key",), "x")                          # a key the entry did not have
    long = ("measured 2026-10-14 - " + "a long provenance note " * 8).strip()
    ed.set(("cameras", "sources", "ego"), long)
    ed.save(backup=False)
    raw = yaml.safe_load(rig_copy.read_text())
    ego = raw["cameras"]["ego"]
    assert "intrinsics_file" not in ego and "extrinsics_file" not in ego
    assert ego["eye_m"] == [-0.05, 0.0, 0.69] and ego["tilt_down_deg"] == 30.0 and ego["new_key"] == "x"
    assert ego["distortion"] == {"model": "opencv_pinhole", "focal_px": [800.0, 801.5]}
    assert raw["cameras"]["sources"]["ego"] == long
    assert all(len(ln) <= 120 for ln in rig_copy.read_text().splitlines() if "a long provenance" in ln)
    assert "60.0 below horizontal" in rig_copy.read_text()
    assert raw["cameras"]["wrist_left"] == yaml.safe_load((REPO / "robot.yaml").read_text())["cameras"]["wrist_left"]


def test_the_rig_editor_refuses_to_write_an_invalid_rig(rig_copy):
    before = rig_copy.read_text()
    ed = cs.RigEditor(rig_copy)
    ed.set(("version",), 2)
    with pytest.raises(Exception, match="version"):
        ed.save()
    assert rig_copy.read_text() == before


# ------------------------------------------------------------------------------------------- the arm step
def _ranged(arm, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not arm.snapshot()["short"] and arm.snapshot()["mins"]:
            return True
        time.sleep(0.05)
    return False


def test_the_arm_step_writes_lerobots_calibration_and_never_turns_torque_on(tmp_path):
    bus = DemoBus(sweep_ticks=800, period_s=0.6)
    out = tmp_path / "bi_follower_right.json"
    arm = cs.ArmCalibration("right", "demo", "bi_follower_right", out, bus_factory=lambda port: bus)
    with pytest.raises(cs.ArmError):
        arm.set_middle()                                     # torque off comes first
    arm.go_limp()
    assert all(bus.regs[j]["Torque_Enable"] == 0 for j in cs.ARM_JOINTS)
    arm.set_middle()
    assert _ranged(arm)
    cal = arm.finish()
    arm.close()
    saved = json.loads(out.read_text())
    assert saved == cal and list(saved) == list(cs.ARM_JOINTS)
    for i, j in enumerate(cs.ARM_JOINTS):
        assert set(saved[j]) == {"id", "drive_mode", "homing_offset", "range_min", "range_max"}
        assert saved[j]["id"] == i + 1 and saved[j]["drive_mode"] == 0
        assert bus.regs[j]["Homing_Offset"] == saved[j]["homing_offset"]          # the servos hold it too
        assert bus.regs[j]["Min_Position_Limit"] == saved[j]["range_min"]
    assert (saved["wrist_roll"]["range_min"], saved["wrist_roll"]["range_max"]) == (0, 4095)
    assert saved["elbow_flex"]["range_max"] - saved["elbow_flex"]["range_min"] > cs.MIN_RANGE_TICKS
    from stack.robotd.calibration import read_joint_limits
    assert read_joint_limits("bi_follower_right", tmp_path) is not None             # robotd reads it


def test_a_trial_or_a_cancel_gives_the_servos_back_what_they_held(tmp_path):
    for how in ("trial", "cancel", "close"):
        bus = DemoBus(sweep_ticks=800, period_s=0.6)
        before = {j: {k: bus.regs[j][k] for k in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")}
                  for j in cs.ARM_JOINTS}
        arm = cs.ArmCalibration("left", "demo", "bi_follower_left", tmp_path / how / "c.json",
                                restore_after=how == "trial", bus_factory=lambda port: bus)
        arm.go_limp()
        arm.set_middle()
        assert any(bus.regs[j]["Homing_Offset"] != before[j]["Homing_Offset"] for j in cs.ARM_JOINTS)
        if how == "trial":
            assert _ranged(arm)
            arm.finish()
            assert (tmp_path / how / "c.json").exists()
        elif how == "cancel":
            arm.cancel()
        arm.close()
        after = {j: {k: bus.regs[j][k] for k in before[j]} for j in cs.ARM_JOINTS}
        assert after == before, how
        assert all(bus.regs[j]["Torque_Enable"] == 0 for j in cs.ARM_JOINTS)


def test_a_joint_not_moved_through_its_range_is_refused(tmp_path):
    bus = DemoBus(sweep_ticks=50, period_s=0.5)              # barely moving
    arm = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: bus)
    arm.go_limp()
    arm.set_middle()
    time.sleep(0.3)
    with pytest.raises(cs.ArmError, match="whole range"):
        arm.finish()
    assert arm.phase == "ranging" and not (tmp_path / "x.json").exists()           # still recording
    arm.close()


# ------------------------------------------------------------------------------------------- synthetic images
COLS, ROWS, SQ = 9, 6, 0.025


def _render_board(rvec, tvec, K, dist, size):
    """A printed 10x7-square board (9x6 inner corners) seen by a camera: board frame -> camera by (rvec, tvec)."""
    w, h = size
    img = np.full((h, w, 3), 235, np.uint8)
    margin = [(-2 * SQ, -2 * SQ), ((COLS + 1) * SQ, -2 * SQ), ((COLS + 1) * SQ, (ROWS + 1) * SQ), (-2 * SQ, (ROWS + 1) * SQ)]
    pts, _ = cv2.projectPoints(np.array([[x, y, 0.0] for x, y in margin]), rvec, tvec, K, dist)
    cv2.fillConvexPoly(img, np.round(pts.reshape(-1, 2) * 16).astype(np.int32), (255, 255, 255), cv2.LINE_AA, 4)
    for i in range(-1, COLS):
        for j in range(-1, ROWS):
            if (i + j) % 2:
                continue
            x0, y0 = i * SQ, j * SQ
            edge = [(x0 + SQ * t, y0) for t in np.linspace(0, 1, 5)] + [(x0 + SQ, y0 + SQ * t) for t in np.linspace(0, 1, 5)] + \
                   [(x0 + SQ * (1 - t), y0 + SQ) for t in np.linspace(0, 1, 5)] + [(x0, y0 + SQ * (1 - t)) for t in np.linspace(0, 1, 5)]
            p, _ = cv2.projectPoints(np.array([[x, y, 0.0] for x, y in edge]), rvec, tvec, K, dist)
            cv2.fillPoly(img, [np.round(p.reshape(-1, 2) * 16).astype(np.int32)], (20, 20, 20), cv2.LINE_AA, 4)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _ego():
    from stack.rig.rig import load_rig
    return load_rig().camera_model("ego")[0]


def _lens_frames(folder: Path, K, dist, size):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    n = 0
    for ax in (-35, -20, 0, 20, 35):
        for ay in (-30, 0, 30):
            R = cv2.Rodrigues(np.radians([ax, 0.0, 0.0]))[0] @ cv2.Rodrigues(np.radians([0.0, ay, 0.0]))[0]
            centre = np.array([(COLS - 1) * SQ / 2, (ROWS - 1) * SQ / 2, 0.0])
            t = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.38, 0.5)]) - R @ centre
            img = _render_board(cv2.Rodrigues(R)[0], t, K, dist, size)
            cv2.imwrite(str(folder / f"frame_{n:03d}.png"), img)
            n += 1
    return n


def _table_photo(model, corner_xy, yaw_deg):
    """The board lying flat on the table: its corner (0, 0) at corner_xy, its X axis toward yaw_deg, face up."""
    psi = math.radians(yaw_deg)
    X = np.array([math.cos(psi), math.sin(psi), 0.0])
    Z = np.array([0.0, 0.0, -1.0])                           # the printed numbering's Z points into the table
    R_wb = np.column_stack([X, np.cross(Z, X), Z])
    p0 = np.array([corner_xy[0], corner_xy[1], 0.0])
    R = model.R_cw @ R_wb
    t = model.R_cw @ p0 + model.t_cw
    return _render_board(cv2.Rodrigues(R)[0], t, model.K, model.dist, model.image_size), R_wb, p0


def _pose_step(cal, render_model, seen_model):
    """Steps through the pose: the photo, then the tape measurement of whichever corner the detector circled and the
    arrow's direction, read off the synthetic table."""
    img, R_wb, p0 = _table_photo(render_model, (0.20, -0.10), 90.0)
    cal.latest_raw = img
    out = cal.act({"do": "pose_capture"})
    assert cal.pose_pass1, out
    px = np.array(cal.pose_pass1["corner_px"], float)
    corners = {(i, j): p0 + R_wb @ np.array([i * SQ, j * SQ, 0.0]) for i in (0, COLS - 1) for j in (0, ROWS - 1)}
    proj = {k: seen_model.project([v])[0][0] for k, v in corners.items()}
    first = min(proj, key=lambda k: np.linalg.norm(proj[k] - px))
    to_px = np.array(cal.pose_pass1["arrow_to_px"], float)
    last = min(proj, key=lambda k: np.linalg.norm(proj[k] - to_px))
    d = corners[last] - corners[first]
    yaw = round(math.degrees(math.atan2(d[1], d[0])) / 90) * 90 % 360
    return cal.act({"do": "pose_solve", "corner_x": corners[first][0], "corner_y": corners[first][1], "yaw": yaw})


def test_the_pose_uses_the_lens_rig_yaml_names(tmp_path):
    """A trial that redoes only the pose solves it with the lens file the rig names (here the repository's), not one
    expected in the trial's own folder (re-review D2)."""
    _, cal = _tool(tmp_path)
    truth = _ego()
    out = _pose_step(cal, truth, truth)
    assert "pose saved" in out["message"], out
    ext = json.loads(Path(cal.rig().camera("ego")["extrinsics_file"]).read_text())
    assert ext["intrinsics"] == "calib/ego_intrinsics.json"
    assert np.linalg.norm(cal.rig().camera_model("ego")[0].eye - truth.eye) < 0.01
    assert not list((tmp_path / "trial" / "calib").glob("*.tmp"))


def test_lens_and_pose_steps_recover_a_known_camera(tmp_path):
    """Steps 4 and 5 through the tool's own actions: the lens from guided-capture frames, then the pose from one photo
    of the board on the table and the tape measurement of the circled corner. The camera they write into robot.yaml
    is the one the images were made with."""
    import calibrate
    truth = _ego()
    K, size = truth.K.copy(), truth.image_size
    a = SimpleNamespace(trial=str(tmp_path / "trial"), rig=str(REPO / "robot.yaml"), calib_dir=None,
                        config=str(REPO / "stack/robotd.yaml"), camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS,
                        demo=True)
    cal = calibrate.Calibrator(a)
    frames = tmp_path / "frames"
    n = _lens_frames(frames, K, np.zeros(5), size)
    cal.guided = SimpleNamespace(kept=n, capture_to=str(frames))
    out = cal.act({"do": "lens_solve"})
    assert "lens saved" in out["message"], out
    rig = cal.rig()
    ego = rig.camera("ego")
    assert Path(ego["intrinsics_file"]).is_file() and "extrinsics_file" not in ego
    assert abs(ego["distortion"]["focal_px"][0] - K[0, 0]) / K[0, 0] < 0.01
    intr = json.loads(Path(ego["intrinsics_file"]).read_text())
    assert "/home/" not in json.dumps(intr["frames_dir"])                       # no machine paths in the file

    # the pose: the board on the table, as the true (calibrated) camera sees it
    lens = rig.camera_model("ego")[0]
    from stack.rig.camera_model import CameraModel
    seen = CameraModel(lens.K, lens.dist, truth.R_cw, truth.t_cw, size)
    out = _pose_step(cal, CameraModel(K, np.zeros(5), truth.R_cw, truth.t_cw, size), seen)
    assert "pose saved" in out["message"], out
    model, src = cal.rig().camera_model("ego")
    assert src == {"lens": "calibrated", "pose": "calibrated"}
    assert np.linalg.norm(model.eye - truth.eye) < 0.01                        # within a centimetre
    grid = np.array([[x, y, 0.0] for x in (0.15, 0.3, 0.4) for y in (-0.2, 0.0, 0.2)])
    assert np.abs(model.project(grid)[0] - CameraModel(K, np.zeros(5), truth.R_cw, truth.t_cw, size).project(grid)[0]).max() < 3.0
    ext = json.loads(Path(cal.rig().camera("ego")["extrinsics_file"]).read_text())
    assert "/home/" not in ext["image"] + ext["intrinsics"]


def test_the_tool_saves_measurements_and_an_estimate_into_a_trial_rig(tmp_path):
    import calibrate
    a = SimpleNamespace(trial=str(tmp_path / "trial"), rig=str(REPO / "robot.yaml"), calib_dir=None,
                        config=str(REPO / "stack/robotd.yaml"), camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS,
                        demo=True)
    cal = calibrate.Calibrator(a)
    v = cal.values()
    meas = {k: str(x) for k, x in v["measurements"].items()}
    meas["pan_axis_separation_m"] = "0.38"
    cal.act({"do": "measurements", "values": meas})
    cal.act({"do": "estimate_save", "values": {"eye_x": "-0.05", "eye_y": "0", "eye_z": "0.7", "tilt_down_deg": "26",
                                               "yaw_deg": "0", "fov_h_deg": "100"}})
    rig = cal.rig()
    assert rig.raw["arms"]["pan_axis_separation_m"] == 0.38
    model, src = rig.camera_model("ego")
    assert src == {"lens": "estimated", "pose": "estimated"}
    assert np.allclose(model.eye, [-0.05, 0.0, 0.7])
    steps = {s["id"]: s["status"] for s in cal.status()["steps"]}
    assert steps["measurements"] == "done" and steps["estimate"] == "estimated" and steps["lens"] == "todo"
    assert (REPO / "robot.yaml").read_text() != (tmp_path / "trial" / "robot.yaml").read_text()   # only the copy changed


# ------------------------------------------------------------------------------------------- review findings
class _FailingBus(DemoBus):
    """Loses the link part-way through lerobot's homing: 3 motors reset, then an error."""
    def set_half_turn_homings(self, motors=None):
        for j in list(cs.ARM_JOINTS)[:3]:
            self.write("Homing_Offset", j, 0)
            self.write("Min_Position_Limit", j, 0)
            self.write("Max_Position_Limit", j, 4095)
        raise ConnectionError("no status packet")


def test_a_homing_that_fails_part_way_is_put_back(tmp_path):
    bus = _FailingBus()
    before = {j: {k: bus.regs[j][k] for k in ("Homing_Offset", "Min_Position_Limit", "Max_Position_Limit")}
              for j in cs.ARM_JOINTS}
    arm = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: bus)
    arm.go_limp()
    with pytest.raises(cs.ArmError, match="homing failed"):
        arm.set_middle()
    after = {j: {k: bus.regs[j][k] for k in before[j]} for j in cs.ARM_JOINTS}
    assert after == before and not arm.touched                                    # restored at once
    # a restore that also fails keeps the arm marked: a new attempt refuses rather than adopting the half-reset state
    bus2 = _FailingBus()
    arm2 = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: bus2)
    arm2.go_limp()
    real_write = bus2.write
    bus2.write = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("12 V gone"))
    with pytest.raises(cs.ArmError):
        arm2.set_middle()
    assert arm2.touched and "FAILED" in arm2.message
    with pytest.raises(cs.ArmError, match="Restore"):
        arm2.go_limp()
    arm2.close()                                       # the page closes the link after a Cancel; Ctrl-C does too
    assert arm2.touched and arm2.bus is None
    bus2.write = real_write
    arm2.cancel()                                      # the link is back: reconnects and restores
    assert not arm2.touched and arm2.phase == "cancelled"
    assert {j: bus2.regs[j]["Homing_Offset"] for j in cs.ARM_JOINTS} == {j: 100 + 10 * i for i, j in enumerate(cs.ARM_JOINTS)}


def test_a_restart_still_restores_what_the_servos_held(tmp_path):
    """The servos' original values are on disk while they may differ from them: a new run (after a crash, or a
    restart before the restore worked) refuses to start over and restores them on Cancel."""
    bus = DemoBus()
    pending = tmp_path / "servos_before_x.json"
    arm = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: bus,
                            pending_path=pending)
    arm.go_limp()
    original = arm.original
    arm.set_middle()
    assert pending.exists()                            # the run "crashes" here: nothing restores
    arm._stop_sampling()
    again = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: bus,
                              pending_path=pending)
    assert again.touched and again.original == original
    with pytest.raises(cs.ArmError, match="Restore"):
        again.go_limp()
    again.cancel()
    assert not again.touched and not pending.exists()
    assert {j: {"homing_offset": bus.regs[j]["Homing_Offset"], "range_min": bus.regs[j]["Min_Position_Limit"],
                "range_max": bus.regs[j]["Max_Position_Limit"]} for j in cs.ARM_JOINTS} == original


def test_the_rig_editor_writes_small_numbers_yaml_reads_as_numbers(rig_copy):
    ed = cs.RigEditor(rig_copy)
    ed.set(("cameras", "ego", "distortion", "coeffs"), [-0.41, 0.18, 4e-05, -7e-06, 0.0])
    ed.save(backup=False)
    raw = yaml.safe_load(rig_copy.read_text())
    assert raw["cameras"]["ego"]["distortion"]["coeffs"] == [-0.41, 0.18, 4e-05, -7e-06, 0.0]


def test_rig_backups_go_to_the_data_folder_not_beside_rig_yaml(rig_copy, tmp_path):
    ed = cs.RigEditor(rig_copy)
    ed.set(("board", "thickness_m"), 0.019)
    ed.save(backup_dir=tmp_path / "backups")
    assert not list(rig_copy.parent.glob("robot.yaml.bak-*"))
    assert len(list((tmp_path / "backups").glob("robot.yaml.bak-*"))) == 1
    from stack import paths
    assert paths.calibration() in paths.all_dirs()                       # denied to the operating agent


def _tool(tmp_path):
    import calibrate
    a = SimpleNamespace(trial=str(tmp_path / "trial"), rig=str(REPO / "robot.yaml"), calib_dir=None,
                        config=str(REPO / "stack/robotd.yaml"), camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS,
                        demo=True)
    return calibrate, calibrate.Calibrator(a)


def test_implausible_numbers_are_refused_and_nothing_is_written(tmp_path):
    _, cal = _tool(tmp_path)
    rig = tmp_path / "trial" / "robot.yaml"
    before = rig.read_text()
    meas = {k: str(x) for k, x in cal.values()["measurements"].items()}
    for bad in ("371.5", "nan", "inf", "", "x"):                      # mm typed for m; not numbers
        with pytest.raises(ValueError, match="Pan axis to pan axis"):
            cal.act({"do": "measurements", "values": {**meas, "pan_axis_separation_m": bad}})
    est = {"eye_x": "-0.05", "eye_y": "0", "eye_z": "700", "tilt_down_deg": "26", "yaw_deg": "0", "fov_h_deg": "100"}
    with pytest.raises(ValueError, match="height"):
        cal.act({"do": "estimate_save", "values": est})
    assert rig.read_text() == before


def test_the_page_refuses_requests_from_other_pages(tmp_path):
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    calibrate, cal = _tool(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), calibrate.make_handler(cal))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def post(headers):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("POST", "/action", body=json.dumps({"do": "view"}), headers=headers)
        r = c.getresponse()
        return r.status
    try:
        assert post({"Content-Type": "text/plain"}) == 403                     # a "simple" cross-site request
        assert post({"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
        assert post({"Content-Type": "application/json", "Host": "evil.example"}) == 403       # DNS rebinding
        assert post({"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"}) == 200
    finally:
        srv.shutdown()


def test_the_page_offers_the_restore_after_a_restart_and_a_demo_keeps_to_itself(tmp_path):
    """Third review: a pending restore must be reachable from the page (status "restore", the arm's snapshot says
    touched), and a demo uses pretend servos and its own folder - never the real pending files or serial ports."""
    import calibrate
    from stack import paths
    a = SimpleNamespace(trial=None, rig=str(REPO / "robot.yaml"), calib_dir=None, config=str(REPO / "stack/robotd.yaml"),
                        camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS, demo=True)
    cal = calibrate.Calibrator(a)
    assert cal.trial and str(cal.trial).startswith(str(paths.calibration()))          # a demo implies a trial
    pend = cal.pending_path("right")
    assert pend.parent == cal.home / "demo" and pend.parent != paths.calibration()
    pend.parent.mkdir(parents=True, exist_ok=True)
    real = paths.calibration() / pend.name
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("{}")                                                          # the real one: untouched by a demo
    original = {j: {"homing_offset": 100 + 10 * i, "range_min": 700 + i, "range_max": 3400 - i}
                for i, j in enumerate(cs.ARM_JOINTS)}
    pend.write_text(json.dumps(original))
    cal2 = calibrate.Calibrator(SimpleNamespace(**{**a.__dict__, "trial": str(cal.trial)}))
    arm = cal2.arms["right"]
    assert arm.bus_factory is DemoBus and arm.touched
    steps = {s["id"]: s for s in cal2.status()["steps"]}
    assert steps["arm_right"]["status"] == "restore" and steps["arm_right"]["arm"]["touched"]
    with pytest.raises(cs.ArmError, match="Restore"):
        cal2.act({"do": "arm_limp", "arm": "right"})
    out = cal2.act({"do": "arm_cancel", "arm": "right"})
    assert "previous calibration back" in out["message"] and not pend.exists()
    assert real.read_text() == "{}"


def test_an_unreadable_record_of_the_servos_is_reported_not_guessed(tmp_path):
    pend = tmp_path / "servos_before_x.json"
    for bad in ("", "null", "{}", '{"shoulder_pan": 1}'):
        pend.write_text(bad)
        arm = cs.ArmCalibration("left", "demo", "x", tmp_path / "x.json", bus_factory=lambda port: DemoBus(),
                                pending_path=pend)
        assert arm.touched and arm.original is None and "fix or delete" in arm.message
        arm.cancel()
        assert arm.touched and "fix or delete" in arm.message and pend.exists()      # nothing written, still blocked


def _cal_file(path: Path, shift=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({j: {"id": i + 1, "drive_mode": 0, "homing_offset": 10 * i + shift, "range_min": 700,
                                    "range_max": 3400} for i, j in enumerate(cs.ARM_JOINTS)}, indent=4))
    return path


def test_an_arm_calibrated_before_is_used_as_it_is(tmp_path):
    """Task 2c: lerobot files under other ids or robot folders are found; "Use" points robot.yaml at one (copying it
    where robotd reads), writes nothing to the servos, and refuses one id for two arms or a clashing file."""
    root = tmp_path / "lerobot-calibration"
    mine = _cal_file(root / "robots" / "so101_follower" / "my_left.json")
    _cal_file(root / "robots" / "so_follower" / "other.json", shift=5)
    (root / "robots" / "so_follower" / "notes.json").write_text('{"not": "a calibration"}')
    _cal_file(root / "teleoperators" / "so_leader" / "leader.json")           # a leader arm: not offered
    bad = root / "robots" / "so_follower"                                    # none of these may break the page
    (bad / "inf.json").write_text(json.dumps({j: {"homing_offset": float("inf"), "range_min": 1, "range_max": 2}
                                              for j in cs.ARM_JOINTS}))
    (bad / "deep.json").write_text("[" * 100_000 + "]" * 100_000)
    (bad / "flipped.json").write_text(json.dumps({j: {"homing_offset": 0, "range_min": 4000, "range_max": 10}
                                                  for j in cs.ARM_JOINTS}))
    (bad / "big.json").write_text(" " * 70_000 + "{}")
    _, cal = _tool(tmp_path)
    steps = {st["id"]: st for st in cal.status()["steps"]}
    ids = {e["id"] for e in steps["arm_left"]["existing"]}
    assert ids == {"my_left", "other"}
    out = cal.act({"do": "arm_use", "arm": "left", "path": str(mine)})
    assert "uses" in out["message"]
    assert cal.rig().arm("left")["calibration_id"] == "my_left"
    assert (cal.arm_dir / "my_left.json").read_bytes() == mine.read_bytes()
    assert {st["id"]: st["status"] for st in cal.status()["steps"]}["arm_left"] == "done"
    with pytest.raises(ValueError, match="already uses"):
        cal.act({"do": "arm_use", "arm": "right", "path": str(mine)})
    cal.act({"do": "arm_use", "arm": "right", "path": str(root / "robots" / "so_follower" / "other.json")})
    clash = _cal_file(root / "robots" / "koch_follower" / "other.json", shift=9)    # same name, other contents
    with pytest.raises(ValueError, match="different calibration"):
        cal.act({"do": "arm_use", "arm": "right", "path": str(clash)})
    with pytest.raises(ValueError, match="listed"):
        cal.act({"do": "arm_use", "arm": "right", "path": "/etc/passwd"})


def test_the_camera_steps_need_only_the_camera(tmp_path, monkeypatch):
    calibrate, cal = _tool(tmp_path)
    monkeypatch.setattr(calibrate, "ROBOTD_PORTS", ())              # no robotd ports probed in a test
    cal.a.demo = False
    cam = tmp_path / "cam_ego"
    cam.write_text("")
    cal.cam_dev = str(cam)
    for c in cal.cfg.arms.values():
        c.port = str(tmp_path / f"absent_{c.name}")
    assert cal.preflight(camera_only=True) == []
    assert any("absent_left" in p for p in cal.preflight())


def test_a_new_rig_resets_the_joint_offsets(tmp_path):
    _, cal = _tool(tmp_path)
    meas = {k: str(x) for k, x in cal.values()["measurements"].items()}
    assert cal.values()["offsets"]["left"]["joint_offsets_deg"] != [0, 0, 0, 0, 0]       # this rig's, inherited
    cal.act({"do": "measurements", "values": {**meas, "new_rig": True}})
    raw = cal.rig().raw["arms"]
    for n in ("left", "right"):
        assert raw[n]["joint_offsets_deg"] == [0.0] * 5 and raw[n]["base_z_offset_m"] == 0.0
    assert raw["sources"]["joint_offsets_deg"].startswith("reset to 0")


def test_a_rig_other_than_robotds_is_refused_outside_a_trial(tmp_path):
    """The arms' calibration ids come from robot.yaml: the tool and robotd must read the same one."""
    import calibrate
    other = tmp_path / "other_robot.yaml"
    shutil.copy2(REPO / "robot.yaml", other)
    a = SimpleNamespace(trial=None, rig=str(other), calib_dir=str(tmp_path / "calib"),
                        config=str(REPO / "stack/robotd.yaml"), camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS,
                        demo=False)
    with pytest.raises(SystemExit, match="robotd's config"):
        calibrate.Calibrator(a)


def test_camera_only_leaves_the_arms_alone(tmp_path, monkeypatch):
    """Task 2d: --camera-only shows no arm steps, refuses every arm action, checks the camera alone, and only reports
    an arm left needing a restore (restoring it would open the arm's port)."""
    import calibrate
    a = SimpleNamespace(trial=str(tmp_path / "trial"), rig=str(REPO / "robot.yaml"), calib_dir=None,
                        config=str(REPO / "stack/robotd.yaml"), camera="ego", squares_mm=25.0, cols=COLS, rows=ROWS,
                        demo=True, camera_only=True)
    pending = calibrate.Calibrator(a).pending_path("left")
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(json.dumps({j: {"homing_offset": 0, "range_min": 1, "range_max": 2} for j in cs.ARM_JOINTS}))
    cal = calibrate.Calibrator(a)
    opened = []
    cal.arm_bus_factory = lambda port: opened.append(port)
    st = cal.status()
    assert st["camera_only"] and not [s for s in st["steps"] if s["id"].startswith("arm_")]
    assert not cal.arms and any("without --camera-only" in m["text"] for m in st["messages"])
    for do in ("arm_limp", "arm_middle", "arm_finish", "arm_cancel", "arm_use"):
        with pytest.raises(ValueError, match="camera-only"):
            cal.act({"do": do, "arm": "left", "path": "x"})
    assert opened == [] and pending.exists()                   # no port opened, the record left for a full run
    monkeypatch.setattr(calibrate, "ROBOTD_PORTS", ())
    cal.a.demo = False
    for c in cal.cfg.arms.values():
        c.port = str(tmp_path / f"absent_{c.name}")
    cam = tmp_path / "cam"
    cam.write_text("")
    cal.cam_dev = str(cam)
    assert cal.status()["preflight"] == []                     # the arms' absent ports do not block
