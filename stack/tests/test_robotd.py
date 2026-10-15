"""robotd step 2 (read-only) against mock hardware: config, loop, faults, HTTP, camera model."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from stack.rig.rig import load_rig
from stack.robotd.config import load_config
from stack.robotd.daemon import EventLog, RobotD
from stack.robotd.hardware import MockArmIO, MockCameraIO, device_holders
from stack.robotd.http import serve
from stack.robotd.safety import ArmLimits

from .conftest import CALIBRATED

REPO_CONFIG = Path(__file__).resolve().parents[1] / "robotd.yaml"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_repo_config_loads():
    cfg = load_config(REPO_CONFIG)
    assert set(cfg.arms) == {"left", "right"} and cfg.motion_enabled is False
    assert cfg.api.host == "127.0.0.1"
    assert set(cfg.cameras) == {"ego", "wrist_left", "wrist_right"}


@pytest.mark.parametrize("patch, msg", [
    ("api: {host: 0.0.0.0, port: 8770}", "localhost"),
    ("approval_mode: sometimes", "approval_mode"),
    ("safety: {no_such_key: 1}", "unknown keys"),
    ("safety: {hull_margins_m: {floor: {camera: 0}}}", "0 < m"),
    ("safety: {hull_margins_m: {floor: {camera: -0.01}}}", "0 < m"),
    ("safety: {hull_margins_m: {floor: {fingers: 0.01}}}", "parts are"),
    ("safety: {hull_margins_m: {ceiling: {camera: 0.01}}}", "targets are"),
    ("safety: {rest_roll_deg: 90}", "0 or -90"),
])
def test_config_rejects_unsafe_or_unknown(tmp_path, patch, msg):
    text = REPO_CONFIG.read_text()
    key = patch.split(":")[0]
    lines, skip = [], False
    for l in text.splitlines():                  # drop the whole top-level block for this key, children too
        if l.startswith(key + ":"):
            skip = True
            continue
        if skip and (l.startswith(" ") or not l.strip()):
            continue
        skip = False
        lines.append(l)
    p = tmp_path / "c.yaml"
    p.write_text("\n".join(lines) + "\n" + patch + "\n")
    with pytest.raises(ValueError, match=msg):
        load_config(p)


def test_partial_hull_margins_merge_with_the_defaults(tmp_path):
    text = REPO_CONFIG.read_text()
    lines, skip = [], False
    for l in text.splitlines():
        if l.startswith("safety:"):
            skip = True
            continue
        if skip and (l.startswith(" ") or not l.strip()):
            continue
        skip = False
        lines.append(l)
    p = tmp_path / "c.yaml"
    p.write_text("\n".join(lines) + "\nsafety: {hull_margins_m: {floor: {camera: 0.02}}}\n")
    m = load_config(p).safety.hull_margins_m
    assert m["floor"] == {"camera": 0.02, "jaws": 0.002, "gripper": 0.010}
    assert m["other_arm"]["jaws"] == 0.020


def test_config_refuses_geometry_outside_rig_yaml(tmp_path):
    text = REPO_CONFIG.read_text()
    for bad, msg in (("safety: {board_front_x: 0.1}", "robot.yaml"),
                     ("arms:\n  left: {port: /dev/x, mount: {signs: [1,1,1,1,1]}}\n  right: {port: /dev/y}", "robot.yaml")):
        key = bad.split(":")[0]
        lines = text.splitlines()
        out, skip = [], False
        for line in lines:                       # drop the existing top-level block for this key
            if line.startswith(key + ":"):
                skip = True
                continue
            if skip and (line.startswith(" ") or not line.strip()):
                continue
            skip = False
            out.append(line)
        p = tmp_path / "c.yaml"
        p.write_text("\n".join(out) + "\n" + bad + "\n")
        with pytest.raises(ValueError, match=msg):
            load_config(p)


@pytest.fixture
def robotd(tmp_path):
    cfg = load_config(REPO_CONFIG)
    cfg.log_dir = tmp_path
    cfg.overlay = None
    arms = {n: MockArmIO(n) for n in cfg.arms}
    cams = {n: MockCameraIO(n, c.width, c.height) for n, c in cfg.cameras.items()}
    limits = {n: ArmLimits(dict(CALIBRATED[n])) for n in cfg.arms}
    rd = RobotD(cfg, arms, cams, limits=limits, event_log=EventLog(tmp_path))
    rd.start()
    yield rd
    rd.stop()


def _wait(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_loop_runs_at_rate_and_reports_state(robotd):
    assert _wait(lambda: robotd.ticks >= 60)
    time.sleep(1.0)
    s = robotd.state()
    assert s["motion_enabled"] is False and s["supervisor"]["armed"] is False
    assert 45 <= s["loop"]["measured_hz"] <= 55
    for arm in ("left", "right"):
        a = s["arms"][arm]
        assert a["connected"] and a["fault"] is None
        assert set(a["joints"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"}
        assert len(a["tip_xyz_m"]) == 3 and a["age_s"] < 0.2


def test_read_failures_fault_then_recover(robotd):
    io = robotd.arm_ios["left"]
    io.torque_off()                      # an unpowered arm's bus hiccup clears by itself...
    robotd.arms["left"].torque = [0] * 6
    io.fail_reads = 5
    assert _wait(lambda: robotd.arms["left"].fault is not None)
    assert robotd.arms["left"].fault.startswith("bus:")
    assert _wait(lambda: robotd.arms["left"].fault is None)
    events = [json.loads(l)["event"] for l in robotd.events.path.read_text().splitlines()]
    assert "arm_fault" in events and "arm_recovered" in events


def test_firmware_limit_mismatch_is_a_fault(tmp_path):
    cfg = load_config(REPO_CONFIG)
    cfg.log_dir, cfg.overlay, cfg.cameras = tmp_path, None, {}
    arms = {n: MockArmIO(n) for n in cfg.arms}
    arms["right"].limits_match = False
    rd = RobotD(cfg, arms, {}, limits={n: ArmLimits(dict(CALIBRATED[n])) for n in cfg.arms}, event_log=EventLog(tmp_path))
    rd.start()
    try:
        assert "firmware position limits differ" in rd.arms["right"].fault
        assert rd.arms["left"].fault is None
    finally:
        rd.stop()


def test_limits_include_rest_pose(robotd):
    lim = robotd.limits()
    for arm in ("left", "right"):
        rp = lim["arms"][arm]["rest_pose"]
        assert rp["shoulder_pan"] == 0 and rp["gripper"] == 2.0 and rp["wrist_flex"] <= 70


@pytest.fixture
def servers(robotd):
    api_port, op_port = _free_port(), _free_port()
    api = serve(robotd, "127.0.0.1", api_port, operator=False)
    op = serve(robotd, "127.0.0.1", op_port, operator=True)
    yield f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{op_port}"
    api.shutdown()
    op.shutdown()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_routes(servers):
    api, _ = servers
    code, _, body = _get(api + "/state")
    assert code == 200 and "arms" in json.loads(body)
    code, h, body = _get(api + "/frame/ego?max_side=256&quality=60")
    assert code == 200 and h["Content-Type"] == "image/jpeg" and body[:3] == b"\xff\xd8\xff"
    import cv2
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 256
    assert _get(api + "/frame/nope")[0] == 404
    assert _get(api + "/frame/ego?max_side=99999")[0] == 400
    # operator-only routes are not on the API side
    assert _get(api + "/")[0] == 404
    assert _post(api + "/torque_off/left", {"confirm": True})[0] == 404


def test_operator_routes(servers, robotd):
    _, op = servers
    code, _, body = _get(op + "/")
    assert code == 200 and b"robotd" in body
    code, body = _post(op + "/torque_off/left", {})
    assert code == 400 and "confirm" in body["error"]
    assert robotd.arm_ios["left"].torque == [1] * 6
    code, body = _post(op + "/torque_off/left", {"confirm": True})
    assert code == 200 and body["ok"] and body["torque"] == [0] * 6
    assert robotd.arm_ios["right"].torque == [1] * 6, "torque off must only touch the named arm"
    assert _post(op + "/torque_off/middle", {"confirm": True})[0] == 404
    code, body = _post(op + "/stop", {})
    assert code == 200 and body["ok"]


def test_device_holders_sees_another_process(tmp_path):
    f = tmp_path / "held"
    f.write_text("x")
    p = subprocess.Popen([sys.executable, "-c", f"f=open({str(f)!r}); import time; time.sleep(30)"])
    try:
        assert _wait(lambda: str(f) in device_holders([str(f)]), timeout=5)
        assert p.pid in device_holders([str(f)])[str(f)]
    finally:
        p.kill()
    assert device_holders([str(tmp_path / "missing")]) == {}


def test_camera_model_reproduces_calibration():
    from stack.rig.camera_model import CameraModel

    rig = load_rig()
    intr, ext = rig.camera_file("ego", "intrinsics_file"), rig.camera_file("ego", "extrinsics_file")
    cam = CameraModel.from_files(intr, ext)
    e = json.loads(ext.read_text())
    assert np.allclose(cam.eye, e["eye_m"], atol=1e-4)
    px, front = cam.project([[e["board"]["corner_xy"][0], e["board"]["corner_xy"][1], 0.0]])
    assert front[0] and np.linalg.norm(px[0] - e["board"]["corner_px"]) < 2.0


def test_read_failures_with_torque_on_are_sticky(robotd):
    """...but on a powered arm it needs an operator reset (it may have been bumped or lost power)."""
    robotd.arm_ios["right"].fail_reads = 5
    assert _wait(lambda: robotd.arms["right"].fault is not None)
    assert _wait(lambda: robotd.arms["right"].failed_reads == 0)
    time.sleep(0.2)
    assert robotd.arms["right"].sticky_fault and robotd.arms["right"].fault
    assert robotd.reset_arm("right")["ok"] and robotd.arms["right"].fault is None


def test_overlay_draws_static_geometry():
    from stack.rig.camera_model import CameraModel, draw_static
    rig = load_rig()
    intr, ext = rig.camera_file("ego", "intrinsics_file"), rig.camera_file("ego", "extrinsics_file")
    cam = CameraModel.from_files(intr, ext)
    f = np.zeros((720, 1280, 3), np.uint8)
    draw_static(f, cam, ({"name": "camera stand", "min": [-0.115, -0.0325, 0.018], "max": [-0.055, 0.0125, 0.700]},),
                (0.025, 0.24975, 0.018))
    assert (f[..., 1] > 0).sum() > 500, "the board outline was drawn"



class _FakeBus:
    """Records the order of bus operations; holds stale goals like a real servo."""

    def __init__(self):
        self.calls, self.torque, self.goal = [], False, {j: 1000 for j in
                                                         ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                          "wrist_roll", "gripper")}
        self.present = {j: 2048 for j in self.goal}
        self.port_handler = None

    def disable_torque(self, *a, **k):
        self.calls.append("torque_off"); self.torque = False

    def enable_torque(self, *a, **k):
        assert self.goal == self.present, "torque enabled while goals are stale: the arm would jump"
        self.calls.append("torque_on"); self.torque = True

    def configure_motors(self):
        self.calls.append("configure")

    def write(self, name, motor, value, **k):
        self.calls.append(f"write:{name}")

    def sync_read(self, name, normalize=True, **k):
        assert name == "Present_Position" and normalize is False
        self.calls.append("read_present")
        return dict(self.present)

    def sync_write(self, name, values, normalize=True, **k):
        assert name == "Goal_Position" and normalize is False
        self.calls.append("write_goal"); self.goal = dict(values)


@pytest.mark.hardware
def test_first_motion_never_enables_torque_with_stale_goals():
    """Regression, rig 2026-10-03: configuring re-enabled torque with the servos' old goals; the wrist jumped."""
    from stack.robotd.hardware import ArmIO
    io = ArmIO("left", "/dev/null", "x")
    io.bus = _FakeBus()
    io.configure_for_motion()
    assert "torque_on" not in io.bus.calls and io.bus.torque is False
    io.hold_here()
    c = io.bus.calls
    assert c[-3:] == ["read_present", "write_goal", "torque_on"], c


def test_cross_site_posts_are_refused(servers):
    """Review 2026-10-03: a page elsewhere could POST a form to the operator's /arm."""
    _, op = servers
    for headers in ({"Content-Type": "application/x-www-form-urlencoded"}, {"Content-Type": "text/plain"}, {}):
        req = urllib.request.Request(op + "/arm", data=b"x=1", method="POST", headers=headers)
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError(f"accepted {headers}")
        except urllib.error.HTTPError as e:
            assert e.code == 415
    req = urllib.request.Request(op + "/arm", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("accepted a foreign Origin")
    except urllib.error.HTTPError as e:
        assert e.code == 403
    code, _ = _post(op + "/arm", {})
    assert code == 200


def test_dns_rebinding_host_is_refused(servers):
    _, op = servers
    req = urllib.request.Request(op + "/arm", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Host": "evil.example:8772",
                                          "Origin": "http://evil.example:8772"})
    try:
        urllib.request.urlopen(req, timeout=5)
        raise AssertionError("accepted a rebound host name")
    except urllib.error.HTTPError as e:
        assert e.code == 403


def test_the_twin_preview_takes_robotds_limits_and_needs_no_calibration_files(servers, monkeypatch, tmp_path):
    """A machine with no calibrated arms (packaging sprint 2026-10-13): the preview checks against the running
    robotd's joint ranges instead of failing on missing calibration files."""
    pytest.importorskip("mujoco")
    from stack.twin_mujoco.preview import robotd_hard_limits, twin_preview
    from stack.twin_mujoco.scene import assets_dir
    api, _ = servers
    lim = robotd_hard_limits(api)
    assert lim == {a: {j: pytest.approx([v[0] + 0.05, v[1] - 0.05]) for j, v in CALIBRATED[a].items()}
                   for a in CALIBRATED}                                       # moved in by half the rounding
    assert robotd_hard_limits(f"http://127.0.0.1:{_free_port()}") is None
    try:
        assets_dir()
    except Exception:
        pytest.skip("no MuJoCo assets (tools/fetch_so101_assets.py)")
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "none"))     # no calibration files at all
    code = 'from robot_use import arm\narm("left").open(60)\n'
    rep = twin_preview(code)
    assert rep["ok"] is False and "no calibrated limits" in rep["reason"], rep  # as before: refuses to guess
    rep = twin_preview(code, robotd_url=api)
    assert rep["ok"] and rep["status"] == "done", rep
    from stack.robotd.safety import ArmLimits                                 # a one-arm robotd: a one-arm twin
    from stack.twin_mujoco.preview import preview_program
    one = {"right": ArmLimits({j: tuple(v) for j, v in lim["right"].items()})}
    rep = preview_program('from robot_use import arm\narm("right").open(60)\n', limits=one, work_dir=tmp_path / "w")
    assert rep["ok"] and rep["status"] == "done", rep


# ---------------------------------------------------------------- the operator page's data (sprint 2026-10-14)
@pytest.fixture
def unattended(tmp_path):
    cfg = load_config(REPO_CONFIG)
    cfg.log_dir, cfg.overlay, cfg.cameras = tmp_path, None, {}
    cfg.motion_enabled, cfg.approval_mode, cfg.mcp_localhost_only = True, "none", True
    arms = {n: MockArmIO(n) for n in cfg.arms}
    rd = RobotD(cfg, arms, {}, limits={n: ArmLimits(dict(CALIBRATED[n])) for n in cfg.arms},
                event_log=EventLog(tmp_path))
    rd.start()
    yield rd
    rd.stop()


def test_program_progress_and_history_for_the_operator_page(unattended):
    rd = unattended
    api, op = (f"http://127.0.0.1:{p}" for p in (_free_port(), _free_port()))
    a_srv = serve(rd, "127.0.0.1", int(api.rsplit(":", 1)[1]), operator=False)
    o_srv = serve(rd, "127.0.0.1", int(op.rsplit(":", 1)[1]), operator=True)
    try:
        assert rd.request_program("p1", "test", "pick", "x = 1")["status"] == "approved"
        lease = rd.acquire_lease("test", 30)["token"]
        assert rd.start_program("p1", lease)["ok"]
        # progress: only the running program's own lease; bounded fields; unknown fields refused
        code, r = _post(api + "/program/progress", {"program_id": "p1", "lease": lease, "step": 1, "kind": "move_joints"})
        assert code == 200 and r["ok"]
        code, r = _post(api + "/program/progress", {"program_id": "p1", "lease": "wrong", "checkpoint": "x"})
        assert not r["ok"]
        code, r = _post(api + "/program/progress", {"program_id": "p1", "lease": lease, "motors": "on"})
        assert code == 400
        code, r = _post(op + "/program/progress", {"program_id": "p1", "lease": lease, "step": 2})
        assert code == 404                                            # not on the operator (LAN) side
        # its motions are recorded under it: a finished one and a refused one
        m = rd.move_joints("left", {"shoulder_pan": 5}, speed=1.0, lease=lease)
        assert _wait(lambda: rd.motion_status(m["motion_id"])["status"] == "done", 10)
        assert rd.move_joints("left", {"elbow_flex": 150}, lease=lease)["status"] == "rejected"
        _post(api + "/program/progress", {"program_id": "p1", "lease": lease, "step": 3, "checkpoint": "lifted"})
        code, _, body = _get(op + "/history")
        h = json.loads(body)
        assert code == 200 and h[0]["program_id"] == "p1" and h[0]["status"] == "running"
        assert h[0]["progress"]["checkpoint"] == "lifted" and h[0]["started"]
        kinds = [(x["kind"], x["status"]) for x in h[0]["motions"]]
        assert ("move_joints", "done") in kinds and ("move_joints", "rejected") in kinds
        assert "code" not in h[0]
        assert _get(api + "/history")[0] == 404                       # the operator side only
        assert rd.end_program("p1", lease, "done: test")["ok"]
        h = json.loads(_get(op + "/history")[2])
        assert h[0]["status"] == "ended" and h[0]["reason"] == "done: test" and h[0]["progress"] == {}
        # bounded: the oldest finished programs are forgotten; the page sees the newest HISTORY_SHOWN
        for i in range(rd.PROGRAMS_KEPT + 10):
            rd.request_program(f"q{i}", "test", "t", "x = 1")
            rd.start_program(f"q{i}", lease)
            rd.end_program(f"q{i}", lease, "done")
        assert len(rd.supervisor.programs) <= rd.PROGRAMS_KEPT + 1
        h = json.loads(_get(op + "/history")[2])
        assert len(h) == rd.HISTORY_SHOWN and h[0]["program_id"] == f"q{rd.PROGRAMS_KEPT + 9}"
    finally:
        a_srv.shutdown()
        o_srv.shutdown()


def test_the_operator_page_keeps_its_safety_controls():
    """STOP (button and Escape), torque off only with confirmation, reset, ARM and mode stay wired; the page can't
    energise an arm and fetches nothing beyond robotd."""
    page = (Path(__file__).resolve().parents[1] / "robotd" / "operator.html").read_text()
    for needed in ('post("/stop")', 'e.key === "Escape"', 'onclick="stopAll()"', "/torque_off/${arm}`, {confirm:true}",
                   "post(`/reset/${arm}`)", 'post("/arm")', 'post("/mode"', "/program/${pid}/${approve"):
        assert needed in page, needed
    assert "torque_on" not in page and "hold_here" not in page
    assert "http://" not in page and "https://" not in page                 # no CDN, no other host
    import re
    assert not re.search(r'on[a-z]+="[^"]*\$\{', page)                      # no handler built from server strings


def test_history_is_cached_bounded_and_complete_on_request(unattended):
    rd = unattended
    lease = rd.acquire_lease("test", 30)["token"]
    assert not rd.request_program("bad id'><img src=x>", "t", "t", "x = 1")["ok"]      # shown on the page: plain ids
    for i in range(5):
        rd.request_program(f"h{i}", "test", f"t{i}", "x = 1")
        rd.start_program(f"h{i}", lease)
        rd.move_joints("left", {"elbow_flex": 150}, lease=lease)                    # refused: filed under h{i}
        rd.end_program(f"h{i}", lease, "done")
    a, b = rd.history_json(), rd.history_json()
    assert a is b                                                               # served from cache until a change
    h = json.loads(a)
    assert [p["program_id"] for p in h[:3]] == ["h4", "h3", "h2"]
    assert all(p["motions"] for p in h[:rd.HISTORY_WITH_MOTIONS])
    assert all(p["motions"] is None and p["n_motions"] == 1 for p in h[rd.HISTORY_WITH_MOTIONS:])
    full = rd.program_history("h0")
    assert full["ok"] and full["motions"][0]["status"] == "rejected"
    rd.request_program("h5", "test", "t", "x = 1")
    assert rd.history_json() is not a                                           # a change: serialised afresh


def test_every_change_a_program_sees_refreshes_the_cached_history(unattended):
    rd = unattended
    lease = rd.acquire_lease("test", 30)["token"]
    rd.request_program("c1", "test", "t", "x = 1")
    rd.start_program("c1", lease)

    def changed(step):                                   # the cache must not outlive a change
        before = rd.history_json()
        step()
        return rd.history_json() is not before
    assert changed(lambda: rd.program_progress("c1", lease, step=1, kind="move_joints"))
    assert changed(lambda: rd.move_joints("left", {"elbow_flex": 150}, lease=lease))           # a refused step
    m = {}
    assert changed(lambda: m.update(rd.move_joints("left", {"shoulder_pan": 2}, lease=lease))
                   or _wait(lambda: rd.motion_status(m["motion_id"])["status"] not in
                            ("pending_approval", "starting", "running"), 10))                # a motion filed
    h = json.loads(rd.history_json())[0]
    assert [r["status"] for r in h["motions"]][0] == "rejected" and len(h["motions"]) == 2
    assert "age_s" not in h                              # cached: no age that would go stale
    recent = rd.state()["supervisor"]["recent"]
    assert any(r.get("program") == "c1" for r in recent)  # the page tells a program's motions from others


def test_a_lease_that_expires_ends_its_program_with_its_stopped_motions(unattended):
    rd = unattended
    import threading
    lease = rd.acquire_lease("test", 0.5)["token"]
    renewing = threading.Event()
    renewing.set()

    def renew():                                     # as a runner does, until the motion runs
        while renewing.is_set():
            rd.renew_lease(lease)
            time.sleep(0.1)
    threading.Thread(target=renew, daemon=True).start()
    rd.request_program("e1", "test", "t", "x = 1")
    assert rd.start_program("e1", lease)["ok"]
    m = rd.move_joints("left", {"shoulder_pan": 60}, speed=0.1, lease=lease)
    assert _wait(lambda: rd.motion_status(m["motion_id"])["status"] == "running", 5), m
    renewing.clear()                                 # the runner stops renewing: the lease expires mid-motion
    assert _wait(lambda: rd.program_history("e1")["status"] == "ended", 5)
    p = rd.program_history("e1")
    assert p["reason"] == "lease expired" and p["ended"] and p["progress"] == {}
    assert [x["status"] for x in p["motions"] if x["motion_id"] == m["motion_id"]] == ["stopped"]


def test_robotd_says_when_its_camera_is_an_estimate(tmp_path):
    """Calibration sprint 2026-10-14, D2: a camera built from robot.yaml's numbers is labelled on /state (the
    operator page's System card shows it as a warning); a calibrated one is not."""
    import yaml as _yaml

    from stack.rig.rig import REPO, Rig
    from stack.robotd.config import OverlayConfig
    raw = _yaml.safe_load((REPO / "robot.yaml").read_text())
    for k in ("intrinsics_file", "extrinsics_file"):
        raw["cameras"]["ego"].pop(k)
    for rig_, expect in ((Rig(raw, tmp_path / "robot.yaml"), "estimated"), (None, "calibrated")):
        cfg = load_config(REPO_CONFIG)
        cfg.log_dir, cfg.overlay = tmp_path, OverlayConfig("ego")
        model, src = (rig_ or cfg.rig).camera_model("ego")
        arms = {n: MockArmIO(n) for n in cfg.arms}
        cams = {n: MockCameraIO(n, c.width, c.height) for n, c in cfg.cameras.items()}
        rd = RobotD(cfg, arms, cams, model, limits={n: ArmLimits(dict(CALIBRATED[n])) for n in cfg.arms},
                    event_log=EventLog(tmp_path))
        st = rd.state()
        assert st["cameras"]["ego"]["model"] == {"lens": expect, "pose": expect}
        assert any("estimates from robot.yaml" in w for w in st["warnings"]) == (expect == "estimated")
