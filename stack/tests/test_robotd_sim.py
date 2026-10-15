"""robotd --sim: the simulated rig behind the real daemon. Never hardware; the agent's surface works end to end."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mujoco")
try:
    from stack.twin_mujoco.scene import assets_dir
    assets_dir()
except FileNotFoundError:
    pytest.skip("SO-101 MuJoCo assets not fetched (tools/fetch_so101_assets.py)", allow_module_level=True)

from stack.robotd import __main__ as robotd_main  # noqa: E402
from stack.robotd.config import DEFAULT_CONFIG, load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# a fixed scene: the block square-on where the twin's roll-0 grasp is known to work (test_twin's PICK), cups apart
SCENE = {"name": "test_fixed", "objects": [
    {"name": "blue_block", "shape": "box", "size": [0.04, 0.04, 0.04], "rgba": [0.1, 0.35, 0.85, 1], "mass": 0.03,
     "xy": [0.22, -0.17]},
    {"name": "pink_cup", "shape": "cup", "size": [0.085, 0.085, 0.05], "rgba": [0.95, 0.4, 0.65, 1], "mass": 0.04,
     "xy": [0.28, 0.05]},
]}
PICK = textwrap.dedent("""
    from robot_use import arm, estimate, contact, log
    r = arm("right")
    block = estimate("block", [0.22, -0.17], size=(0.04, 0.04, 0.04), tol=0.03, source="test")
    dx = -0.0236
    r.open(80)
    r.move_to([block.xy[0] + dx, block.xy[1], 0.12], pitch=-70, roll=0, speed=0.8)
    r.move_to([block.xy[0] + dx, block.xy[1], 0.022], pitch=-70, speed=0.3, until=contact())
    g = r.close(effort=0.4)
    log("grasped " + str(bool(g.grasped)))
    r.move_by(dz=0.06)
""")


def _no_hardware(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("--sim/--mock touched hardware")
    for name in ("ArmIO", "CameraIO", "device_holders"):
        monkeypatch.setattr(robotd_main, name, refuse)


def test_sim_and_mock_build_no_hardware(monkeypatch, tmp_path):
    from stack.robotd import sim as simmod
    from stack.twin_mujoco.render import SimCameraIO
    from stack.twin_mujoco.sim import SimArmIO

    _no_hardware(monkeypatch)
    monkeypatch.setattr(simmod, "choose_gl", lambda: None)          # no GL needed: nothing is rendered here
    a = robotd_main.parse_args(["--sim", "--seed", "3"])
    cfg = load_config()
    cfg.log_dir = tmp_path
    arm_ios, cam_ios, limits, sim = robotd_main.backends(a, cfg)
    try:
        assert {type(io) for io in arm_ios.values()} == {SimArmIO} and set(arm_ios) == set(cfg.arms)
        assert {type(io) for io in cam_ios.values()} == {SimCameraIO} and set(cam_ios) == set(cfg.cameras)
        assert cfg.motion_enabled and cfg.log_dir == tmp_path / "sim" and set(limits) == set(cfg.arms)
        assert sim.layout.seed == 3
    finally:
        sim.stop()
    arm_ios, cam_ios, limits, sim = robotd_main.backends(robotd_main.parse_args(["--mock"]), load_config())
    assert sim is None and arm_ios and limits


def test_choose_gl_sets_the_backend_before_mujoco_is_imported(monkeypatch):
    from stack.robotd import sim as simmod
    from stack.twin_mujoco import render

    monkeypatch.setenv("MUJOCO_GL", "placeholder")              # recorded, so teardown restores the environment
    monkeypatch.delenv("MUJOCO_GL")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "placeholder")
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_FILENAMES")
    monkeypatch.delitem(sys.modules, "mujoco")                  # as in robotd's process at start-up
    monkeypatch.setattr(render, "find_gl_env", lambda: {"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "x"})
    assert "MUJOCO_GL=egl" in simmod.choose_gl()
    assert os.environ["MUJOCO_GL"] == "egl" and os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] == "x"
    assert simmod.choose_gl() == "MUJOCO_GL=egl (given)"          # an explicit setting is left alone
    monkeypatch.delenv("MUJOCO_GL")
    monkeypatch.setattr(render, "find_gl_env", lambda: None)
    assert simmod.choose_gl() is None                           # nothing renders: robotd says so, cameras too


def test_a_bad_scene_is_a_logged_error_not_a_traceback(monkeypatch, tmp_path):
    from stack.robotd import sim as simmod

    _no_hardware(monkeypatch)
    monkeypatch.setattr(simmod, "choose_gl", lambda: None)
    bad = tmp_path / "bad.yaml"
    bad.write_text("objects:\n  - {name: a, shape: ball, xy: [0.2, 0.0]}\n")
    cfg = load_config()
    cfg.log_dir = tmp_path
    assert robotd_main.backends(robotd_main.parse_args(["--sim", "--scene", str(bad)]), cfg) is None


def test_one_arm_sim_has_that_arm_and_its_wrist_camera(monkeypatch, tmp_path):
    from stack.robotd import sim as simmod

    _no_hardware(monkeypatch)
    monkeypatch.setattr(simmod, "choose_gl", lambda: None)
    cfg = load_config()
    cfg.log_dir = tmp_path
    cfg.arms = {"right": cfg.arms["right"]}
    arm_ios, cam_ios, limits, sim = robotd_main.backends(robotd_main.parse_args(["--sim", "--seed", "2"]), cfg)
    try:
        assert set(arm_ios) == set(limits) == {"right"} and set(cam_ios) == {"ego", "wrist_right"}
    finally:
        sim.stop()


@pytest.mark.parametrize("argv", [["--sim", "--mock"], ["--sim", "--enable-motion"], ["--seed", "1"],
                                  ["--scene", "block_cup"]])
def test_sim_flags_that_make_no_sense_are_refused(argv):
    with pytest.raises(SystemExit) as e:
        robotd_main.parse_args(argv)
    assert e.value.code == 2


def _free_port() -> int:
    from stack.twin_mujoco.preview import _free_port
    return _free_port()


def _get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


@pytest.fixture(scope="module")
def sim_robotd(tmp_path_factory):
    """`python -m stack.robotd --sim` as robotd runs it, on free ports, with the fixed scene."""
    from stack.twin_mujoco.render import find_gl_env

    d = tmp_path_factory.mktemp("sim")
    raw = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    api, op = _free_port(), _free_port()
    raw.update(api={"host": "127.0.0.1", "port": api}, operator={"host": "127.0.0.1", "port": op},
               log_dir=str(d / "logs"), approval_mode="none", mcp_localhost_only=True)
    (d / "robotd.yaml").write_text(yaml.safe_dump(raw))
    (d / "scene.yaml").write_text(yaml.safe_dump(SCENE))
    gl = find_gl_env()
    if gl is None:
        pytest.skip("no OpenGL backend renders here (python -m stack.twin_mujoco.render)")
    env = {**os.environ, **gl}
    log = open(d / "robotd.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "stack.robotd", "--sim", "--scene", str(d / "scene.yaml"),
                          "--config", str(d / "robotd.yaml")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{api}"
    try:
        for _ in range(120):
            try:
                _get(base + "/state", timeout=2)
                break
            except OSError:
                assert p.poll() is None, (d / "robotd.log").read_text()[-3000:]
                time.sleep(0.5)
        else:
            raise AssertionError("robotd --sim did not come up: " + (d / "robotd.log").read_text()[-3000:])
        yield {"api": base, "operator": f"http://127.0.0.1:{op}", "dir": d, "proc": p}
    finally:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()


def test_sim_robotd_serves_the_agent_surface_without_revealing_the_scene(sim_robotd):
    base = sim_robotd["api"]
    _, _, body = _get(base + "/state")
    state = json.loads(body)
    assert state["motion_enabled"] is True and set(state["arms"]) == {"left", "right"}
    assert state["backend"] == "sim"                                  # for the runner's sandbox choice
    assert all(a["connected"] for a in state["arms"].values())
    status, ctype, jpg = _get(base + "/frame/ego?max_side=1280")
    assert status == 200 and ctype == "image/jpeg" and len(jpg) > 20000
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (720, 1280, 3)
    for cam in ("wrist_left", "wrist_right"):
        assert _get(f"{base}/frame/{cam}")[0] == 200
    log = (sim_robotd["dir"] / "robotd.log").read_text()
    cameras = state["cameras"]
    assert all(c["error"] is None for c in cameras.values()), (cameras, log[-2000:])   # it rendered, not "no GL"
    # where the objects are is the agent's to find: not in the API, nor in the event log
    _, _, limits = _get(base + "/limits")
    events = "".join(p.read_text() for p in (sim_robotd["dir"] / "logs" / "sim").glob("*.jsonl"))
    assert '"event": "start"' in events                               # the sim's event log, where it should be
    for text in (body.decode(), limits.decode(), events):
        assert "blue_block" not in text and "pink_cup" not in text and "seed" not in text
    assert "seed" in log                                              # the console says how to repeat the run
    assert _get(sim_robotd["operator"] + "/")[0] == 200                # the operator page


def test_a_program_picks_the_block_in_the_sim_and_the_loop_keeps_time(sim_robotd, tmp_path):
    from stack.runner.runner import END_STATES, Budgets, ProgramRunner

    base = sim_robotd["api"]
    runner = ProgramRunner(base, tmp_path / "episodes", Budgets(wall_clock_s=120),
                           preview=lambda c, s=None: {"twin": "self"}, holder="test")
    out = runner.submit(PICK, title="sim pick")
    assert out.get("ok"), out
    pid = out["program_id"]
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        st = runner.status(pid, wait_s=10)
        if st["status"] in END_STATES:
            break
    assert st["status"] == "done", st
    trace = (Path(st["dir"]) / "trace.jsonl").read_text()
    assert "grasped True" in trace
    # the block left the table: the ego view where it stood no longer shows it
    from robot_use import geometry
    import cv2
    import numpy as np
    u, v = (int(round(c)) for c in geometry.table_to_pixel([0.22, -0.17, 0.0]))
    img = cv2.imdecode(np.frombuffer(_get(base + "/frame/ego?max_side=1280")[2], np.uint8), cv2.IMREAD_COLOR)
    b, g, r = (int(x) for x in img[v, u])
    assert not (b > 150 and r < 80), (u, v, (b, g, r))               # not the block's blue at its old spot
    loop = json.loads(_get(base + "/state")[2])["loop"]
    assert loop["work_ms_p99"] < 20 and loop["measured_hz"] > 45, loop


def test_stopping_robotd_mid_motion_halts_the_motion_first(tmp_path):
    """Review of run_rig (2026-10-14): a SIGTERM (Ctrl-C in the launcher, or the launcher dying) used to stop the loop
    with a motion running: the arm froze under torque, its motion never finished in the log. Now it is halted, as the
    operator's STOP does, before robotd stops."""
    raw = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    api, op = _free_port(), _free_port()
    raw.update(api={"host": "127.0.0.1", "port": api}, operator={"host": "127.0.0.1", "port": op},
               log_dir=str(tmp_path / "logs"), approval_mode="none", mcp_localhost_only=True)
    (tmp_path / "robotd.yaml").write_text(yaml.safe_dump(raw))
    out = open(tmp_path / "robotd.out", "w")
    p = subprocess.Popen([sys.executable, "-m", "stack.robotd", "--mock", "--enable-motion",
                          "--config", str(tmp_path / "robotd.yaml")], cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{api}"
    try:
        for _ in range(120):
            try:
                _get(base + "/state", timeout=2)
                break
            except OSError:
                assert p.poll() is None, (tmp_path / "robotd.out").read_text()[-2000:]
                time.sleep(0.5)
        req = urllib.request.Request(base + "/motion/move_joints", method="POST",
                                     data=json.dumps({"arm": "left", "targets": {"shoulder_pan": 60},
                                                      "speed": 0.1}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            mid = json.loads(r.read())["motion_id"]
        for _ in range(100):
            st = json.loads(_get(f"{base}/motion/{mid}")[2])
            if st["status"] == "running":
                break
            time.sleep(0.1)
        assert st["status"] == "running", st
        for _ in range(100):                          # moving for real (a velocity to decelerate from), not just started
            pan = json.loads(_get(base + "/state")[2])["arms"]["left"]["joints"]["shoulder_pan"]
            if pan > 3:
                break
            time.sleep(0.05)
        assert pan > 3, pan
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 0
    finally:
        if p.poll() is None:
            p.kill()
        out.close()
    text = (tmp_path / "robotd.out").read_text()
    assert "halting the running motion before stopping" in text, text[-2000:]
    events = "".join(f.read_text() for f in (tmp_path / "logs").glob("*.jsonl"))
    ev = [json.loads(ln) for ln in events.splitlines() if ln.strip()]
    names = [e["event"] for e in ev]
    assert ev[0]["event"] == "start" and ev[0]["backend"] == "mock"
    assert "shutdown_halt" in names and "stop_pressed" in names
    ends = {e["motion_id"]: e for e in ev if e["event"] == "motion_end"}
    assert ends[mid]["status"] == "stopped", ends[mid]
    stops = [e for e in ev if e["event"] == "motion_start" and e["kind"] == "stop"]
    assert stops and ends[stops[0]["motion_id"]]["status"] == "done"                  # the deceleration ran to a hold
    started = {e["motion_id"] for e in ev if e["event"] == "motion_start"}
    assert started == {mid, stops[0]["motion_id"]}, started                           # nothing else launched
    last = [e for e in ev if e["event"] == "state"][-1]                        # (logged once a second)
    assert last["left"]["joints"]["shoulder_pan"] < 55, last["left"]["joints"]    # stopped well short of 60
