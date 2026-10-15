"""tools/run_sim.py (the sim launcher) and tools/doctor.py: what a newcomer runs first."""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))


def _free_base() -> int:
    """Three consecutive free ports."""
    for base in range(18770, 19770, 10):
        ok = True
        for p in range(base, base + 3):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
        if ok:
            return base
    pytest.skip("no three consecutive free ports")


def test_sim_config_keeps_the_sim_local_and_apart():
    import run_sim

    cfg = run_sim.sim_config({"approval_mode": "first_motion", "log_dir": "~/logs",
                              "api": {"host": "127.0.0.1", "port": 8770}}, Path("/x/sim"), 9000)
    assert cfg["api"] == {"host": "127.0.0.1", "port": 9000}
    assert cfg["operator"] == {"host": "127.0.0.1", "port": 9002}             # not on the LAN, unlike the rig's
    assert cfg["approval_mode"] == "none" and cfg["mcp_localhost_only"] is True
    assert cfg["log_dir"] == "/x/sim/logs" and cfg["motion_enabled"] is False


def test_the_doctor_names_what_is_missing_and_how_to_fix_it(tmp_path):
    env = {**os.environ, "ROBOT_USE_SO101_ASSETS": str(tmp_path / "nowhere")}
    p = subprocess.run([sys.executable, "tools/doctor.py"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 1, p.stdout
    assert "FAIL  SO-101 MuJoCo model" in p.stdout and "tools/fetch_so101_assets.py" in p.stdout
    assert "ok    package mujoco" in p.stdout and "problem(s) to fix first" in p.stdout


def _needs_sim():
    pytest.importorskip("mujoco")
    try:
        from stack.twin_mujoco.scene import assets_dir
        assets_dir()
    except FileNotFoundError:
        pytest.skip("SO-101 MuJoCo assets not fetched")


def _launch(tmp_path, base, *extra):
    out = tmp_path / "run_sim.out"
    f = open(out, "w")
    p = subprocess.Popen([sys.executable, "tools/run_sim.py", "--port-base", str(base), "--home", str(tmp_path / "home"),
                          *extra], cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    f.close()
    return p, out


def _wait_for(out: Path, p, text: str, timeout_s: float = 240):
    deadline = time.monotonic() + timeout_s
    while text not in out.read_text():
        assert p.poll() is None and time.monotonic() < deadline, out.read_text()
        time.sleep(0.2)


def _children(tmp_path) -> list[int]:
    try:
        return [int(x) for x in (tmp_path / "home" / "run_sim.pids").read_text().split()]
    except OSError:
        return []


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                          # a zombie is dead for our purposes
        return Path(f"/proc/{pid}/status").read_text().find("State:\tZ") < 0
    except OSError:
        return True


def _cleanup(p, tmp_path, kids):
    """Never leave a sim behind, whatever failed: the launcher first (it stops its children), then by PID."""
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    for pid in set(kids) | set(_children(tmp_path)):
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _ports_free(base):
    import run_sim
    assert all(run_sim.port_free(port) for port in (base, base + 1, base + 2))


def test_run_sim_starts_both_and_ctrl_c_stops_both(tmp_path):
    _needs_sim()
    base = _free_base()
    p, out = _launch(tmp_path, base, "--seed", "5")
    kids = []
    try:
        _wait_for(out, p, "Ctrl-C here")
        kids = _children(tmp_path)
        assert len(kids) == 2
        text = out.read_text()
        assert "seed 5" in text and f"http://127.0.0.1:{base + 1}/mcp" in text
        with urllib.request.urlopen(f"http://127.0.0.1:{base}/state", timeout=5) as r:
            assert r.status == 200
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=60) == 0
        assert not any(_alive(k) for k in kids) and not _children(tmp_path)
    finally:
        _cleanup(p, tmp_path, kids)
    time.sleep(0.5)
    _ports_free(base)
    assert (tmp_path / "home" / "logs" / "robotd.log").is_file()
    assert list((tmp_path / "home" / "logs" / "sim").glob("*.jsonl"))   # the sim's events in their own folder


def test_ctrl_c_during_start_up_stops_at_once(tmp_path):
    _needs_sim()
    base = _free_base()
    p, out = _launch(tmp_path, base)
    kids = []
    try:
        _wait_for(out, p, "starting robotd --sim")
        kids = _children(tmp_path)
        t0 = time.monotonic()
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=30) == 0
        assert time.monotonic() - t0 < 20 and "is running" not in out.read_text()
        assert not any(_alive(k) for k in kids)
    finally:
        _cleanup(p, tmp_path, kids)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the parent-death signal is Linux's")
def test_a_killed_launcher_takes_the_sim_with_it(tmp_path):
    _needs_sim()
    base = _free_base()
    p, out = _launch(tmp_path, base)
    kids = []
    try:
        _wait_for(out, p, "Ctrl-C here")
        kids = _children(tmp_path)
        p.kill()                                                  # no chance to clean up
        p.wait(timeout=10)
        deadline = time.monotonic() + 30
        while any(_alive(k) for k in kids) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not any(_alive(k) for k in kids)
    finally:
        _cleanup(p, tmp_path, kids)


def test_run_sim_takes_random_or_a_seed_not_both(tmp_path):
    p = subprocess.run([sys.executable, "tools/run_sim.py", "--random", "--seed", "1", "--home", str(tmp_path / "home")],
                       cwd=REPO, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "--random draws a new layout" in p.stderr


def test_run_sim_refuses_ports_in_use(tmp_path):
    base = _free_base()
    with socket.socket() as s:
        s.bind(("127.0.0.1", base + 1))
        s.listen()
        p = subprocess.run([sys.executable, "tools/run_sim.py", "--port-base", str(base), "--home",
                            str(tmp_path / "home")], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and f"MCP server port {base + 1} in use" in p.stderr


def test_a_refusal_names_a_leftover_sim_by_pid(tmp_path):
    base = _free_base()
    home = tmp_path / "home"
    home.mkdir()
    sim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "stack.robotd"])   # looks like one
    (home / "run_sim.pids").write_text(f"{sim.pid} {os.getpid()} 999999999\n")  # a sim; alive but not a sim; gone
    with socket.socket() as s:
        s.bind(("127.0.0.1", base))
        s.listen()
        p = subprocess.run([sys.executable, "tools/run_sim.py", "--port-base", str(base), "--home", str(home)],
                           cwd=REPO, capture_output=True, text=True, timeout=60)
    sim.kill()
    sim.wait()
    assert p.returncode == 2 and f"kill {sim.pid}" in p.stderr, p.stderr
    assert str(os.getpid()) not in p.stderr and "999999999" not in p.stderr


# ---------------------------------------------------------------- tools/run_rig.py (the real rig's launcher)
def test_rig_config_is_the_repos_with_only_the_sessions_choices(tmp_path):
    import yaml
    import run_rig
    from stack.robotd.config import DEFAULT_CONFIG

    base = yaml.safe_load(Path(DEFAULT_CONFIG).read_text())
    cfg = run_rig.rig_config(base, "none", 9000, tmp_path / "logs")
    assert cfg["approval_mode"] == "none" and cfg["mcp_localhost_only"] is True
    assert cfg["api"] == {"host": "127.0.0.1", "port": 9000}
    assert cfg["operator"] == {**base["operator"], "port": 9002}                 # still on the LAN, as configured
    assert cfg["motion_enabled"] is False and cfg["log_dir"] == str(tmp_path / "logs")
    for k in set(base) - {"approval_mode", "mcp_localhost_only", "api", "operator", "log_dir", "motion_enabled"}:
        assert cfg[k] == base[k], k                                              # everything else: the repo's
    assert run_rig.rig_config(base, "first_motion", 9000, tmp_path)["approval_mode"] == "first_motion"


def test_rig_launcher_refuses_missing_or_held_devices(tmp_path, monkeypatch):
    import types
    import run_rig
    from stack.robotd import hardware

    held = tmp_path / "ttyACM9"
    held.write_text("")
    cfg = types.SimpleNamespace(
        arms={"left": types.SimpleNamespace(port=str(tmp_path / "gone")), "right": types.SimpleNamespace(port=str(held))},
        cameras={"ego": types.SimpleNamespace(path=str(tmp_path / "video9"))})
    monkeypatch.setattr(hardware, "device_holders", lambda paths: {str(held): [4242]})
    problems, warnings = run_rig.device_problems(cfg)
    assert any("left arm" in p and "does not exist" in p for p in problems)
    assert any("held by pid 4242" in p for p in problems)
    assert any("camera ego" in w for w in warnings)


def test_rig_launcher_with_mock_arms_starts_both_and_stops_both(tmp_path):
    _needs_sim()
    base = _free_base()
    out = tmp_path / "run_rig.out"
    env = {**os.environ, "ROBOT_USE_HOME": str(tmp_path / "home")}
    with open(out, "w") as f:
        p = subprocess.Popen([sys.executable, "tools/run_rig.py", "--mock", "--enable-motion", "--approval", "none",
                              "--port-base", str(base)], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    kids = []
    try:
        _wait_for(out, p, "Ctrl-C here")
        kids = [int(x) for x in (tmp_path / "home" / "mock" / "run_rig.pids").read_text().split()]
        text = out.read_text()
        assert "MOCK" in text and "NONE: programs run" in text and f"127.0.0.1:{base + 1}/mcp" in text
        import json
        with urllib.request.urlopen(f"http://127.0.0.1:{base}/state", timeout=5) as r:
            st = json.loads(r.read())
        assert st["backend"] == "mock" and st["supervisor"]["mode"] == "none" and st["motion_enabled"] is True
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=60) == 0
        assert not any(_alive(k) for k in kids)
        assert (tmp_path / "home" / "mock" / "agent_logs").is_dir()         # a mock run records apart,
        assert not (tmp_path / "home" / "agent_logs").exists()               # never in the rig's folders
    finally:
        _cleanup(p, tmp_path, kids)
