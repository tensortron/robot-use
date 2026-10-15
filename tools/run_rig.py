#!/usr/bin/env python3
"""Start the real rig: robotd on the arms and the MCP server, together. Ctrl-C stops both.

    .venv/bin/python tools/run_rig.py                                   # read-only: robotd reads, plans, moves nothing
    .venv/bin/python tools/run_rig.py --enable-motion                   # attended: a human approves each program
    .venv/bin/python tools/run_rig.py --enable-motion --approval none   # unattended: programs run without approval
    .venv/bin/python tools/run_rig.py --mock                            # mock arms, no hardware (to try the launcher)

**Only move the real arms with a person at the 12 V supply** (AGENTS.md). Power the arms (12 V) before the USB hub.

robotd's settings are the repository's `stack/robotd.yaml`, the one source, with this session's choices applied:
the approval mode (`--approval`, default `first_motion`), motion (`--enable-motion`, off unless given), the API and
the MCP server on 127.0.0.1 (so `approval_mode: none` is allowed), the operator page on the LAN as configured, and
the recording folders: robotd's log in $ROBOT_USE_HOME/agent_logs and the programs and episodes in
$ROBOT_USE_HOME/episodes (default ~/.tensortron/robot-use; stack/paths.py); a --mock run records apart, in
$ROBOT_USE_HOME/mock. Before starting, it checks that the
arms' and cameras' devices exist and no other process holds them (teleop, calibration, another robotd), and that
the ports are free. It never moves anything itself: robotd reads the arms at start and leaves their torque as it
finds it. Stopping it halts a running motion, then the arms hold where they are under torque, with nothing watching
for contact until robotd runs again.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from launch_common import (Stopping, _parent_death_hook, agent_settings_current, leftover_pids,  # noqa: E402
                           port_free, spawn, stop, tail, wait_http)

PIDS_FILE = "run_rig.pids"
START_TIMEOUT_S = 120                  # the arms' buses and calibration, the cameras, the planning workers


def rig_config(base: dict, approval: str, port_base: int, log_dir: Path) -> dict:
    """robotd's settings for this session: the repository's, with the session's choices."""
    cfg = dict(base)
    operator = dict(base.get("operator", {"host": "0.0.0.0"}))
    operator["port"] = port_base + 2
    cfg.update(api={"host": "127.0.0.1", "port": port_base}, operator=operator, approval_mode=approval,
               mcp_localhost_only=True, log_dir=str(log_dir), motion_enabled=False)
    return cfg                              # motion stays off in the file: --enable-motion turns it on


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))     # no packet is sent: this only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def device_problems(cfg) -> tuple[list[str], list[str]]:
    """(problems that stop the start, warnings) about the arms' and cameras' devices."""
    from stack.robotd.hardware import device_holders
    stop_, warn = [], []
    for name, a in cfg.arms.items():
        if not os.path.exists(a.port):
            stop_.append(f"{name} arm: {a.port} does not exist (12 V on, then the USB hub; docs/HARDWARE_SETUP.md)")
    for name, c in cfg.cameras.items():
        if not os.path.exists(c.path):
            warn.append(f"camera {name}: {c.path} does not exist (robotd will keep trying; its frames are missing)")
    held = device_holders([a.port for a in cfg.arms.values()] + [c.path for c in cfg.cameras.values()])
    for path, pids in held.items():
        stop_.append(f"{path} is held by pid {', '.join(map(str, pids))} (teleop, calibration, another robotd?)")
    return stop_, warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enable-motion", action="store_true", help="allow motion (a person at the 12 V supply)")
    ap.add_argument("--approval", choices=("first_motion", "every_motion", "none"), default="first_motion",
                    help="who approves programs: first_motion (a human, once per program; default), every_motion, "
                         "none (unattended)")
    ap.add_argument("--mock", action="store_true", help="mock arms and cameras instead of the hardware")
    ap.add_argument("--port-base", type=int, default=8770, help="robotd API at this port, MCP +1, operator +2")
    a = ap.parse_args()
    if not 1024 <= a.port_base <= 65533:
        ap.error("--port-base must be between 1024 and 65533")

    import yaml

    from stack import paths
    from stack.robotd.config import DEFAULT_CONFIG, load_config

    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig == signal.SIGHUP and signal.getsignal(sig) is signal.SIG_IGN:
            continue
        signal.signal(sig, lambda *_: stopping.set())

    home = paths.base() / "mock" if a.mock else paths.base()     # a mock run never mixes into the rig's records
    log_dir, episodes = home / "agent_logs", home / "episodes"
    log_dir.mkdir(parents=True, exist_ok=True)
    busy = [f"{n} port {p}" for n, p in (("robotd API", a.port_base), ("MCP server", a.port_base + 1),
                                         ("operator page", a.port_base + 2)) if not port_free(p)]
    if busy:
        left = leftover_pids(home, PIDS_FILE)
        hint = (f"A rig launcher's servers are still up: stop them with  kill {' '.join(map(str, left))}" if left
                else "Is robotd (real or sim) or an MCP server already running? Stop it, or use --port-base.")
        print(f"cannot start: {', '.join(busy)} in use. {hint}", file=sys.stderr)
        return 2

    cfg_dir = Path(tempfile.mkdtemp(prefix="robot_use_rig_"))
    procs = []
    rd_log, mcp_log = log_dir / "robotd.out", log_dir / "mcp_server.out"
    api = f"http://127.0.0.1:{a.port_base}"
    try:
        cfg_path = cfg_dir / "robotd.yaml"
        cfg_path.write_text(yaml.safe_dump(rig_config(yaml.safe_load(Path(DEFAULT_CONFIG).read_text()), a.approval,
                                                      a.port_base, log_dir)))
        if not a.mock:
            problems, warnings = device_problems(load_config(cfg_path))
            if not shutil.which("fuser"):
                warnings.append("fuser (psmisc) is not installed, so whether another process holds the arms can't "
                                "be checked here; robotd's exclusive open still refuses a held port")
            for w in warnings:
                print(f"WARNING: {w}", file=sys.stderr)
            if problems:
                print("cannot start:\n  " + "\n  ".join(problems), file=sys.stderr)
                return 2
        if not agent_settings_current(refuse=not a.mock):
            return 2
        hook = _parent_death_hook()
        rd_cmd = [sys.executable, "-m", "stack.robotd", "--config", str(cfg_path)]
        rd_cmd += ["--enable-motion"] if a.enable_motion else []
        rd_cmd += ["--mock"] if a.mock else []
        procs.append(spawn(rd_cmd, rd_log, dict(os.environ), hook))
        (home / PIDS_FILE).write_text(f"{procs[0].pid}\n")
        print("starting robotd ...", flush=True)
        if not wait_http(api + "/state", procs[0], START_TIMEOUT_S, stopping):
            print(f"robotd did not start. Its output ({rd_log}):\n{tail(rd_log)}", file=sys.stderr)
            return 1
        if stopping.is_set():
            raise Stopping
        mcp_cmd = [sys.executable, "-m", "stack.mcp_server", "--host", "127.0.0.1", "--port", str(a.port_base + 1),
                   "--robotd", api, "--episodes", str(episodes)]
        procs.append(spawn(mcp_cmd, mcp_log, dict(os.environ), hook))
        (home / PIDS_FILE).write_text(" ".join(str(p.pid) for p in procs) + "\n")
        if not wait_http(f"http://127.0.0.1:{a.port_base + 1}/mcp", procs[1], 60, stopping, any_status=True):
            print(f"the MCP server did not start. Its output ({mcp_log}):\n{tail(mcp_log)}", file=sys.stderr)
            return 1
        arms = [ln.split(" INFO ")[-1] for ln in tail(rd_log, 200).splitlines() if ": connected=" in ln]
        motion = "ENABLED" if a.enable_motion else "off (read-only: add --enable-motion to move)"
        approval = {"first_motion": "a human approves each program on the operator page",
                    "every_motion": "a human approves every motion on the operator page",
                    "none": "NONE: programs run as soon as the agent submits them"}[a.approval]
        print(f"""
The rig is running{' (MOCK arms)' if a.mock else ''}.
  motion          {motion}
  approval        {approval}
  arms            {'; '.join(arms) or '(see the log)'}
  operator page   http://{lan_address()}:{a.port_base + 2}/   (live cameras, approvals, STOP, torque off)
  MCP server      http://127.0.0.1:{a.port_base + 1}/mcp
  logs            {log_dir}
  episodes        {episodes}
""" + ("""
  ** Motion is enabled: a person must be at the 12 V supply (the e-stop) while the arms can move. **
""" if a.enable_motion and not a.mock else "") + f"""
Next, in another terminal:  cd {REPO / 'agent_ws'} && claude     (or codex)
Ctrl-C here stops both: a running motion is halted first; the arms then hold where they are, under torque, with
nothing watching for contact until robotd runs again (torque off: the operator page, or tools/torque_off.py).
""", flush=True)
        while not stopping.is_set():
            for p, name, out in ((procs[0], "robotd", rd_log), (procs[1], "the MCP server", mcp_log)):
                if p.poll() is not None:
                    print(f"{name} exited ({p.returncode}). Its output ({out}):\n{tail(out)}", file=sys.stderr)
                    return 1
            stopping.wait(0.5)
        raise Stopping
    except Stopping:
        print("stopping...", flush=True)
        return 0
    finally:
        stop(procs)
        (home / PIDS_FILE).unlink(missing_ok=True)
        for f in cfg_dir.iterdir():
            f.unlink()
        cfg_dir.rmdir()


if __name__ == "__main__":
    sys.exit(main())
