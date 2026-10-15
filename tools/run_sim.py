#!/usr/bin/env python3
"""Start the simulated rig: robotd --sim and the MCP server, together. Ctrl-C stops both.

    python tools/run_sim.py                     # the block_cup scene, a new layout each run
    python tools/run_sim.py --seed 7            # repeat a layout (the seed is printed at start)
    python tools/run_sim.py --scene my.yaml     # another scene file (stack/twin_mujoco/scenes/*.yaml format)

Then, in another terminal: `cd agent_ws && claude` (or `codex`), and give the agent a task, e.g. "put the blue block
in the pink cup". Watch it on the operator page (printed below).

The sim runs with the repository's robotd settings, except: motion is on (nothing physical moves), the agent's
programs run without a human approving them (approval_mode none, which robotd allows only with everything bound to
127.0.0.1), and its event log, programs and episodes go to a folder of their own (--home, default
$ROBOT_USE_HOME/sim, ~/.tensortron/robot-use/sim), apart
from any real rig's. The ports are the ones agent_ws/.mcp.json expects (api 8770, MCP 8771, operator 8772);
--port-base moves all three, for development beside a running rig (the agent's config then needs the new MCP port).

robotd and the MCP server run in their own process session, so the terminal's Ctrl-C reaches only this launcher,
which stops them in order. They also stop if the launcher dies (closing the terminal: SIGHUP is handled; on Linux
even a kill -9, through a parent-death signal). Their PIDs are kept in <home>/run_sim.pids while they run, so a
leftover sim (macOS after a kill -9) can be named and stopped by PID.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

START_TIMEOUT_S = 180                   # GL probe, scene, first compute workers
PIDS_FILE = "run_sim.pids"

from launch_common import (Stopping, _parent_death_hook, agent_settings_current, leftover_pids,  # noqa: E402,F401
                           port_free, spawn, stop, tail, wait_http)


def sim_config(base: dict, home: Path, port_base: int) -> dict:
    """robotd's settings for the sim: the repository's, with the sim's ports, folder and approval."""
    cfg = dict(base)
    cfg.update(api={"host": "127.0.0.1", "port": port_base}, operator={"host": "127.0.0.1", "port": port_base + 2},
               log_dir=str(home / "logs"), approval_mode="none", mcp_localhost_only=True, motion_enabled=False)
    return cfg                                   # motion_enabled stays false in the file: --sim turns it on


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None, help="a scene name (stack/twin_mujoco/scenes) or a .yaml path")
    ap.add_argument("--seed", type=int, default=None, help="repeat a layout")
    ap.add_argument("--home", default=None,
                    help="where the sim's logs, programs and episodes go (default $ROBOT_USE_HOME/sim, ~/.tensortron/robot-use/sim)")
    ap.add_argument("--port-base", type=int, default=8770, help="robotd API at this port, MCP +1, operator +2")
    a = ap.parse_args()
    if not 1024 <= a.port_base <= 65533:
        ap.error("--port-base must be between 1024 and 65533")

    import threading
    stopping = threading.Event()

    def on_signal(*_):
        stopping.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig == signal.SIGHUP and signal.getsignal(sig) is signal.SIG_IGN:
            continue                                     # under nohup: keep running when the terminal goes
        signal.signal(sig, on_signal)

    from stack import paths
    home = Path(a.home).expanduser() if a.home else paths.sim_home()
    (home / "logs").mkdir(parents=True, exist_ok=True)
    ports = {"robotd API": a.port_base, "MCP server": a.port_base + 1, "operator page": a.port_base + 2}
    busy = [f"{name} port {p}" for name, p in ports.items() if not port_free(p)]
    if busy:
        left = leftover_pids(home, PIDS_FILE)
        hint = (f"A sim left from an earlier run is still up: stop it with  kill {' '.join(map(str, left))}"
                if left else "Is robotd (real or sim) or an MCP server already running? Stop it, or use "
                             "--port-base for other ports.")
        print(f"cannot start: {', '.join(busy)} in use. {hint}", file=sys.stderr)
        return 2

    import yaml

    from stack.robotd.config import DEFAULT_CONFIG
    from stack.twin_mujoco.render import gl_env_candidates, probe_gl
    if not agent_settings_current():
        return 2
    print("checking which OpenGL setting renders here...", flush=True)
    gl = None
    for cand in gl_env_candidates():                     # find_gl_env, one candidate at a time, so Ctrl-C stops it
        if stopping.is_set():
            return 0
        if probe_gl(cand):
            gl = cand
            break
    if stopping.is_set():
        return 0
    if gl is None:
        print("WARNING: no OpenGL setting renders here; the sim runs, but its cameras will only say so. "
              "Run tools/doctor.py.", file=sys.stderr)
        gl = gl_env_candidates()[0] or {}        # pass a setting on, so robotd does not probe all over again
    env = {**os.environ, **gl}
    hook = _parent_death_hook()

    rd_log, mcp_log = home / "logs" / "robotd.log", home / "logs" / "mcp_server.log"
    api = f"http://127.0.0.1:{a.port_base}"
    procs = []
    cfg_dir = Path(tempfile.mkdtemp(prefix="robot_use_sim_"))
    try:
        cfg_path = cfg_dir / "robotd.yaml"
        cfg_path.write_text(yaml.safe_dump(sim_config(yaml.safe_load(Path(DEFAULT_CONFIG).read_text()), home,
                                                      a.port_base)))
        rd_cmd = [sys.executable, "-m", "stack.robotd", "--sim", "--config", str(cfg_path)]
        if a.scene:
            rd_cmd += ["--scene", a.scene]
        if a.seed is not None:
            rd_cmd += ["--seed", str(a.seed)]
        procs.append(spawn(rd_cmd, rd_log, env, hook))
        (home / PIDS_FILE).write_text(f"{procs[0].pid}\n")
        print("starting robotd --sim ...", flush=True)
        if not wait_http(api + "/state", procs[0], START_TIMEOUT_S, stopping):
            print(f"robotd --sim did not start. Its log ({rd_log}):\n{tail(rd_log)}", file=sys.stderr)
            return 1
        mcp_cmd = [sys.executable, "-m", "stack.mcp_server", "--host", "127.0.0.1", "--port", str(a.port_base + 1),
                   "--robotd", api, "--episodes", str(home / "episodes")]
        if stopping.is_set():
            raise Stopping
        procs.append(spawn(mcp_cmd, mcp_log, env, hook))
        (home / PIDS_FILE).write_text(" ".join(str(p.pid) for p in procs) + "\n")
        if not wait_http(f"http://127.0.0.1:{a.port_base + 1}/mcp", procs[1], 60, stopping, any_status=True):
            print(f"the MCP server did not start. Its log ({mcp_log}):\n{tail(mcp_log)}", file=sys.stderr)
            return 1
        lines = tail(rd_log, 200).splitlines()
        seed_line = next((ln for ln in lines if "sim scene" in ln), "")
        render_line = next((ln for ln in lines if "sim rendering" in ln), "")
        print(f"""
The simulated rig is running.
  {seed_line.split(' INFO ')[-1]}
  {render_line.split(' INFO ')[-1]}
  operator page   http://127.0.0.1:{a.port_base + 2}/      (live cameras, STOP; on a remote machine, an SSH tunnel)
  MCP server      http://127.0.0.1:{a.port_base + 1}/mcp   (agent_ws/.mcp.json{'' if a.port_base == 8770 else ' needs this port'})
  robotd API      {api}/state
  logs            {home / 'logs'}
  episodes        {home / 'episodes'}

Next, in another terminal:  cd {REPO / 'agent_ws'} && claude
and ask, for example: "put the blue block in the pink cup".   Ctrl-C here stops the sim.
""", flush=True)
        while not stopping.is_set():
            for p, name, log in ((procs[0], "robotd", rd_log), (procs[1], "the MCP server", mcp_log)):
                if p.poll() is not None:
                    print(f"{name} exited ({p.returncode}). Its log ({log}):\n{tail(log)}", file=sys.stderr)
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
