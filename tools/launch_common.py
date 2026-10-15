"""What tools/run_sim.py and tools/run_rig.py share: starting robotd and the MCP server as children, waiting for
them, and stopping them however the launcher ends.

The children run in their own process session, so the terminal's Ctrl-C reaches only the launcher, which stops them
in order. They also stop if the launcher dies (SIGHUP is handled by the launchers; on Linux even a kill -9, through a
parent-death signal). Their PIDs are kept in a file while they run, so a leftover can be named and stopped by PID.
No third-party or stack imports at module level: tools/doctor.py imports this for port_free on machines where
packages may be missing.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class Stopping(Exception):
    """Ctrl-C, SIGTERM or SIGHUP arrived."""


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Whether a server could listen here: nothing accepts a connection on it, and it can be bound as the servers
    bind (SO_REUSEADDR). The connection test finds a listener on 0.0.0.0 that macOS would let a SO_REUSEADDR bind
    sit beside; SO_REUSEADDR lets connections in TIME_WAIT from a run just stopped not count (found on a Mac)."""
    with socket.socket() as c:
        c.settimeout(0.5)
        if c.connect_ex((host, port)) == 0:
            return False
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def wait_http(url: str, proc: subprocess.Popen, timeout_s: float, stopping, any_status: bool = False) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stopping.is_set():
            raise Stopping
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=2).close()
            return True
        except urllib.error.HTTPError:
            if any_status:                       # up, just not a GET route (the MCP endpoint)
                return True
        except OSError:
            pass
        stopping.wait(0.5)
    return False


def tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _parent_death_hook():
    """Linux (glibc): a preexec_fn that makes the child get SIGTERM when the launcher dies, however it dies.
    libc's prctl is resolved here, in the parent: the forked child (of a process with BLAS threads) must not dlopen.
    If the launcher died before the child set it up, the child exits instead of running orphaned. None elsewhere
    (macOS, musl): there the PIDs file names a leftover."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl
    except OSError:
        return None
    parent = os.getpid()

    def hook():
        prctl(1, int(signal.SIGTERM))                       # PR_SET_PDEATHSIG
        if os.getppid() != parent:
            os._exit(1)
    return hook


def spawn(cmd, log_path: Path, env, hook) -> subprocess.Popen:
    with open(log_path, "w") as f:
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                preexec_fn=hook)


def _is_stack_process(pid: int) -> bool:
    """Whether pid is alive and is a robotd or MCP server (a recorded PID may since belong to something else)."""
    try:
        cmd = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "stack.robotd" in cmd or "stack.mcp_server" in cmd


def leftover_pids(home: Path, pids_file: str) -> list[int]:
    """The PIDs a previous launcher recorded that are still a robotd or MCP server. A stale file is removed."""
    f = home / pids_file
    try:
        pids = [int(x) for x in f.read_text().split()]
    except (OSError, ValueError):
        return []
    alive = [pid for pid in pids if _is_stack_process(pid)]
    if not alive:
        f.unlink(missing_ok=True)
    return alive


def stop(procs):
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:                # stuck in the kernel: report it, clean up the rest
                print(f"process {p.pid} did not stop; kill {p.pid}", file=sys.stderr)


def agent_settings_current(refuse: bool = True) -> bool:
    """Before a launcher starts the stack: are the operating agent's permissions (tools/setup_agent_ws.py) what the
    reviewed templates give on this machine now? Stale ones could leave a renamed or new file readable to the agent,
    so `refuse` (the real arms, the sim) stops the start and says what would change; a mock run only warns (tests
    start it with a temporary data folder, which the data-folder denies follow). Nothing is written here: rendering
    is `tools/setup_agent_ws.py`'s job. Not written yet: a warning, in every mode."""
    import setup_agent_ws
    if not setup_agent_ws.CLAUDE_OUT.exists():
        print("WARNING: the agent's settings are not written yet: run python tools/setup_agent_ws.py before starting "
              "an agent in agent_ws", file=sys.stderr)
        return True
    try:
        d = setup_agent_ws.drift()
    except Exception as e:                                   # a template not the reviewed one, unreadable files
        print(f"{'cannot start' if refuse else 'WARNING'}: the agent's permissions cannot be checked: {e}",
              file=sys.stderr)
        return not refuse
    if d is None:
        return True
    lines = [f"  + {r}" for r in d["added"]] + [f"  - {r}" for r in d["removed"]]
    print(f"{'cannot start' if refuse else 'WARNING'}: the agent's permissions are out of date "
          f"({', '.join(d['files'])}); run python tools/setup_agent_ws.py" + (":\n" + "\n".join(lines) if lines else ""),
          file=sys.stderr)
    return not refuse
