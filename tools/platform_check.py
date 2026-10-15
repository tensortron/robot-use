#!/usr/bin/env python3
"""Check this machine end to end and write one report to send back: the platform, the agent settings, the doctor,
the sandbox escape probe, and the whole no-hardware test suite (with what was skipped, and why).

On a machine that has never had the repository (macOS: `xcode-select --install` first, for git):

    curl -LsSf https://astral.sh/uv/install.sh | sh        # or: brew install uv
    git clone <the repository> && cd <it>
    uv sync --extra dev                                     # fetches Python 3.12 itself if needed
    uv run python tools/platform_check.py                   # writes platform_check-<host>-<date>.txt here

It fetches the SO-101 MuJoCo model if it is missing (a git clone, ~20 MB), and writes the operating agent's settings
for this machine (tools/setup_agent_ws.py). Nothing else is changed; no hardware is touched. Never on the Jetson
with uv: there, use .venv/bin/python (docs/environment.md).
"""

from __future__ import annotations

import datetime
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(title: str, cmd: list[str], out, timeout: float = 1800, env=None) -> int:
    out.write(f"\n===== {title}\n$ {' '.join(cmd)}\n")
    out.flush()
    print(f"-- {title} ...", flush=True)
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        text, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        part = "".join(x.decode(errors="replace") if isinstance(x, bytes) else (x or "") for x in (e.stdout, e.stderr))
        text, code = f"{part}\nTIMED OUT after {timeout:.0f} s", -1
    out.write(text)
    out.write(f"\n[exit {code}, {time.monotonic() - t0:.0f} s]\n")
    out.flush()
    print(f"   exit {code} ({time.monotonic() - t0:.0f} s)", flush=True)
    run.last_text = text
    return code


def main() -> int:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = Path.cwd() / f"platform_check-{socket.gethostname().split('.')[0]}-{stamp}.txt"
    py = sys.executable
    with open(path, "w") as out:
        out.write(f"robot-use platform check {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"platform: {platform.platform()}  machine: {platform.machine()}\n")
        out.write(f"python: {sys.version.split()[0]} ({py})\n")
        try:
            g = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
            commit = g.stdout.strip() if g.returncode == 0 and g.stdout.strip() else "? (not a git checkout)"
        except OSError:
            commit = "? (no git)"
        out.write(f"repo: {REPO} at {commit}\n")
        if sys.platform.startswith("linux"):
            out.write(f"kernel: {platform.release()}\n")
        sys.path.insert(0, str(REPO))
        try:
            from stack.twin_mujoco.scene import assets_dir
            assets_dir()
        except FileNotFoundError:
            run("fetch the SO-101 MuJoCo model", [py, "tools/fetch_so101_assets.py"], out, timeout=600)
        except Exception as e:
            out.write(f"\n(cannot check the model: {type(e).__name__}: {e})\n")
        codes = {"settings": run("agent settings", [py, "tools/setup_agent_ws.py"], out)}   # before the doctor checks it
        codes["doctor"] = run("doctor", [py, "tools/doctor.py"], out, timeout=600)
        codes["escape"] = run("sandbox escape probe", [py, "-m", "pytest", "stack/tests/test_sandbox_escape.py",
                                                        "-s", "-q", "-rs", "-p", "no:cacheprovider"], out, timeout=300)
        escape_skipped = codes["escape"] == 0 and " passed" not in run.last_text
        codes["suite"] = run("test suite", [py, "-m", "pytest", "stack/tests", "-q", "-rs", "-p", "no:cacheprovider"],
                             out, timeout=3600)

        def verdict(k, v):
            if k == "escape" and escape_skipped:
                return "SKIPPED (no kernel sandbox here: nothing proved; see the report)"
            return "ok" if v == 0 else f"exit {v}"
        summary = "".join(f"  {k}: {verdict(k, v)}\n" for k, v in codes.items())
        out.write("\n===== summary\n" + summary)
    print(f"\nreport: {path}\n" + summary)
    return 0 if all(v == 0 for v in codes.values()) and not escape_skipped else 1


if __name__ == "__main__":
    sys.exit(main())
