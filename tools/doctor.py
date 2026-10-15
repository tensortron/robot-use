#!/usr/bin/env python3
"""Check that this machine can run the stack, and say what to do about anything missing.

    python tools/doctor.py            # the simulated rig (no hardware)
    python tools/doctor.py --hardware # also what the real arms need

Each line is ok, WARN (works, with a caveat) or FAIL (fix it first), with the fix. Exits 1 if anything FAILs.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

CORE = [("numpy", "numpy"), ("scipy", "scipy"), ("yaml", "pyyaml"), ("cv2", "opencv-python-headless"),
        ("mujoco", "mujoco"), ("mcp", "mcp"), ("pydantic", "pydantic"), ("anyio", "anyio"), ("starlette", "starlette")]
HARDWARE = [("lerobot", "lerobot"), ("scservo_sdk", "feetech-servo-sdk"), ("serial.tools", "pyserial")]
PORTS = {8770: "robotd API", 8771: "MCP server", 8772: "operator page"}
MUJOCO_MIN = (3, 14)
INSTALL = "uv sync --extra dev (see README; on a Jetson never uv sync: docs/environment.md)"


class Report:
    def __init__(self):
        self.failed = 0
        self.ports_busy = False

    def line(self, status: str, what: str, detail: str = ""):
        if status == "FAIL":
            self.failed += 1
        print(f"  {status:<4}  {what}" + (f": {detail}" if detail else ""), flush=True)


def check_python(r: Report):
    v = sys.version_info
    if v >= (3, 12):
        r.line("ok", "Python", f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")
    else:
        r.line("FAIL", "Python", f"{v.major}.{v.minor} found, 3.12 or newer needed; {INSTALL} picks one")


def check_packages(r: Report, mods, required: bool, why: str = "", fix: str = INSTALL):
    for mod, pkg in mods:
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            r.line("FAIL" if required else "WARN", f"package {pkg}",
                   f"not importable ({type(e).__name__}){why}; {fix}")
            continue
        ver = getattr(m, "__version__", "")
        found = tuple(int(x) for x in re.findall(r"\d+", ver)[:2]) if mod == "mujoco" else ()
        if mod == "mujoco" and len(found) == 2 and found < MUJOCO_MIN:
            r.line("FAIL", "package mujoco", f"{ver} found, {MUJOCO_MIN[0]}.{MUJOCO_MIN[1]} or newer needed; {INSTALL}")
        else:
            r.line("ok", f"package {pkg}", ver)


def check_assets(r: Report):
    try:
        from stack.twin_mujoco.scene import assets_dir
        r.line("ok", "SO-101 MuJoCo model", str(assets_dir()))
    except FileNotFoundError:
        import os
        where = os.environ.get("ROBOT_USE_SO101_ASSETS")
        r.line("FAIL", "SO-101 MuJoCo model",
               f"not in $ROBOT_USE_SO101_ASSETS ({where}): fix that path, or unset it and run "
               "python tools/fetch_so101_assets.py" if where else "not fetched; run: python tools/fetch_so101_assets.py")
    except Exception as e:
        r.line("FAIL", "SO-101 MuJoCo model", f"{type(e).__name__}: {e}")


def check_rig(r: Report):
    try:
        from stack.rig.rig import load_rig
        rig = load_rig()
        c = rig.camera("ego")
        missing = [str(rig.camera_file("ego", k)) for k in ("intrinsics_file", "extrinsics_file")
                   if k in c and not rig.camera_file("ego", k).is_file()]
        if missing:
            r.line("FAIL", "camera calibration", f"missing {', '.join(missing)} (robot.yaml names them)")
            return
        _, src = rig.camera_model("ego")
        est = [k for k, v in src.items() if v == "estimated"]
        if est:
            r.line("WARN", "ego camera", f"its {' and '.join(est)} are estimates from robot.yaml: the stack runs, "
                   "positions are approximate (calibrate: python tools/calibrate.py, docs/CALIBRATION.md)")
        else:
            r.line("ok", "robot.yaml and camera calibration", str(rig.path))
    except Exception as e:
        r.line("FAIL", "robot.yaml", f"{type(e).__name__}: {e}")


def check_rendering(r: Report):
    try:
        from stack.twin_mujoco.render import find_gl_env
    except Exception as e:
        r.line("FAIL", "rendering", f"cannot check ({type(e).__name__}: {e})")
        return
    print("  ....  rendering: probing OpenGL settings (a few seconds)...", flush=True)
    env = find_gl_env()
    if env is None:
        hint = ("install Mesa's OSMesa (Debian/Ubuntu: sudo apt install libosmesa6) or use a machine with a GPU "
                "driver" if sys.platform.startswith("linux") else "MuJoCo's CGL backend should work on macOS")
        r.line("WARN", "rendering", f"no OpenGL setting renders here: the sim runs, its cameras can't; {hint}")
    else:
        r.line("ok", "rendering", " ".join(f"{k}={v}" for k, v in env.items()) or "the default GL")


def check_ports(r: Report):
    for port, name in PORTS.items():
        from launch_common import port_free                 # the launchers' own test of a port
        if port_free(port):
            r.line("ok", f"port {port} ({name})", "free")
        else:
            r.ports_busy = True
            r.line("WARN", f"port {port} ({name})", "in use: a robotd or MCP server is running (stop it before "
                   "tools/run_sim.py, or use its --port-base)")


def check_sandbox(r: Report):
    try:
        from stack.runner.sandbox import choose_box
        box, why = choose_box("sim")
    except Exception as e:
        r.line("FAIL", "program sandbox", f"cannot check ({type(e).__name__}: {e})")
        return
    if box is None:
        r.line("FAIL", "program sandbox", f"{why}: agent programs would be refused (needs Linux with Landlock, "
               "6.7+ for the real arms, or macOS)")
    else:
        hw, _ = choose_box("hardware")
        r.line("ok", "program sandbox", why + ("" if hw else "; the real arms need Landlock ABI 4 (Linux 6.7+)"))


def check_agent_settings(r: Report):
    try:
        import setup_agent_ws
        from pathlib import Path as P
        import json
        want = json.dumps(setup_agent_ws.render_claude(REPO, P.home().resolve()), indent=2, ensure_ascii=False) + "\n"
        have = setup_agent_ws.CLAUDE_OUT.read_text() if setup_agent_ws.CLAUDE_OUT.exists() else None
        pi_ok = setup_agent_ws.PI_OUT.exists() and setup_agent_ws.PI_OUT.read_text() == setup_agent_ws.render_pi(REPO)
    except Exception as e:
        r.line("FAIL", "agent settings", f"cannot check ({type(e).__name__}: {e})")
        return
    if have is None:
        r.line("FAIL", "agent settings", "not written yet: python tools/setup_agent_ws.py")
    elif have != want or not pi_ok:
        r.line("WARN", "agent settings", "out of date (new folders, or the template changed): "
               "python tools/setup_agent_ws.py")
    else:
        r.line("ok", "agent settings", str(setup_agent_ws.CLAUDE_OUT.relative_to(REPO)))


def check_agents(r: Report):
    found = [c for c in ("claude", "codex") if shutil.which(c)]
    if found:
        r.line("ok", "agent CLI", ", ".join(found))
    else:
        r.line("WARN", "agent CLI", "neither claude nor codex on PATH: install Claude Code (or Codex) to let an agent "
               "drive the sim; the sim itself runs without one")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hardware", action="store_true", help="also check what the real arms need")
    a = ap.parse_args()
    r = Report()
    print("robot-use doctor")
    check_python(r)
    check_packages(r, CORE, required=True)
    check_assets(r)
    check_rig(r)
    check_rendering(r)
    check_ports(r)
    check_sandbox(r)
    check_agent_settings(r)
    check_agents(r)
    if a.hardware:
        check_packages(r, HARDWARE, required=True, why=" (the real arms need it)",
                       fix="uv sync --extra dev --extra hardware (on a Jetson: docs/environment.md)")
    if r.failed:
        print(f"{r.failed} problem(s) to fix first")
    else:
        print("all set: python tools/run_sim.py" + ("  (the usual ports are taken: stop what holds them, or add "
                                                     "--port-base 8790)" if r.ports_busy else ""))
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
