"""Twin preview: run a program in the MuJoCo twin before a human is asked to approve it.

A private robotd (same safety configuration as the real one; no human approval inside the twin) runs on
SimArmIO, on a free localhost port; a ProgramRunner runs the program against it, continuing automatically
at checkpoints. The scene holds the objects the program believes in (its own estimates), so the preview
tests the program's logic, reach, safety rejections and contacts - not whether its beliefs are right.

Real time: a preview takes as long as the program's motions (the robotd loop runs on the wall clock).
"""

from __future__ import annotations

import dataclasses
import socket
import tempfile
import time
from pathlib import Path

from stack.rig.rig import load_rig
from stack.robotd.config import ArmConfig, Endpoint, load_config
from stack.robotd.daemon import EventLog, RobotD
from stack.robotd.http import serve
from stack.robotd.safety import load_arm_limits, rest_pose
from stack.runner.runner import END_STATES, Budgets, ProgramRunner

from .scene import build_scene
from .sim import SimArmIO, SimWorld


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# What a preview can and cannot tell (sprint 2026-10-07, C6): on 2026-10-06 the twin's object report matched the
# real grasp outcome in only 7 of 12 programs; reach, plan and safety results were reliable.
PREVIEW_NOTE = ("the twin checks logic, reach, safety and contacts with objects where the program believes they are - "
                "those results are reliable. Its grasp physics are not: whether an object ends up moved, held or "
                "dropped often differs from the real robot. Scene objects are placed square (no rotation).")


def preview_program(code: str, scene=None, timeout_s: float = 240.0, limits=None, work_dir=None,
                    backend: str = "hardware") -> dict:
    """Run `code` in the twin. Returns a report for the human and the agent; "ok" is True only if it ended done.

    `backend` is that of the robotd the program is meant for, not the twin's: the program's sandbox is chosen
    from it (stack/runner/sandbox.py), so a preview for real arms gets the same box as the real run, beside a
    robotd it must not reach (review of sprint 2026-10-13 task 7). Unknown means hardware."""
    t_start = time.time()
    rig = load_rig()
    base = load_config()
    names = tuple(limits) if limits else ("left", "right")      # the arms robotd has (one or two)
    work = Path(work_dir or tempfile.mkdtemp(prefix="robot_use_twin_"))
    cfg = dataclasses.replace(
        base, motion_enabled=True, approval_mode="none", mcp_localhost_only=True, cameras={}, overlay=None,
        backend=backend,
        log_dir=work / "robotd_log", idle_rest_s=1e9, api=Endpoint("127.0.0.1", _free_port()),
        arms={a: ArmConfig(a, "twin", rig.arm(a)["calibration_id"], rig.arm_mount(a)) for a in names})
    limits = limits or {a: load_arm_limits(rig.arm(a)["calibration_id"]) for a in names}
    objects = list(scene or [])
    world = None
    srv = rd = None
    try:
        from stack.robotd.safety import ArmModel
        from stack.rig.kinematics import ArmKinematics
        start = {a: rest_pose(ArmModel(a, ArmKinematics(rig.arm_mount(a)), limits[a]), cfg.safety) for a in limits}
        world = SimWorld(build_scene(rig, objects), start=start)
        world.start()
        initial = {o["name"]: world.object_pose(o["name"]) for o in objects}
        rd = RobotD(cfg, {a: SimArmIO(world, a) for a in names}, {}, limits=limits,
                    event_log=EventLog(cfg.log_dir))
        rd.start()
        srv = serve(rd, cfg.api.host, cfg.api.port, operator=False)
        runner = ProgramRunner(f"http://{cfg.api.host}:{cfg.api.port}", work / "episodes",
                               Budgets(wall_clock_s=timeout_s, approval_timeout_s=10, checkpoint_timeout_s=10),
                               preview=lambda c, s=None: {"twin": "self"}, holder="twin-preview")
        out = runner.submit(code, title="twin preview")
        if not out.get("ok"):
            return {"twin": "mujoco", "ok": False, "status": out.get("status"), "reason": out.get("reason")}
        pid = out["program_id"]
        deadline = time.monotonic() + timeout_s + 30
        checkpoints = []
        while time.monotonic() < deadline:
            st = runner.status(pid, wait_s=5)
            if st["status"] == "checkpoint":
                checkpoints.append(st["checkpoint"]["name"])
                runner.resume(pid, "continue")
            elif st["status"] in END_STATES:
                break
        else:
            runner.stop(pid, "twin preview timed out")
            st = runner.status(pid, wait_s=10)
        failures = []
        import json
        for line in (Path(st["dir"]) / "trace.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "step" and (r.get("result") or {}).get("status") not in ("done", "condition_met"):
                failures.append({"step": r["n"], "kind": r["kind"], "status": r["result"].get("status"),
                                 "reason": r["result"].get("reason")})
        report = {"twin": "mujoco (headless)", "ok": st["status"] == "done", "status": st["status"],
                  "reason": st.get("reason"), "steps": st["steps"], "motion_s": round(st["motion_s"], 1),
                  "checkpoints_passed": checkpoints, "step_failures": failures,
                  "objects": {n: {"initial": [round(v, 4) for v in initial[n]],
                                  "final": [round(v, 4) for v in world.object_pose(n)]} for n in initial},
                  "trace_dir": st["dir"], "wall_s": round(time.time() - t_start, 1),
                  "note": PREVIEW_NOTE}
        return report
    except Exception as e:
        return {"twin": "mujoco", "ok": False, "status": "error", "reason": f"{type(e).__name__}: {e}"}
    finally:
        if srv is not None:
            srv.shutdown()
        if rd is not None:
            rd.stop()
        if world is not None:
            world.stop()


PREVIEW_NICE = 10            # below robotd: a preview must never starve the real robot's 50 Hz loop


HARD_ROUNDING_DEG = 0.05                       # half of /limits' 0.1 deg rounding


def robotd_hard_limits(robotd_url: str) -> dict | None:
    """The calibrated (hard) joint ranges the running robotd enforces, per arm: {arm: {joint: [lo, hi]}}, or None.
    The preview then checks against the daemon's own ranges (mock, simulated or real arms), and needs no
    calibration files of its own (packaging sprint 2026-10-13). /limits rounds to 0.1 deg, so each end is moved
    in by 0.05 deg: the twin is never looser than robotd, at most 0.1 deg stricter."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(robotd_url.rstrip("/") + "/limits", timeout=5) as r:
            arms = json.loads(r.read())["arms"]
        return {a: {j: [float(v[0]) + HARD_ROUNDING_DEG, float(v[1]) - HARD_ROUNDING_DEG]
                    for j, v in arms[a]["hard"].items()} for a in arms}
    except Exception:
        return None


def robotd_backend(robotd_url: str) -> str:
    """What the robotd at `robotd_url` drives ("hardware", "mock", "sim"); "hardware" if it does not say."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(robotd_url.rstrip("/") + "/state", timeout=5) as r:
            return str(json.loads(r.read()).get("backend") or "hardware")
    except Exception:
        return "hardware"


def twin_preview(code: str, scene=None, timeout_s: float = 300.0, robotd_url: str | None = None) -> dict:
    """The ProgramRunner preview hook. Runs the preview in its own process at lower CPU priority. With
    `robotd_url`, the twin uses that robotd's joint ranges; otherwise the calibration files.

    It used to run inside the MCP server process: MuJoCo and a private robotd then shared the CPU (and
    Python's interpreter lock) with everything else, and on 2026-10-03 the real robotd's loop overran 61 ms
    and aborted a real motion while the agent's previews ran."""
    import json
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": f"{repo / 'lib'}:{repo}"}
    req = {"code": code, "scene": scene or []}
    if robotd_url and (lim := robotd_hard_limits(robotd_url)):
        req["limits"] = lim
    req["backend"] = robotd_backend(robotd_url) if robotd_url else "hardware"
    try:
        r = subprocess.run([sys.executable, "-m", "stack.twin_mujoco.preview"],
                           input=json.dumps(req), capture_output=True, text=True,
                           timeout=timeout_s, cwd=str(repo), env=env)
    except subprocess.TimeoutExpired:
        return {"twin": "mujoco", "ok": False, "status": "error", "reason": f"preview took over {timeout_s:.0f} s"}
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        return {"twin": "mujoco", "ok": False, "status": "error",
                "reason": f"preview process failed (exit {r.returncode}): {r.stderr.strip()[-400:]}"}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"twin": "mujoco", "ok": False, "status": "error", "reason": "preview process printed no valid report"}


def _main() -> int:
    """stdin: {"code", "scene"}; stdout: the preview report as one JSON line."""
    import json
    import sys

    import os

    os.nice(PREVIEW_NICE)                       # here, not via preexec_fn (unsafe in the threaded MCP server)
    req = json.loads(sys.stdin.read())
    from stack.robotd.safety import ArmLimits
    limits = {a: ArmLimits({j: tuple(v) for j, v in lim.items()}) for a, lim in req["limits"].items()} \
        if req.get("limits") else None
    rep = preview_program(req["code"], req.get("scene"), limits=limits, backend=req.get("backend") or "hardware")
    print(json.dumps(rep, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
