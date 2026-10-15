"""The program runner (docs/ARCHITECTURE.md): runs one agent program at a time against robotd.

Lifecycle of a program:

    checking -> previewing -> awaiting_approval -> running [<-> checkpoint] -> done | failed | aborted
                                                          | stopped | denied | rejected | expired

- **checking**: static_check; problems reject the program before a human ever sees it.
- **previewing**: the preview hook (the twin, step 5); its summary goes to the human with the code.
- **awaiting_approval**: robotd's operator page shows the code; the human approves (optionally in step
  mode) or denies.
- **running**: the runner holds robotd's lease (renewed every second; if this process dies, the lease
  expires and robotd stops the arms), binds it to the approved program, and starts the program in its own
  sandboxed process. Every step goes through here: budget checks, keyframes before and after, the robotd
  motion, the trace record, the result back to the program.
- **checkpoint**: the program is suspended with the arm holding; the evidence waits for the agent's
  decision (continue / refine within tolerance / abort) through resume().

Everything is written to <episodes>/<date>/<program_id>/: code.py, trace.jsonl, keyframes/, summary.json.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .. import paths
from . import sandbox, static_check

REPO = Path(__file__).resolve().parents[2]
MOTION_STEPS = {"move_joints": "move_joints", "move_pose": "move_pose", "set_gripper": "set_gripper",
                "trajectory": "trajectory", "go_rest": "go_rest"}
OK_STATUSES = ("done", "condition_met")
END_STATES = ("done", "failed", "aborted", "stopped", "denied", "rejected", "expired")
CAMERA_FOR_ARM = {"left": "wrist_left", "right": "wrist_right"}


@dataclass
class Budgets:
    max_steps: int = 50
    max_queries: int = 100                      # read-only reach / view_pose questions (a view_pose plans up to
                                                # 18 candidates, ~1.4 s of a shared planning worker)
    max_motion_s: float = 180.0
    wall_clock_s: float = 900.0                 # from start of execution (approval wait not counted)
    max_consecutive_failures: int = 2
    checkpoint_timeout_s: float = 900.0
    checkpoint_silence_s: float = 600.0         # abort a checkpoint once the client (agent) has been silent this long
    approval_timeout_s: float = 600.0
    step_margin_s: float = 30.0                 # beyond a motion's own duration before it counts as stuck
    lease_ttl_s: float = 3.0


class RobotdClient:
    def __init__(self, base: str = "http://127.0.0.1:8770", timeout_s: float = 10.0):
        self.base, self.timeout_s = base.rstrip("/"), timeout_s

    def _req(self, method, path, body=None, timeout=None, raw=False):
        if body is None and method == "POST":
            body = {}                                  # robotd requires JSON on every POST
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if raw:
                return None
            try:
                return {"ok": False, "status": "rejected", "reason": json.loads(e.read()).get("error", str(e))}
            except Exception:
                return {"ok": False, "status": "rejected", "reason": str(e)}
        except (urllib.error.URLError, OSError, ValueError) as e:
            # robotd unreachable: a failure the caller acts on (a lost lease aborts the program), never a crash
            return None if raw else {"ok": False, "status": "faulted", "reason": f"robotd unreachable: {e}"}

    def get(self, path, timeout=None):
        return self._req("GET", path, timeout=timeout)

    def post(self, path, body, timeout=None):
        return self._req("POST", path, body, timeout=timeout)

    def jpeg(self, path):
        try:
            return self._req("GET", path, raw=True)
        except (OSError, urllib.error.URLError):
            return None


@dataclass
class Program:
    program_id: str
    title: str
    code: str
    step_mode: bool
    dir: Path
    status: str = "checking"
    reason: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    preview: dict = field(default_factory=dict)
    steps: int = 0
    queries: int = 0
    motion_s: float = 0.0
    consecutive_failures: int = 0
    last_step: dict | None = None
    checkpoint: dict | None = None
    outcome: dict | None = None
    output: list = field(default_factory=list)
    lease: str | None = None
    scene: list = field(default_factory=list)   # the program's beliefs about objects, for the twin preview
    cleaned: bool = False                       # lease released and robotd told; only then may the next start

    def public(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("code", "dir", "lease", "output", "cleaned", "scene")}
        d["dir"] = str(self.dir)
        d["output_tail"] = self.output[-20:]
        return d


def default_preview(code: str, scene=None) -> dict:
    return {"twin": None, "note": "no twin preview configured; review the code"}


class ProgramRunner:
    def __init__(self, robotd: RobotdClient | str = "http://127.0.0.1:8770",
                 episodes_dir: Path | str | None = None, budgets: Budgets | None = None,
                 preview=default_preview, holder: str = "runner"):
        self.robotd = robotd if isinstance(robotd, RobotdClient) else RobotdClient(robotd)
        self.episodes = Path(episodes_dir).expanduser() if episodes_dir else paths.episodes()
        self.budgets = budgets or Budgets()
        self.preview = preview
        self.holder = holder
        self.programs: dict[str, Program] = {}
        self._lock = threading.Condition()
        self._resume: dict[str, dict] = {}
        self._stop_requested: set[str] = set()
        self._last_client: float | None = None          # monotonic time of the client's last call, if tracked

    def note_client_activity(self):
        """The client (the MCP server, for each agent tool call) is alive. A program waiting at a checkpoint is
        aborted once the client has been silent for budgets.checkpoint_silence_s: nobody is left to decide."""
        self._last_client = time.monotonic()

    # ------------------------------------------------------------------ public API (MCP tools wrap these)
    def submit(self, code: str, title: str = "", step_mode: bool = False, scene=None, skip_preview: bool = False,
               episode_dir: Path | str | None = None) -> dict:
        """scene: the objects the program expects, [{name, xy | xyz, size}], placed in the twin for the preview.
        skip_preview: run without the twin preview (the human is told it was skipped).
        episode_dir: file the program under this episode instead of <episodes>/<date>/."""
        with self._lock:
            busy = [p for p in self.programs.values() if p.status not in END_STATES or not p.cleaned]
            if busy:
                b = busy[0]
                what = b.status if b.status not in END_STATES else f"{b.status} (releasing the robot)"
                return {"ok": False, "reason": f"program {b.program_id} is still {what}; one at a time"}
            pid = f"p{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
            d = (Path(episode_dir) if episode_dir else self.episodes / datetime.now().strftime("%Y-%m-%d")) / pid
            (d / "keyframes").mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text(code)
            p = Program(pid, title or pid, code, bool(step_mode), d, scene=list(scene or []))
            p.preview = {"skipped": True} if skip_preview else {}
            self.programs[pid] = p
        problems = static_check.check(code)
        if problems:
            p.cleaned = True
            self._end(p, "rejected", "static check: " + "; ".join(problems))
            return {"ok": False, **p.public()}
        threading.Thread(target=self._run, args=(p,), name=f"runner-{pid}", daemon=True).start()
        return {"ok": True, **p.public()}

    def status(self, program_id: str, wait_s: float = 0.0) -> dict:
        """Waits up to wait_s for the program to reach a checkpoint or an end state."""
        deadline = time.monotonic() + min(max(wait_s, 0.0), 30.0)
        with self._lock:
            p = self.programs.get(program_id)
            if p is None:
                return {"ok": False, "reason": f"no program {program_id!r}"}
            while (p.status not in END_STATES and not (p.status == "checkpoint" and program_id not in self._resume)
                   and time.monotonic() < deadline):
                self._lock.wait(timeout=deadline - time.monotonic())
            return {"ok": True, **p.public()}

    def resume(self, program_id: str, decision: str, updates: dict | None = None) -> dict:
        with self._lock:
            p = self.programs.get(program_id)
            if p is None or p.status != "checkpoint" or not p.checkpoint:
                return {"ok": False, "reason": f"program {program_id!r} is not waiting at a checkpoint"}
            if decision not in ("continue", "refine", "abort"):
                return {"ok": False, "reason": "decision must be continue, refine or abort"}
            est = p.checkpoint["estimates"]
            new = {k: list(v["value"]) for k, v in est.items()}
            if decision == "refine":
                for name, val in (updates or {}).items():
                    if name not in est:
                        return {"ok": False, "reason": f"no estimate {name!r} at this checkpoint ({sorted(est)})"}
                    old, tol = est[name]["value"], float(est[name].get("tol", 0.0))
                    if not isinstance(val, (list, tuple)) or len(val) != len(old):
                        return {"ok": False, "reason": f"{name}: give {len(old)} values"}
                    delta = max(abs(float(a) - float(b)) for a, b in zip(val, old))
                    if delta > tol + 1e-12:
                        return {"ok": False, "reason": f"{name}: change of {delta:.4f} exceeds its tolerance {tol:.4f}; "
                                                       "a larger change needs a revised program (new approval)"}
                    new[name] = [float(v) for v in val]
            self._resume[program_id] = {"decision": decision, "estimates": new}
            self._lock.notify_all()
            return {"ok": True, "decision": decision, "estimates": new}

    def active(self) -> str | None:
        with self._lock:
            for pid, p in self.programs.items():
                if p.status not in END_STATES:
                    return pid
        return None

    def stop(self, program_id: str, reason: str = "stopped on request") -> dict:
        with self._lock:
            p = self.programs.get(program_id)
            if p is None or p.status in END_STATES:
                return {"ok": False, "reason": f"program {program_id!r} is not active"}
            self._stop_requested.add(program_id)
            self._resume[program_id] = {"decision": "abort", "reason": reason}
            self._lock.notify_all()
        self.robotd.post("/motion/stop", {"arm": "both"})
        return {"ok": True}

    # ------------------------------------------------------------------ lifecycle
    def _set(self, p: Program, status: str, reason: str | None = None):
        with self._lock:
            p.status, p.reason = status, reason
            self._lock.notify_all()
        self._trace(p, {"type": "status", "status": status, "reason": reason})

    def _end(self, p: Program, status: str, reason: str | None = None):
        p.ended = time.time()
        self._set(p, status, reason)
        (p.dir / "summary.json").write_text(json.dumps(p.public(), indent=1, default=str))

    def _trace(self, p: Program, rec: dict):
        rec = {"t": round(time.time(), 3), **rec}
        with open(p.dir / "trace.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def _run(self, p: Program):
        try:
            self._run_inner(p)
        finally:
            with self._lock:
                p.cleaned = True
                self._lock.notify_all()

    def _run_inner(self, p: Program):
        try:
            self._set(p, "previewing")
            if p.preview.get("skipped"):
                p.preview = {"twin": None, "skipped": True, "note": "the agent skipped the twin preview"}
            else:
                p.preview = self.preview(p.code, p.scene) or {}
            self._trace(p, {"type": "preview", "preview": p.preview})
            if p.preview.get("ok") is False:
                return self._end(p, "rejected", "preview: " + str(p.preview.get("reason")))
            r = self.robotd.post("/program/request", {"program_id": p.program_id, "holder": self.holder, "title": p.title,
                                                      "code": p.code, "preview": p.preview, "step_mode": p.step_mode})
            if not r.get("ok"):
                return self._end(p, "rejected", f"robotd: {r.get('reason')}")
            self._set(p, "awaiting_approval")
            deadline = time.monotonic() + self.budgets.approval_timeout_s
            while True:
                if p.program_id in self._stop_requested:
                    return self._end(p, "stopped", "stopped before approval")
                r = self.robotd.get(f"/program/{p.program_id}?wait_s=5", timeout=15)
                st = r.get("status")
                if st == "approved":
                    p.step_mode = bool(r.get("step_mode", p.step_mode))
                    break
                if st in ("denied", "expired"):
                    return self._end(p, st, r.get("reason"))
                if time.monotonic() > deadline:
                    return self._end(p, "expired", "not approved in time")
            self._execute(p)
        except Exception as e:                                   # the runner itself failed: stop the robot
            self.robotd.post("/motion/stop", {"arm": "both"})
            self._end(p, "failed", f"runner error: {type(e).__name__}: {e}")

    def _execute(self, p: Program):
        give_up = time.monotonic() + 5.0              # another program may be releasing it right now
        while True:
            lease = self.robotd.post("/lease/acquire", {"holder": f"{self.holder}:{p.program_id}",
                                                        "ttl_s": self.budgets.lease_ttl_s})
            if lease.get("ok") or time.monotonic() > give_up or "already leased" not in str(lease.get("reason")):
                break
            time.sleep(0.2)
        if not lease.get("ok"):
            return self._end(p, "failed", f"could not acquire the lease: {lease.get('reason')}")
        p.lease = lease["token"]
        alive = threading.Event()
        alive.set()

        def keep_lease():
            while alive.is_set():
                time.sleep(1.0)
                if alive.is_set() and not self.robotd.post("/lease/renew", {"token": p.lease}).get("ok"):
                    self._trace(p, {"type": "lease_lost"})
                    with self._lock:
                        self._resume[p.program_id] = {"decision": "abort", "reason": "lease lost"}
                        self._stop_requested.add(p.program_id)
                        self._lock.notify_all()
                    return
        threading.Thread(target=keep_lease, daemon=True).start()
        try:
            r = self.robotd.post("/program/start", {"program_id": p.program_id, "lease": p.lease})
            if not r.get("ok"):
                return self._end(p, "failed", f"robotd refused to start it: {r.get('reason')}")
            p.started = time.time()
            self._set(p, "running")
            self._spawn_and_serve(p)
        finally:
            alive.clear()
            self.robotd.post("/program/end", {"program_id": p.program_id, "lease": p.lease,
                                              "outcome": f"{p.status}: {p.reason or ''}"})
            self.robotd.post("/lease/release", {"token": p.lease})
            if p.status not in END_STATES:
                self._end(p, "failed", "ended without an outcome")
            elif p.ended is None:
                self._end(p, p.status, p.reason)

    # ------------------------------------------------------------------ the program process
    def _backend(self) -> str | None:
        """What robotd drives ("hardware", "mock", "sim"), for the sandbox choice; None (treated as hardware) if
        robotd does not say."""
        try:
            return self.robotd.get("/state", timeout=3).get("backend")
        except Exception:
            return None

    def _spawn_and_serve(self, p: Program):
        c2p_r, c2p_w = os.pipe()          # child -> runner requests
        p2c_r, p2c_w = os.pipe()          # runner -> child replies
        box, why = sandbox.choose_box(self._backend())
        self._trace(p, {"type": "sandbox", "box": box, "why": why})
        env = {"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONPATH": f"{REPO / 'lib'}:{REPO}",
               "ROBOT_USE_RPC_FDS": f"{p2c_r},{c2p_w}", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", sandbox.BOX_ENV: box or "none"}
        cmd = [sys.executable, "-m", "stack.runner.child", str(p.dir / "code.py")]
        proc = subprocess.Popen(sandbox.wrap_command(cmd, box, str(p.dir / "code.py")),
                                cwd=str(REPO), env=env, pass_fds=(p2c_r, c2p_w), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.close(p2c_r)
        os.close(c2p_w)
        rx = os.fdopen(c2p_r, "r", buffering=1)
        tx = os.fdopen(p2c_w, "w", buffering=1)

        def pump_output():
            for line in proc.stdout:
                line = line.rstrip("\n")
                p.output.append(line)
                del p.output[:-500]
                self._trace(p, {"type": "print", "line": line})
        threading.Thread(target=pump_output, daemon=True).start()

        deadline = time.monotonic() + self.budgets.wall_clock_s

        def watchdog():
            while proc.poll() is None:
                if time.monotonic() > deadline and p.status not in END_STATES:
                    self._abort(p, proc, "wall-clock budget exceeded")
                    return
                if p.program_id in self._stop_requested and p.status == "running":
                    self._abort(p, proc, "stopped on request", status="stopped")
                    return
                time.sleep(0.1)
        threading.Thread(target=watchdog, daemon=True).start()

        try:
            for line in rx:
                if p.status in END_STATES:
                    break
                try:
                    req = json.loads(line)
                except ValueError:
                    self._abort(p, proc, "malformed message from the program")
                    break
                reply = self._handle(p, req)
                if req.get("op") == "finish":
                    p.outcome = {k: req.get(k) for k in ("outcome", "summary", "error")}
                    status = {"done": "done", "give_up": "failed", "error": "failed"}[req.get("outcome", "error")]
                    if p.program_id in self._stop_requested:
                        status = "stopped"
                    self._end(p, status, req.get("summary") or None)
                try:
                    tx.write(json.dumps(reply) + "\n")
                    tx.flush()
                except (BrokenPipeError, ValueError):
                    break
                if reply.get("abort"):
                    self._abort(p, proc, reply.get("reason", "aborted"), status=reply.get("_status", "aborted"))
                    break
        finally:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            for f in (rx, tx):
                try:
                    f.close()
                except Exception:
                    pass
            if p.status in ("running", "checkpoint"):
                code = proc.returncode
                self._end(p, "failed", f"the program process exited ({code}) without reporting an outcome")

    def _abort(self, p: Program, proc, reason: str, status: str = "aborted"):
        if p.status in END_STATES:
            return
        self.robotd.post("/motion/stop", {"arm": "both"})
        self._end(p, status, reason)
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            proc.kill()

    # ------------------------------------------------------------------ requests from the program
    def _handle(self, p: Program, req: dict) -> dict:
        op = req.get("op")
        if p.program_id in self._stop_requested:
            return {"ok": False, "abort": True, "reason": "stopped on request", "_status": "stopped"}
        if op == "step":
            return self._step(p, req.get("kind"), req.get("args") or {})
        if op == "state":
            return {"ok": True, "result": self.robotd.get("/state")}
        if op in ("reach", "view_pose"):          # read-only questions; no motion, not a step
            if p.queries >= self.budgets.max_queries:      # a runaway search loop would keep robotd's planners busy
                return {"ok": False, "abort": True,
                        "reason": f"query budget exceeded ({self.budgets.max_queries} reach / view_pose calls)"}
            p.queries += 1
            body = {k: req.get(k) for k in (("arm", "points") if op == "reach"
                                            else ("arm", "target", "distance", "elevations", "start")) if req.get(k) is not None}
            res = self.robotd.post("/motion/check_reach" if op == "reach" else "/motion/view_pose", body)
            self._trace(p, {"type": op, "args": body, "result": res})
            return {"ok": True, "result": res}
        if op == "checkpoint":
            return self._checkpoint(p, req)
        if op == "log":
            self._trace(p, {"type": "log", "msg": req.get("msg")})
            return {"ok": True, "result": None}
        if op == "finish":
            return {"ok": True, "result": None}
        return {"ok": False, "reason": f"unknown request {op!r}"}

    def _keyframes(self, p: Program, tag: str, cams) -> list[str]:
        saved = []
        for cam in cams:
            q = "overlay=1&" if cam == "ego" else ""
            jpg = self.robotd.jpeg(f"/frame/{cam}?{q}max_side=640&quality=80")
            if jpg:
                f = p.dir / "keyframes" / f"{tag}_{cam}.jpg"
                f.write_bytes(jpg)
                saved.append(str(f.relative_to(p.dir)))
        return saved

    def _progress(self, p: Program, **where):
        """Tell robotd where the program is, for the operator page. Best effort: never holds up or fails a program."""
        try:
            self.robotd.post("/program/progress", {"program_id": p.program_id, "lease": p.lease, **where}, timeout=1)
        except Exception:
            pass

    def _step(self, p: Program, kind: str, args: dict) -> dict:
        b = self.budgets
        if kind not in MOTION_STEPS:
            return {"ok": False, "reason": f"unknown step {kind!r}; steps are {sorted(MOTION_STEPS)}"}
        if p.steps >= b.max_steps:
            return {"ok": False, "abort": True, "reason": f"step budget exceeded ({b.max_steps} steps)"}
        if p.motion_s >= b.max_motion_s:
            return {"ok": False, "abort": True, "reason": f"motion-time budget exceeded ({b.max_motion_s:.0f} s)"}
        p.steps += 1
        n = p.steps
        self._progress(p, step=n, kind=kind)
        arms = ["left", "right"] if args.get("arm") == "both" else [args.get("arm")]
        cams = ["ego"] + [CAMERA_FOR_ARM[a] for a in arms if a in CAMERA_FOR_ARM]
        t0 = time.time()
        before = self._keyframes(p, f"step{n:03d}_before", cams)
        body = {k: v for k, v in args.items() if k != "lease"} | {"lease": p.lease}
        out = self.robotd.post(f"/motion/{MOTION_STEPS[kind]}", body, timeout=30)
        result = out
        if out.get("status") != "rejected" and out.get("motion_id"):
            mid = out["motion_id"]
            run_s = float(out.get("duration_s") or 0) + b.step_margin_s
            limit = time.monotonic() + run_s
            pending_since = None
            while True:
                result = self.robotd.get(f"/motion/{mid}?wait_s=5", timeout=15)
                st = result.get("status")
                if st not in ("pending_approval", "running") or p.program_id in self._stop_requested:
                    break
                now = time.monotonic()
                if st == "pending_approval":                    # step mode, or an arm that was blocked
                    pending_since = pending_since or now
                    if now - pending_since > b.approval_timeout_s:
                        self.robotd.post("/motion/stop", {"arm": args.get("arm", "both")})
                        result = self.robotd.get(f"/motion/{mid}") | {"status": "stopped",
                                                                       "reason": "step was not approved in time"}
                        break
                    limit = now + run_s                         # the motion's own time starts once approved
                elif now > limit:
                    self.robotd.post("/motion/stop", {"arm": args.get("arm", "both")})
                    result = self.robotd.get(f"/motion/{mid}") | {"status": "stopped", "reason": "step timed out"}
                    break
            p.motion_s += float(out.get("duration_s") or 0)
            # anything robotd does after a step (a back-off after a block, a deceleration) belongs to it:
            # return only once these arms are idle, so the next step does not find them busy
            idle_by = time.monotonic() + 10.0
            while time.monotonic() < idle_by and p.program_id not in self._stop_requested:
                arms_state = (self.robotd.get("/state").get("arms") or {})
                if not any((arms_state.get(a) or {}).get("motion") for a in arms if a):
                    break
                time.sleep(0.05)
        after = self._keyframes(p, f"step{n:03d}_after", cams)
        ok = result.get("status") in OK_STATUSES
        p.consecutive_failures = 0 if ok else p.consecutive_failures + 1
        p.last_step = {"n": n, "kind": kind, "status": result.get("status"), "reason": result.get("reason")}
        self._trace(p, {"type": "step", "n": n, "kind": kind, "args": args, "t_start": round(t0, 3),
                        "t_end": round(time.time(), 3), "result": result, "keyframes": before + after})
        with self._lock:
            self._lock.notify_all()
        if p.program_id in self._stop_requested:          # STOP mid-step: end the program, not just the step
            return {"ok": True, "result": result, "abort": True, "reason": "stopped on request", "_status": "stopped"}
        if not ok and p.consecutive_failures >= b.max_consecutive_failures:
            return {"ok": True, "result": result, "abort": True,
                    "reason": f"{p.consecutive_failures} steps failed in a row (last: {result.get('status')}: "
                              f"{result.get('reason')}); stopping to avoid thrashing"}
        return {"ok": True, "result": result}

    def _checkpoint(self, p: Program, req: dict) -> dict:
        name = str(req.get("name") or "checkpoint")
        estimates = req.get("estimates") or {}
        for k, v in estimates.items():
            if not isinstance(v, dict) or "value" not in v:
                return {"ok": False, "reason": f"estimate {k!r} needs a value (and a tol)"}
        n = sum(1 for _ in (p.dir / "keyframes").glob("cp*_ego.jpg")) + 1
        views = [v for v in (req.get("views") or ["ego"]) if isinstance(v, str)]
        evidence = self._keyframes(p, f"cp{n:02d}_{name}", views)
        cp = {"name": name, "need": req.get("need", ""), "estimates": estimates, "evidence": evidence,
              "state": self.robotd.get("/state").get("arms"), "since": time.time()}
        self._trace(p, {"type": "checkpoint", **cp})
        self._progress(p, step=p.steps, checkpoint=name)
        cp_since = time.monotonic()                      # silence counts from the checkpoint, not before it
        with self._lock:
            p.checkpoint = cp
            self._resume.pop(p.program_id, None)
            p.status = "checkpoint"
            self._lock.notify_all()
            deadline = time.monotonic() + self.budgets.checkpoint_timeout_s
            silent = False
            while p.program_id not in self._resume and time.monotonic() < deadline:
                if (self._last_client is not None
                        and time.monotonic() - max(self._last_client, cp_since) > self.budgets.checkpoint_silence_s):
                    silent = True
                    break
                self._lock.wait(timeout=min(5.0, deadline - time.monotonic()))
            decision = self._resume.pop(p.program_id, None)
            p.checkpoint = None
            if p.status == "checkpoint":
                p.status = "running"
            self._lock.notify_all()
        if decision is None and silent:
            self._trace(p, {"type": "resume", "checkpoint": name, "decision": "abort",
                            "reason": "the agent went silent at the checkpoint"})
            return {"ok": False, "abort": True, "reason": f"no decision at checkpoint {name!r}: the agent has made no "
                                                          f"call for {self.budgets.checkpoint_silence_s:.0f} s (its "
                                                          f"session has probably ended)"}
        if decision is None:
            return {"ok": False, "abort": True, "reason": f"no decision at checkpoint {name!r} within "
                                                          f"{self.budgets.checkpoint_timeout_s:.0f} s"}
        self._trace(p, {"type": "resume", "checkpoint": name, **decision})
        self._progress(p, step=p.steps)
        if decision["decision"] == "abort":
            return {"ok": False, "abort": True, "reason": decision.get("reason", f"aborted at checkpoint {name!r}"),
                    "_status": "stopped" if p.program_id in self._stop_requested else "aborted"}
        return {"ok": True, "result": decision}
