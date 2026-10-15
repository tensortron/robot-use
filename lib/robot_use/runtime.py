"""The program side of the runner: how an agent program talks to the robot (docs/ARCHITECTURE.md).

A program runs in its own process with no network, file-write or device access. Everything it does to
the world goes through these calls, each one a JSON message to the runner over a pipe; the runner holds
the lease and talks to robotd. Higher-level helpers (arms, estimates, geometry) build on this module.

    step(kind, **args)   one motion: blocks until it ends; returns the result, raises StepFailed unless it
                         ended `done` or `condition_met`
    state()              both arms' state
    checkpoint(...)      suspend with the arm holding; the agent looks and decides (continue / refine / abort)
    log(msg), done(summary), give_up(reason)
"""

from __future__ import annotations

import json
import os

OK_STATUSES = ("done", "condition_met")


class StepFailed(Exception):
    """A step ended in anything but done / condition_met. ``result`` is robotd's full motion record."""

    def __init__(self, result: dict):
        self.result = result
        super().__init__(f"{result.get('kind', 'step')} {result.get('status')}: {result.get('reason')}")


class ProgramAborted(BaseException):
    """The runner ended the program (stop, budget, a checkpoint abort). BaseException on purpose: a
    program's `except Exception:` cannot swallow it."""


class _Channel:
    def __init__(self):
        fds = os.environ.get("ROBOT_USE_RPC_FDS")
        if not fds:
            raise RuntimeError("not running under the robot-use runner (no ROBOT_USE_RPC_FDS)")
        r, w = (int(v) for v in fds.split(","))
        self.rx = os.fdopen(r, "r", buffering=1)
        self.tx = os.fdopen(w, "w", buffering=1)

    def call(self, op: str, **kw):
        self.tx.write(json.dumps({"op": op, **kw}) + "\n")
        self.tx.flush()
        line = self.rx.readline()
        if not line:
            raise ProgramAborted("the runner closed the channel")
        reply = json.loads(line)
        if reply.get("abort"):
            raise ProgramAborted(reply.get("reason", "aborted by the runner"))
        if not reply.get("ok"):
            raise RuntimeError(reply.get("reason", "runner error"))
        return reply.get("result")


_chan: _Channel | None = None


def _rpc(op: str, **kw):
    global _chan
    if _chan is None:
        _chan = _Channel()
    return _chan.call(op, **kw)


def step(kind: str, **args) -> dict:
    result = _rpc("step", kind=kind, args=args)
    if result.get("status") not in OK_STATUSES:
        raise StepFailed(result)
    return result


def state() -> dict:
    return _rpc("state")


def reach(arm: str, points: list) -> dict:
    """Read-only: can the arm's tip be at each point (and at which pitches)? No motion, not a step."""
    return _rpc("reach", arm=arm, points=list(points))


def view_pose(arm: str, target, distance: float = 0.15, elevations=None, start: int = 0) -> dict:
    """Read-only: a gripper pose from which the arm's wrist camera looks at target. No motion, not a step."""
    return _rpc("view_pose", arm=arm, target=[float(v) for v in target], distance=float(distance),
                elevations=None if elevations is None else [float(e) for e in elevations], start=int(start))


def checkpoint(name: str, estimates: dict | None = None, views=("ego",), need: str = "") -> dict:
    """Suspend here. ``estimates``: {name: {"value": [...], "tol": metres}}. Returns
    {"decision": "continue" | "refine", "estimates": {name: [...]}} with refined values where the agent
    refined them (each within its own tolerance); an abort raises ProgramAborted."""
    return _rpc("checkpoint", name=name, estimates=estimates or {}, views=list(views), need=need)


def log(msg: str) -> None:
    _rpc("log", msg=str(msg))


def done(summary: str = "") -> None:
    _rpc("finish", outcome="done", summary=str(summary))
    raise SystemExit(0)


def give_up(reason: str) -> None:
    _rpc("finish", outcome="give_up", summary=str(reason))
    raise SystemExit(0)
