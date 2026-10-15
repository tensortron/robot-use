"""Entry point of a program's process: install the sandbox, then run the program.

    python -m stack.runner.child <program.py>      (started by the runner, never by hand)

Layer two of the sandbox (layer one is static_check). After the allowed modules are imported, an audit
hook refuses: opening files for writing, sockets, subprocesses and exec, fork, signals to other
processes, ctypes, and importing anything outside the allowed set. The program also gets builtins
without open/exec/eval/compile and an __import__ that only admits the allowed modules.
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import traceback

# allowed modules are imported BEFORE the hook, so their own lazy internals are already loaded
import collections, copy, dataclasses, enum, functools, itertools, math, random, re, statistics, time, typing  # noqa
import numpy  # noqa: F401

# the whole library and what it loads lazily (OpenCV and the rig description for geometry), before the hook
import cv2  # noqa: E402,F401
import yaml  # noqa: E402,F401
import robot_use  # noqa: E402,F401
from robot_use import runtime  # noqa: E402
from stack.runner import sandbox  # noqa: E402
from stack.rig import camera_model, rig  # noqa: E402,F401

ALLOWED = {"robot_use", "math", "numpy", "time", "json", "random", "itertools", "functools", "collections",
           "dataclasses", "typing", "statistics", "enum", "copy", "re"}
BLOCKED_EVENTS = ("pickle.find_class", "marshal.load", "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                  "os.forkpty", "pty.", "os.kill", "os.killpg", "ctypes.", "os.remove", "os.rename",
                  "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.truncate", "shutil.", "os.link",
                  "os.symlink", "os.unlink", "webbrowser.")
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _hook(event, args):
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int):
            return
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WRITE_FLAGS):
            raise PermissionError(f"blocked by the runner sandbox: writing {path!r}")
        return
    if event == "import":
        top = str(args[0]).split(".")[0]
        if top not in ALLOWED and top not in sys.modules:
            raise ImportError(f"blocked by the runner sandbox: import {args[0]!r}")
        return
    if event.startswith(BLOCKED_EVENTS):
        raise PermissionError(f"blocked by the runner sandbox: {event}")


def _restricted_builtins():
    b = dict(vars(builtins))
    for name in ("open", "exec", "eval", "compile", "breakpoint", "input", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "memoryview"):
        b.pop(name, None)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED:
            raise ImportError(f"import {name!r} is not allowed in robot programs")
        return real_import(name, globals, locals, fromlist, level)
    b["__import__"] = guarded_import
    return b


def main() -> int:
    path = sys.argv[1]
    code = open(path).read()                                  # before the hook: read-only anyway
    runtime._chan = runtime._Channel()                        # open the RPC pipes now, then drop the
    del runtime.os                                            # runtime's handle on os (review 2026-10-03)
    try:                                                      # the ego camera model, while files may be read
        from robot_use import geometry
        geometry._camera(geometry.EGO)                        # cached under the argument geometry passes
    except Exception:
        pass                                                  # uncalibrated: geometry calls will say so
    box = os.environ.get(sandbox.BOX_ENV, sandbox.LANDLOCK if sys.platform.startswith("linux") else None)
    try:
        sandbox.apply_in_child(box, path)                     # layer 3: the kernel's read-only, no-network box
    except Exception as e:
        runtime._rpc("finish", outcome="error", summary=f"sandbox unavailable: {e}",
                     error={"type": "SandboxUnavailable", "message": str(e)})
        return 1
    sys.addaudithook(_hook)
    g = {"__name__": "__main__", "__builtins__": _restricted_builtins()}
    try:
        exec(compile(code, "program.py", "exec"), g)
    except SystemExit:
        return 0                                              # done() / give_up() already reported
    except runtime.ProgramAborted:
        return 3
    except BaseException as e:                                # report the failure, then exit non-zero
        tb = traceback.format_exc(limit=8)
        try:
            runtime._rpc("finish", outcome="error", summary=f"{type(e).__name__}: {e}",
                         error={"type": type(e).__name__, "message": str(e), "traceback": tb,
                                **({"result": e.result} if isinstance(e, runtime.StepFailed) else {})})
        except BaseException:
            sys.stderr.write(tb)
        return 1
    try:
        runtime._rpc("finish", outcome="done", summary="")
    except runtime.ProgramAborted:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
