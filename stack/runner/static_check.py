"""Static checks on an agent program before it is shown for approval (docs/ARCHITECTURE.md).

The first of two layers; the child process's audit hook is the second (child.py). Neither is a security
boundary against a determined adversary. Together with robotd's exclusive ports and lease they stop a
program from reaching the hardware other than through robot_use's steps, by accident or by a model's
"helpful" shortcut (importing lerobot, opening the serial port, shelling out).
"""

from __future__ import annotations

import ast
from functools import lru_cache

ALLOWED_MODULES = {
    "robot_use", "math", "numpy", "time", "json", "random", "itertools", "functools", "collections",
    "dataclasses", "typing", "statistics", "enum", "copy", "re",
}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
                   "input", "memoryview", "getattr", "setattr", "delattr"}
# things that evaluate strings or unpickle (review 2026-10-03: get_type_hints on a string annotation ran an
# expression the static check never saw); refused as names and as attributes
EVALUATING_NAMES = {"get_type_hints", "ForwardRef", "get_annotations", "singledispatch", "singledispatchmethod",
                    "loads", "load", "fromregex", "genfromtxt", "loadtxt", "literal_eval"}
# submodules of allowed packages that reach compilers, subprocesses or C-level I/O
DENIED_SUBMODULES = ("numpy.f2py", "numpy.testing", "numpy.ctypeslib", "numpy.distutils")
# always refused as attribute / imported names, whatever they are taken from
CORE_DENIED_ATTRS = {"os", "sys", "modules", "builtins", "subprocess", "importlib", "ctypes", "cv2", "socket",
                     "shutil", "pathlib", "io", "posix", "signal", "threading", "multiprocessing", "pickle", "marshal"}


def _module_ok(mod) -> bool:
    name = mod.__name__
    return name.split(".")[0] in ALLOWED_MODULES and not any(name == d or name.startswith(d + ".")
                                                              for d in DENIED_SUBMODULES)


@lru_cache(maxsize=1)
def denied_attrs() -> frozenset:
    """Every public attribute name under which an allowed module (or a module reachable from one) exposes a
    module a program must not reach, e.g. dataclasses.sys, typing.sys, json.codecs, numpy.f2py.

    Review 2026-10-03: `runtime.os.sys.modules["cv2"]` passed the check and OpenCV then wrote files and
    opened URLs in C, unseen by the audit hook. The names are refused on ANY object (an alias such as
    `m = runtime; m.os` defeats tracking which name holds a module), and private names are refused anyway."""
    import importlib
    import types

    deny, seen = set(CORE_DENIED_ATTRS), set()
    frontier = []
    for name in ALLOWED_MODULES:
        try:
            frontier.append(importlib.import_module(name))
        except ImportError:
            pass
    for _ in range(6):
        nxt = []
        for mod in frontier:
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            for a in dir(mod):
                if a.startswith("_"):
                    continue
                try:
                    v = getattr(mod, a)
                except Exception:
                    continue
                if isinstance(v, types.ModuleType):
                    (nxt.append(v) if _module_ok(v) else deny.add(a))
        frontier = nxt
    return frozenset(deny)


def _string_in(node) -> bool:
    return node is not None and any(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in ast.walk(node))


def check(code: str) -> list[str]:
    """Problems with the program, as 'line N: message'. Empty means it may proceed to preview/approval."""
    try:
        tree = ast.parse(code, filename="program.py")
    except SyntaxError as e:
        return [f"line {e.lineno}: syntax error: {e.msg}"]
    problems = []

    def bad(node, msg):
        problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    deny = denied_attrs()

    def denied_module(name: str) -> bool:
        return any(name == d or name.startswith(d + ".") for d in DENIED_SUBMODULES)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_MODULES:
                    bad(node, f"import {a.name!r} is not allowed (allowed: {', '.join(sorted(ALLOWED_MODULES))})")
                elif denied_module(a.name) or any(p in deny or p.startswith("_") for p in a.name.split(".")[1:]):
                    bad(node, f"import {a.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad(node, "relative imports are not allowed")
            elif (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                bad(node, f"import from {node.module!r} is not allowed")
            elif denied_module(node.module) or any(p in deny or p.startswith("_") for p in node.module.split(".")[1:]):
                bad(node, f"import from {node.module!r} is not allowed")
            else:
                for a in node.names:
                    if a.name.startswith("_"):
                        bad(node, f"{a.name!r} is private")
                    elif a.name in EVALUATING_NAMES or a.name in FORBIDDEN_NAMES:
                        bad(node, f"{a.name!r} is not available to robot programs")
                    elif a.name in deny or denied_module(f"{node.module}.{a.name}"):
                        bad(node, f"{a.name!r} from {node.module!r} is not available to robot programs")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad(node, f"{node.id!r} is not available to robot programs")
        elif isinstance(node, ast.Name) and node.id in EVALUATING_NAMES:
            bad(node, f"{node.id!r} evaluates or unpickles data; not available to robot programs")
        elif isinstance(node, ast.Attribute) and node.attr in EVALUATING_NAMES:
            bad(node, f"{node.attr!r} evaluates or unpickles data; not available to robot programs")
        elif isinstance(node, (ast.arg, ast.AnnAssign)) and _string_in(node.annotation):
            bad(node, "string annotations are not allowed (they can be evaluated as code)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _string_in(node.returns):
            bad(node, "string annotations are not allowed (they can be evaluated as code)")
        elif isinstance(node, ast.keyword) and node.arg == "allow_pickle":
            bad(node, "allow_pickle is not available to robot programs")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            bad(node, f"dunder attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            bad(node, f"{node.attr!r} is private")
        elif isinstance(node, ast.Attribute) and node.attr in deny:
            bad(node, f"attribute {node.attr!r} is not available to robot programs (it reaches a module outside "
                      f"the allowed set)")
    return problems
