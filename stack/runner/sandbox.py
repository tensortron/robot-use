"""Layer 3 of the program sandbox, per platform: the kernel's box round a program's process.

The rule is the same everywhere: the program may read only beneath the paths `read_paths()` gives (the Python
install and libraries, the repository, a few devices), may write nothing, and may not use the network. Layers 1
and 2 (static_check, the audit hook in child.py) see only Python; this layer is enforced by the kernel.

- **Linux, Landlock ABI 4** (6.7+): `landlock.restrict`, files and TCP. Required for the real arms.
- **Linux, Landlock ABI 1-3**, the simulated rig only: Landlock's file rules; TCP is left to the audit hook
  (sprint 2026-10-13, D14). robotd tells the runner which backend it runs, so this is never chosen for hardware.
- **macOS: Seatbelt**, through `sandbox-exec` with `seatbelt_profile()`: reading only beneath the same paths, no
  writes, no network, no fork or exec. The runner starts the child inside it; the child checks it is inside one
  (`sandbox_check`) and refuses to run the program otherwise. Untested on this Jetson: macOS CI proves it.
- Anything else: no box, and the child refuses to run programs, as before.

The runner chooses (`choose_box`) and tells the child through ROBOT_USE_KERNEL_BOX in the child's environment, which the
runner builds from scratch, so nothing outside can set it.
"""

from __future__ import annotations

import os
import sys

BOX_ENV = "ROBOT_USE_KERNEL_BOX"
LANDLOCK, LANDLOCK_FS, SEATBELT = "landlock", "landlock-fs", "seatbelt"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINUX_SYSTEM = ("/usr", "/lib")
# where macOS keeps the system libraries, the dyld cache (a cryptex on 13+), frameworks, Homebrew's libraries (not
# its var/ or etc/) and time zones
DARWIN_SYSTEM = ("/usr/lib", "/usr/share", "/System", "/Library/Frameworks", "/Library/Developer/CommandLineTools",
                 *(f"{b}/{d}" for b in ("/opt/homebrew", "/usr/local") for d in ("Cellar", "opt", "lib", "Frameworks")),
                 "/private/var/db/dyld", "/private/var/db/timezone", "/System/Volumes/Preboot/Cryptexes",
                 "/private/etc/localtime")
# the few system services a Python process may still look up under Seatbelt (logging, user and notification lookups)
DARWIN_MACH_ALLOW = ("com.apple.system.logger", "com.apple.logd", "com.apple.system.notification_center",
                     "com.apple.system.opendirectoryd.libinfo", "com.apple.system.opendirectoryd.membership")
DEVICES = ("/dev/null", "/dev/urandom", "/dev/zero")


def read_paths(extra=()) -> list[str]:
    """What a running program may still read: the Python install and its site-packages, the repository, the
    system's shared libraries, a few devices, and `extra` (the program file). Not the home directory (keys,
    tokens), not /proc of other processes. Independent of the caller's working directory and sys.path (the MCP
    server computes it for macOS, and may run from anywhere): a sys.path entry counts only inside these."""
    import site
    system = DARWIN_SYSTEM if sys.platform == "darwin" else LINUX_SYSTEM
    roots = {REPO, sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages()}
    user_site = site.getusersitepackages()
    if user_site in sys.path:                           # a non-venv install may keep packages there
        roots.add(user_site)
    roots = {r for r in roots if r and os.path.exists(r)}
    inside = [p for p in sys.path if p and os.path.exists(p)
              and any(os.path.realpath(p).startswith(os.path.realpath(r) + os.sep) or
                      os.path.realpath(p) == os.path.realpath(r) for r in roots)]
    paths = set()
    for p in (*roots, *inside, *system, *DEVICES, *extra):
        if p:
            paths.add(p)
            if os.path.exists(p):
                paths.add(os.path.realpath(p))          # Seatbelt and Landlock both see resolved paths
    return sorted(paths)


def choose_box(backend: str | None) -> tuple[str | None, str]:
    """(box, why) for a program on this machine, given robotd's backend ("hardware", "mock" or "sim"; None if
    unknown, treated as hardware). box None: there is none, and `why` says what is missing."""
    if sys.platform.startswith("linux"):
        from . import landlock
        abi = landlock.abi_version()
        if abi >= 4:
            return LANDLOCK, f"Landlock ABI {abi}"
        if abi >= 1 and backend == "sim":
            return LANDLOCK_FS, f"Landlock ABI {abi}: files only, the simulated rig (TCP refused by the audit hook)"
        return None, (f"Landlock ABI {abi}; programs need ABI 4 (Linux 6.7+)"
                      + ("" if backend == "sim" else ", or ABI 1+ for the simulated rig"))
    if sys.platform == "darwin":
        if os.path.exists(SANDBOX_EXEC):
            return SEATBELT, "Seatbelt (sandbox-exec)"
        return None, f"{SANDBOX_EXEC} not found"
    return None, f"no kernel sandbox for {sys.platform}"


def _sb_quote(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pythons(python: str) -> list[str]:
    """The executables starting the child may exec: the given Python, the base interpreter, their resolved paths,
    and a framework build's real binary (python.org and Homebrew stubs re-exec Python.app)."""
    out = {python, os.path.realpath(python)}
    base = getattr(sys, "_base_executable", None)
    if base:
        out |= {base, os.path.realpath(base)}
    app = os.path.join(sys.base_prefix, "Resources", "Python.app", "Contents", "MacOS", "Python")
    if os.path.exists(app):
        out |= {app, os.path.realpath(app)}
    return sorted(out)


SEATBELT_GROUPS = ("network", "write", "read", "fork", "exec", "mach", "appleevent", "iokit", "signal",
                   "procinfo", "procargs")


def seatbelt_rules(paths, python: str) -> dict[str, list[str]]:
    """The profile's rules by group (tools/seatbelt_diagnose.py leaves each out, and tries each alone). Each deny is
    followed by its exceptions: in SBPL the last matching rule wins."""
    reads = " ".join(f"(subpath {_sb_quote(p)})" if os.path.isdir(p) else f"(literal {_sb_quote(p)})"
                     for p in paths if os.path.exists(p))
    execs = " ".join(f"(literal {_sb_quote(p)})" for p in _pythons(python))
    mach = " ".join(f"(global-name {_sb_quote(n)})" for n in DARWIN_MACH_ALLOW)
    return {
        "network": ["(deny network*)"],
        "write": ["(deny file-write*)",
                  '(allow file-write-data (literal "/dev/null") (literal "/dev/dtracehelper"))'],
        # (literal "/") is the root directory alone (its listing), nothing beneath: macOS's loader opens "/" as
        # the root for its openat(2) lookups (Apple's dyld-support.sb, "libignition"), and without it every
        # process aborts in dyld before main (found on macOS 27 with tools/seatbelt_diagnose.py)
        "read": ["(deny file-read-data)", f'(allow file-read-data (literal "/") {reads})'],
        "fork": ["(deny process-fork)"],
        "exec": ["(deny process-exec*)", f"(allow process-exec* {execs})"],
        "mach": ["(deny mach-lookup)", f"(allow mach-lookup {mach})"],
        "appleevent": ["(deny appleevent-send)"],
        "iokit": ["(deny iokit-open)"],
        "signal": ["(deny signal)", "(allow signal (target self))"],
        "procinfo": ["(deny process-info*)", "(allow process-info* (target self))"],   # other processes' args, env
        "procargs": ['(deny sysctl-read (sysctl-name-prefix "kern.procargs"))'],      # the same, by sysctl
    }


def seatbelt_profile(paths, python: str, skip=()) -> str:
    """The macOS profile. Everything else the child does stays allowed, but not: reading file contents outside
    `paths`, writing anything but /dev/null, the network, fork, exec (beyond the Python that runs it), system
    services (mach-lookup, beyond a few a Python process needs: no LaunchServices, no camera), Apple events
    (scripting other apps), IOKit (devices), signals to and information about other processes. `skip`: groups to
    leave out (diagnosis only)."""
    rules = seatbelt_rules(paths, python)
    return "\n".join(["(version 1)", "(allow default)"]
                     + [line for g in SEATBELT_GROUPS if g not in skip for line in rules[g]]) + "\n"


def wrap_command(cmd: list[str], box: str | None, program: str) -> list[str]:
    """The child's command line inside its box: macOS starts it under sandbox-exec; Linux applies Landlock from
    inside the child (child.py), so the command is unchanged."""
    if box == SEATBELT:
        return [SANDBOX_EXEC, "-p", seatbelt_profile(read_paths(extra=(program,)), cmd[0]), *cmd]
    return cmd


def apply_in_child(box: str | None, program: str) -> str:
    """In the child, before the program runs: enter (Linux) or confirm (macOS) the box. Raises if there is none,
    so the program is refused rather than run unboxed. Returns a description."""
    if box in (LANDLOCK, LANDLOCK_FS):
        from . import landlock
        abi = landlock.restrict(read_paths(), min_abi=4 if box == LANDLOCK else 1)   # program read already
        return f"landlock ABI {abi}"
    if box == SEATBELT:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.sandbox_check.restype = ctypes.c_int
        if lib.sandbox_check(os.getpid(), None, 0) == 0:        # 0: not sandboxed
            raise RuntimeError("expected to run under sandbox-exec, but this process has no sandbox")
        return "seatbelt"
    raise RuntimeError(f"no kernel sandbox for programs here ({box!r})")
