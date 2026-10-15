"""Layer 3 for real, on whatever platform this runs: a probe inside the box tries each escape.

Linux: Landlock (as the child applies it). macOS: Seatbelt (sandbox-exec with the runner's profile). This is the
test that proves the macOS box (sprint 2026-10-13 D14): run it on a Mac (tools/platform_check.py, or CI).
Only layer 3 is in play here; the static check and the audit hook (layers 1 and 2) are tested elsewhere.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from stack.runner import sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]
START_S = 30                         # a probe that needs longer to import numpy/cv2/robot_use under the box: a stall

PROBE = textwrap.dedent("""
    import json, os, socket, sys, time
    t0 = time.monotonic()
    sys.path[:0] = [{lib!r}, {repo!r}]
    BOX, SECRET, OUT_DIR, PORT, OTHER = {box!r}, {secret!r}, {out_dir!r}, {port}, {other}
    if BOX in ("landlock", "landlock-fs"):
        from stack.runner import sandbox
        sandbox.apply_in_child(BOX, "probe")
    import numpy, cv2, robot_use                                  # what a program's process imports
    from stack.rig import camera_model, rig
    res = {{"import_s": round(time.monotonic() - t0, 2)}}

    def attempt(name, fn):
        try:
            fn()
            res[name] = "ALLOWED"
        except OSError as e:                                     # refused: PermissionError, EPERM, ...
            res[name] = f"{{type(e).__name__}}: {{e}}"[:120]
        except Exception as e:                                   # a probe bug, not a refusal: say so
            res[name] = f"PROBE-ERROR {{type(e).__name__}}: {{e}}"[:120]

    attempt("read_outside", lambda: open(SECRET).read())

    def openat_root():                                           # relative to an open "/" (the box allows "/" itself)
        d = os.open("/", os.O_RDONLY)
        try:
            f = os.open(os.path.realpath(SECRET).lstrip("/"), os.O_RDONLY, dir_fd=d)
            os.read(f, 5)
            os.close(f)
        finally:
            os.close(d)
    attempt("read_openat_root", openat_root)
    if sys.platform == "darwin":                                 # the data volume under its other name, in /System
        attempt("read_data_volume", lambda: open("/System/Volumes/Data" + os.path.realpath(SECRET)).read())
    attempt("write", lambda: open(os.path.join(OUT_DIR, "w.txt"), "w").write("x"))
    attempt("tcp", lambda: socket.create_connection(("127.0.0.1", PORT), timeout=3).close())

    def other_env():
        if sys.platform == "darwin":                             # KERN_PROCARGS2: argv and environment
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            libc.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
            libc.sysctl.restype = ctypes.c_int
            argmax = ctypes.c_int(0)                             # size the buffer as ps does: kern.argmax
            n = ctypes.c_size_t(ctypes.sizeof(argmax))
            if libc.sysctl((ctypes.c_int * 2)(1, 8), 2, ctypes.byref(argmax), ctypes.byref(n), None, 0) != 0:
                raise OSError(ctypes.get_errno(), "sysctl kern.argmax")
            size = ctypes.c_size_t(argmax.value)
            buf = ctypes.create_string_buffer(size.value)
            if libc.sysctl((ctypes.c_int * 3)(1, 49, OTHER), 3, buf, ctypes.byref(size), None, 0) != 0:
                raise OSError(ctypes.get_errno(), "sysctl kern.procargs2")     # refused by the kernel
            # the marker is in its arguments and its environment: macOS may strip another process's environment
            # even unboxed, so the arguments are what shows the read worked
            if b"ROBOT_USE_PROBE_MARKER" not in buf.raw[:size.value]:
                raise RuntimeError("the sysctl returned, but without the other process's arguments")
        else:
            if b"ROBOT_USE_PROBE_MARKER" not in open(f"/proc/{{OTHER}}/environ", "rb").read():
                raise RuntimeError("read /proc/<pid>/environ, but without the marker")
    attempt("other_env", other_env)

    def fork():
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
    attempt("fork", fork)
    attempt("exec", lambda: __import__("subprocess").run(["/bin/echo", "x"], capture_output=True, check=True))

    def spawn():                                                 # no fork: exec's own check (macOS posix_spawn)
        pid = os.posix_spawn("/bin/echo", ["echo", "x"], {{}})
        os.waitpid(pid, 0)
    attempt("spawn", spawn)
    res["read_repo"] = len(open({ships!r}).read()) > 0
    if sys.platform == "darwin":
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.sandbox_check.restype = ctypes.c_int
        res["sandboxed"] = lib.sandbox_check(os.getpid(), None, 0) != 0
    print("RESULT " + json.dumps(res), flush=True)
""")


@pytest.fixture()
def box():
    b, why = sandbox.choose_box("hardware")
    if b is None:
        b, why = sandbox.choose_box("sim")
    if b is None:
        pytest.skip(f"no kernel sandbox here: {why}")
    return b


ESCAPES = ("read_outside", "read_openat_root", "write", "tcp", "other_env", "fork", "exec", "spawn") + \
    (("read_data_volume",) if sys.platform == "darwin" else ())


def _probe(tmp_path, box, port, other):
    secret = tmp_path / "outside" / "secret.txt"                 # beneath no allowed path
    secret.parent.mkdir(exist_ok=True)
    secret.write_text("token")
    out_dir = tmp_path / f"out-{box}"
    out_dir.mkdir()
    code = PROBE.format(lib=str(REPO_ROOT / "lib"), repo=str(REPO_ROOT), box=box, secret=str(secret),
                        out_dir=str(out_dir), port=port, other=other, ships=str(REPO_ROOT / "stack/runner/sandbox.py"))
    cmd = sandbox.wrap_command([sys.executable, "-c", code], box, str(tmp_path / "unused.py"))
    env = {"PATH": os.environ.get("PATH", "/usr/bin"), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}           # as the runner gives a program
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120)
    wall = time.monotonic() - t0
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"the probe did not run ({box}, exit {p.returncode}):\n{p.stderr[-3000:]}"
    res = json.loads(line[7:])
    print(f"{sys.platform} {box}: wall {wall:.1f} s {res}")
    return res, wall, out_dir


def test_a_program_in_the_box_cannot_escape(tmp_path, box):
    """First without the box (each escape must work: otherwise the probe proves nothing), then in it."""
    srv = socket.socket()
    other = None
    try:
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)  # ROBOT_USE_PROBE_MARKER"],
                                 env={**os.environ, "ROBOT_USE_PROBE_MARKER": "1"})
        time.sleep(0.3)                                          # let it exec, so its environment is the marker's
        control, _, _ = _probe(tmp_path, "none", srv.getsockname()[1], other.pid)
        res, wall, out_dir = _probe(tmp_path, box, srv.getsockname()[1], other.pid)
    finally:
        if other is not None:
            other.kill()
            other.wait()
        srv.close()
    assert all(control[e] == "ALLOWED" for e in ESCAPES), f"the probe cannot do these even unboxed: {control}"
    assert not any(str(v).startswith("PROBE-ERROR") for v in res.values()), res
    assert wall < START_S, f"start-up took {wall:.1f} s under {box}: {res}"
    assert res["read_repo"] is True
    for escape in ("read_outside", "read_openat_root", "write", "other_env") + \
            (("read_data_volume",) if sys.platform == "darwin" else ()):
        assert res[escape] != "ALLOWED", (escape, res)
    if box != sandbox.LANDLOCK_FS:                               # files-only Landlock leaves TCP to the audit hook
        assert res["tcp"] != "ALLOWED", res
    if box == sandbox.SEATBELT:                                  # Landlock leaves fork and exec to the audit hook
        for escape in ("fork", "exec", "spawn"):
            assert res[escape] != "ALLOWED", (escape, res)
        assert res["sandboxed"] is True, res
    assert not list(out_dir.iterdir())
