"""Layer 3 of the program sandbox per platform (stack/runner/sandbox.py, sprint 2026-10-13 D14)."""

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from stack.runner import landlock, sandbox

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("platform_, abi, backend, box", [
    ("linux", 4, "hardware", sandbox.LANDLOCK),
    ("linux", 5, None, sandbox.LANDLOCK),
    ("linux", 3, "sim", sandbox.LANDLOCK_FS),
    ("linux", 1, "sim", sandbox.LANDLOCK_FS),
    ("linux", 3, "hardware", None),            # the real arms always need ABI 4
    ("linux", 3, "mock", None),
    ("linux", 3, None, None),                  # robotd did not say: treated as hardware
    ("linux", 0, "sim", None),                 # no Landlock at all
    ("win32", 0, "sim", None),
])
def test_the_box_is_chosen_from_the_kernel_and_robotds_backend(monkeypatch, platform_, abi, backend, box):
    monkeypatch.setattr(sys, "platform", platform_)
    monkeypatch.setattr(landlock, "abi_version", lambda: abi)
    got, why = sandbox.choose_box(backend)
    assert got == box, why
    if box is None:
        assert why


def test_macos_uses_seatbelt_when_sandbox_exec_exists(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    real_exists = sandbox.os.path.exists
    monkeypatch.setattr(sandbox.os.path, "exists", lambda p: True if p == sandbox.SANDBOX_EXEC else real_exists(p))
    assert sandbox.choose_box("hardware")[0] == sandbox.SEATBELT
    monkeypatch.setattr(sandbox.os.path, "exists", lambda p: False if p == sandbox.SANDBOX_EXEC else real_exists(p))
    assert sandbox.choose_box("sim")[0] is None


def test_the_seatbelt_profile_denies_what_landlock_denies(tmp_path):
    program = tmp_path / "code.py"
    program.write_text("x = 1\n")
    prof = sandbox.seatbelt_profile(sandbox.read_paths(extra=(str(program),)), sys.executable)
    lines = prof.splitlines()
    assert lines[:2] == ["(version 1)", "(allow default)"]
    for rule in ("(deny network*)", "(deny file-write*)", "(deny file-read-data)", "(deny process-fork)",
                 "(deny process-exec*)", "(deny mach-lookup)", "(deny appleevent-send)", "(deny iokit-open)",
                 "(deny signal)", "(allow signal (target self))", "(deny process-info*)",
                 "(allow process-info* (target self))"):
        assert rule in lines
    reads = next(ln for ln in lines if ln.startswith("(allow file-read-data"))
    assert '(literal "/")' in reads and '(subpath "/")' not in reads    # the root directory alone, for dyld
    assert f'(subpath "{REPO_ROOT}")' in reads and f'(literal "{program}")' in reads
    assert str(Path.home()) + '")' not in reads                     # not the home directory itself
    # each deny precedes its exceptions: in SBPL the last matching rule wins
    assert lines.index("(deny file-read-data)") < lines.index(reads)
    cmd = sandbox.wrap_command([sys.executable, "-m", "stack.runner.child", str(program)], sandbox.SEATBELT,
                               str(program))
    assert cmd[:2] == [sandbox.SANDBOX_EXEC, "-p"] and cmd[3:] == [sys.executable, "-m", "stack.runner.child",
                                                                   str(program)]
    assert sandbox.wrap_command(["python"], sandbox.LANDLOCK, str(program)) == ["python"]


def test_without_a_box_the_child_refuses():
    with pytest.raises(RuntimeError, match="no kernel sandbox"):
        sandbox.apply_in_child(None, "x.py")
    with pytest.raises(RuntimeError):
        sandbox.apply_in_child("none", "x.py")


@pytest.mark.parametrize("use_abi", [1, 2, 3])
def test_landlock_below_abi_4_still_makes_files_read_only(tmp_path, use_abi):
    """The simulated rig on an older kernel: the file rules hold; TCP is the audit hook's to refuse."""
    if landlock.abi_version() < 1:
        pytest.skip("this kernel has no Landlock")
    secret = tmp_path / "home" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text("token")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    code = textwrap.dedent(f"""
        import os, socket, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from stack.runner import landlock, sandbox
        landlock.restrict(sandbox.read_paths(), min_abi=1, use_abi={use_abi})
        out = []
        for label, fn in [
            ("write", lambda: open({str(tmp_path / 'w.txt')!r}, "w")),
            ("read_home", lambda: open({str(secret)!r}).read()),
            ("tcp", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=2).close()),
        ]:
            try:
                fn(); out.append(label + ":ALLOWED")
            except OSError as e:
                out.append(label + ":" + type(e).__name__)
        out.append("read_repo:" + str(len(open({str(REPO_ROOT / 'AGENTS.md')!r}).read()) > 0))
        print(" ".join(out))
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    srv.close()
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["write:PermissionError", "read_home:PermissionError", "tcp:ALLOWED",
                                  "read_repo:True"], res.stdout


def test_the_audit_hook_refuses_sockets_where_landlock_cannot():
    """With files-only Landlock, TCP rests on layer 2: the audit hook refuses socket creation."""
    from stack.runner import child
    with pytest.raises(PermissionError, match="socket"):
        child._hook("socket.__new__", ())


def test_the_read_set_does_not_follow_the_callers_folder_or_path(tmp_path, monkeypatch):
    """Review of task 7: the MCP server computes it for macOS and may run from anywhere."""
    monkeypatch.chdir(Path.home())
    monkeypatch.setattr(sys, "path", ["", str(Path.home()), str(tmp_path), *sys.path])
    paths = sandbox.read_paths()
    for p in (str(Path.home()), str(Path.home().resolve()), str(tmp_path), "", "."):
        assert p not in paths, p
    assert str(REPO_ROOT) in paths and sys.prefix in paths
