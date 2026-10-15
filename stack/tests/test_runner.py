"""The program runner end to end: a mock robotd over HTTP, real sandboxed child processes."""

import json
import socket
import textwrap
import time

import numpy as np
import pytest

from stack.robotd.http import serve
from stack.runner.runner import Budgets, ProgramRunner
from stack.runner.static_check import check

from .test_motion import J, make, wait

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


@pytest.fixture
def rig(tmp_path):
    rd = make(tmp_path / "robotd")
    port = _port()
    srv = serve(rd, "127.0.0.1", port, operator=False)
    runners = []

    def runner(**budgets):
        r = ProgramRunner(f"http://127.0.0.1:{port}", tmp_path / "episodes", Budgets(**budgets), holder="test")
        runners.append(r)
        return r
    yield rd, runner
    for r in runners:
        for pid, p in r.programs.items():
            if p.status not in ("done", "failed", "aborted", "stopped", "denied", "rejected", "expired"):
                r.stop(pid)
    srv.shutdown()
    rd.stop()


def approve(rd, pid, step_mode=None, timeout=10):
    wait(lambda: pid in rd.supervisor.programs and rd.supervisor.programs[pid].status == "pending", timeout)
    assert rd.decide_program(pid, True, step_mode)["ok"]


def run(rd, runner, code, approve_it=True, wait_s=30, **kw):
    out = runner.submit(textwrap.dedent(code), title="test", **kw)
    if not out["ok"]:
        return out
    if approve_it:
        approve(rd, out["program_id"])
    t0 = time.time()
    while time.time() - t0 < wait_s:
        st = runner.status(out["program_id"], wait_s=5)
        if st["status"] in ("done", "failed", "aborted", "stopped", "denied", "rejected", "expired", "checkpoint"):
            return st
    raise AssertionError(f"program still {st['status']}")


def trace(st):
    from pathlib import Path
    return [json.loads(l) for l in (Path(st["dir"]) / "trace.jsonl").read_text().splitlines()]


GOOD = """
from robot_use.runtime import step, log
step("move_joints", arm="left", targets={"elbow_flex": 50}, speed=1.0)
r = step("move_joints", arm="left", targets={"shoulder_pan": 10}, speed=1.0)
log("second step " + r["status"])
"""


def test_a_program_runs_after_approval_under_its_lease(rig):
    rd, runner = rig
    r = runner()
    st = run(rd, r, GOOD)
    assert st["status"] == "done", st
    assert st["steps"] == 2
    q = rd.arms["left"].reading.q
    assert abs(q[J["elbow_flex"]] - 50) < 3 and abs(q[J["shoulder_pan"]] - 10) < 3
    tr = trace(st)
    assert [x["n"] for x in tr if x["type"] == "step"] == [1, 2]
    assert any(x["type"] == "log" and x["msg"] == "second step done" for x in tr)
    wait(lambda: rd.lease is None)                                # released right after the outcome
    assert rd.supervisor.programs[st["program_id"]].status == "ended"


@pytest.mark.parametrize("code, msg", [
    ("import os\n", "import 'os' is not allowed"),
    ("open('/tmp/x', 'w')\n", "'open' is not available"),
    ("x = (1).__class__\n", "dunder attribute"),
    ("from robot_use import runtime\nruntime._rpc('step')\n", "is private"),
    ("import lerobot\n", "import 'lerobot' is not allowed"),
    ("def f(:\n", "syntax error"),
])
def test_static_check_rejects_before_anyone_sees_it(rig, code, msg):
    rd, runner = rig
    out = runner().submit(code)
    assert not out["ok"] and out["status"] == "rejected" and msg in out["reason"], out
    assert not rd.supervisor.programs, "never reached the approval page"


def test_the_sandbox_blocks_what_the_static_check_cannot_see(rig, tmp_path):
    """numpy is allowed, but numpy.save opening a file for writing is refused by the audit hook."""
    rd, runner = rig
    target = tmp_path / "leak.npy"
    st = run(rd, runner(), f"import numpy\nnumpy.save({str(target)!r}, numpy.zeros(3))\n")
    assert st["status"] == "failed" and "blocked by the runner sandbox" in st["reason"], st
    assert not target.exists()


def test_a_failed_step_raises_in_the_program(rig):
    rd, runner = rig
    st = run(rd, runner(), """
        from robot_use.runtime import step
        step("move_joints", arm="left", targets={"elbow_flex": 150})
    """)
    assert st["status"] == "failed" and "StepFailed" in st["reason"] and "outside usable range" in st["reason"], st
    assert st["outcome"]["error"]["result"]["status"] == "rejected"


def test_repeated_failures_stop_the_program_even_if_it_catches_them(rig):
    rd, runner = rig
    st = run(rd, runner(), """
        from robot_use.runtime import step, StepFailed
        for i in range(10):
            try:
                step("move_joints", arm="left", targets={"elbow_flex": 150})
            except StepFailed:
                pass
    """)
    assert st["status"] == "aborted" and "in a row" in st["reason"] and st["steps"] == 2, st


def test_step_and_wall_clock_budgets(rig):
    rd, runner = rig
    st = run(rd, runner(max_steps=3), """
        from robot_use.runtime import step
        for i in range(10):
            step("move_joints", arm="right", targets={"shoulder_pan": -5 * (i % 2)}, speed=1.0)
    """)
    assert st["status"] == "aborted" and "step budget" in st["reason"] and st["steps"] == 3, st
    st = run(rd, runner(wall_clock_s=2.0), "while True:\n    pass\n")
    assert st["status"] == "aborted" and "wall-clock" in st["reason"], st


CP = """
from robot_use.runtime import step, checkpoint, log
step("move_joints", arm="right", targets={"elbow_flex": 50}, speed=1.0)
r = checkpoint("hover", estimates={"block": {"value": [0.20, -0.10], "tol": 0.03}}, views=["ego", "wrist_right"],
               need="block centred between the jaws")
log("block now " + str(r["estimates"]["block"]) + " after " + r["decision"])
"""


def test_checkpoint_suspends_and_resumes_with_a_refined_estimate(rig):
    rd, runner = rig
    r = runner()
    st = run(rd, r, CP)
    assert st["status"] == "checkpoint" and st["checkpoint"]["name"] == "hover", st
    assert st["checkpoint"]["need"] == "block centred between the jaws"
    # while suspended the program still holds the robot: nobody else can command it
    assert rd.move_joints("left", {"elbow_flex": 50})["status"] == "rejected"
    pid = st["program_id"]
    bad = r.resume(pid, "refine", {"block": [0.30, -0.10]})
    assert not bad["ok"] and "exceeds its tolerance" in bad["reason"]
    assert r.status(pid)["status"] == "checkpoint", "a refused refinement leaves it waiting"
    assert r.resume(pid, "refine", {"block": [0.215, -0.11]})["ok"]
    st = r.status(pid, wait_s=10)
    assert st["status"] == "done", st
    assert any(x.get("msg") == "block now [0.215, -0.11] after refine" for x in trace(st) if x["type"] == "log")


def test_checkpoint_abort_and_stop(rig):
    rd, runner = rig
    r = runner()
    st = run(rd, r, CP)
    assert r.resume(st["program_id"], "abort")["ok"]
    assert r.status(st["program_id"], wait_s=10)["status"] == "aborted"
    wait(lambda: r.programs[st["program_id"]].cleaned and rd.lease is None)
    out = r.submit(textwrap.dedent("""
        from robot_use.runtime import step
        step("move_joints", arm="left", targets={"shoulder_pan": 60}, speed=0.1)
    """))
    approve(rd, out["program_id"])
    wait(lambda: r.programs[out["program_id"]].steps == 1 and rd._active("left"))
    assert r.stop(out["program_id"])["ok"]
    st = r.status(out["program_id"], wait_s=10)
    assert st["status"] == "stopped", st
    wait(lambda: not rd._active("left"))
    assert rd.arms["left"].reading.q[J["shoulder_pan"]] < 50, "stopped well short of the target"


def test_denied_programs_never_move(rig):
    rd, runner = rig
    r = runner()
    out = r.submit(GOOD)
    wait(lambda: out["program_id"] in rd.supervisor.programs)
    rd.decide_program(out["program_id"], False)
    st = r.status(out["program_id"], wait_s=15)
    assert st["status"] == "denied" and st["steps"] == 0
    assert rd.arm_ios["left"].writes == 0


def test_step_mode_waits_for_arm_on_every_step(rig):
    rd, runner = rig
    r = runner()
    out = r.submit(GOOD, step_mode=True)
    approve(rd, out["program_id"])
    for n in (1, 2):
        wait(lambda: any(m.status == "pending_approval" for m in rd.motions.values()), timeout=10)
        assert r.programs[out["program_id"]].steps == n
        time.sleep(0.3)
        assert any(m.status == "pending_approval" for m in rd.motions.values()), "still waiting for the human"
        rd.approve()
    assert r.status(out["program_id"], wait_s=15)["status"] == "done"


def test_one_program_at_a_time(rig):
    rd, runner = rig
    r = runner()
    a = r.submit(GOOD)
    b = r.submit(GOOD)
    assert a["ok"] and not b["ok"] and "one at a time" in b["reason"]
    r.stop(a["program_id"])



@pytest.mark.parametrize("code", [
    'from robot_use import runtime\nm = runtime.os.sys.modules["cv2"]\n',     # review 2026-10-03: OpenCV wrote files
    'import dataclasses\ndataclasses.sys.modules\n',
    'from robot_use import runtime\nm = runtime\nm.os\n',                        # aliasing defeats name tracking
    'from typing import sys\n',
    'import numpy.f2py\n',
    'import numpy as np\nnp.f2py\n',
    'import random\nrandom._os\n',
    'import json\njson.codecs\n',
    'import typing\ndef f(x: "getattr"): pass\n',                          # string annotations get evaluated
    'import typing\ndef f(x): pass\ntyping.get_type_hints(f)\n',
    'import numpy as np\nnp.load(None, allow_pickle=True)\n',
    'from functools import singledispatch\n',
])
def test_static_check_refuses_routes_to_modules_outside_the_allowed_set(code):
    assert check(code), code


def test_the_librarys_former_name_is_refused():
    """D15: the library is robot_use; the old name is gone (no alias), so a program with it fails clearly."""
    assert any("so2x_lib" in m for m in check("from so2x_lib import arm\n"))


def test_static_check_still_admits_an_ordinary_program():
    code = ('from robot_use import arm, geo, estimate, contact\nimport numpy as np\nr = arm("right")\nr.open()\n'
            'x = np.linalg.norm([1, 2])\nprint("{}".format(x))\nb = estimate("b", xy=[0.2, 0.1])\nb.above(0.05)\n')
    assert check(code) == []


def test_landlock_alone_makes_the_process_read_only_and_offline(tmp_path):
    """Layer 3 of the sandbox (review 2026-10-03): even code that escapes the Python layers cannot write
    files, read the home directory or open TCP connections."""
    import subprocess
    import sys
    from stack.runner import landlock
    if landlock.abi_version() < 4:
        pytest.skip("this kernel has no Landlock ABI 4")
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
        from stack.runner import landlock
        landlock.restrict([sys.prefix, sys.base_prefix, "/usr", "/lib", {str(REPO_ROOT)!r}])
        out = []
        for label, fn in [
            ("write", lambda: open({str(tmp_path / 'w.txt')!r}, "w")),
            ("read_home", lambda: open({str(secret)!r}).read()),
            ("tcp", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=2)),
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
    assert res.stdout.split() == ["write:PermissionError", "read_home:PermissionError", "tcp:PermissionError",
                                  "read_repo:True"], res.stdout


def test_a_checkpoint_is_abandoned_when_the_agent_goes_silent(rig):
    """Review 2026-10-03: closing the agent left its program at a checkpoint holding the lease for 15 min."""
    rd, runner = rig
    r = runner(checkpoint_silence_s=0.5)
    r.note_client_activity()
    st = run(rd, r, "from robot_use import checkpoint\ncheckpoint('look')\n")
    assert st["status"] == "checkpoint", st
    t0 = time.time()
    while st["status"] == "checkpoint" and time.time() - t0 < 15:
        time.sleep(0.2)
        st = r.status(st["program_id"])
    assert st["status"] in ("aborted", "stopped") and "made no call" in json.dumps(st), st


def test_programs_can_ask_reach_and_look_with_the_idle_arm(rig):
    rd, runner = rig
    code = textwrap.dedent("""
        from robot_use import arm, log, NoViewpoint
        r = arm("left").reach([[0.2, -0.08, 0.04], [0.9, 0.0, 0.0]])
        log("reach " + str([p["reachable"] for p in r["points"]]))
        v = arm("left").look_at([0.20, -0.08, 0.02], distance=0.15, speed=0.8)
        log("looking from " + v["camera"] + " at elevation " + str(v["elevation"]))
        try:
            arm("left").look_at([0.9, -0.5, 0.0], distance=0.1)
        except NoViewpoint:
            log("no viewpoint, as expected")
    """)
    st = run(rd, runner(), code)
    assert st["status"] == "done", st
    logs = [x["msg"] for x in trace(st) if x["type"] == "log"]
    assert "reach [True, False]" in logs and any(m.startswith("looking from wrist_left") for m in logs), logs
    assert "no viewpoint, as expected" in logs


def test_read_only_queries_have_a_budget(rig):
    rd, runner = rig
    code = "from robot_use import arm\nwhile True:\n    arm('left').reach([[0.2, -0.08, 0.04]])\n"
    st = run(rd, runner(max_queries=5), code)
    assert st["status"] == "aborted" and "query budget exceeded" in json.dumps(st), st
