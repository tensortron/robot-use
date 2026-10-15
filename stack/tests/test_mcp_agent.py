"""The agent's MCP surface: it acts only through programs; episodes; geometry; checkpoints with evidence."""

import asyncio
import base64
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from stack.mcp_server.server import RobotdClient, build_server
from stack.robotd.http import serve
from stack.runner.runner import ProgramRunner

from .test_motion import make, wait
from .test_runner import approve, _port



@pytest.fixture
def agent(tmp_path):
    rd = make(tmp_path / "robotd")
    port = _port()
    srv = serve(rd, "127.0.0.1", port, operator=False)
    runner = ProgramRunner(f"http://127.0.0.1:{port}", tmp_path / "ep",
                           preview=lambda code, scene=None: {"twin": "test", "ok": True, "scene": scene})
    server = build_server(RobotdClient(f"http://127.0.0.1:{port}"), runner=runner, episodes_dir=tmp_path / "ep")
    yield rd, runner, server, tmp_path
    for pid in list(runner.programs):
        runner.stop(pid)
    srv.shutdown()
    rd.stop()


def _run(server, calls):
    from mcp import Client

    async def go():
        async with Client(server) as c:
            return [await (c.list_tools() if name == "__list__" else c.call_tool(name, args)) for name, args in calls]
    return asyncio.run(go())


def _json(r):
    return json.loads(r.content[0].text)


def test_the_agent_cannot_command_motion_directly(agent):
    _, _, server, _ = agent
    (tools,) = _run(server, [("__list__", None)])
    names = {t.name for t in tools.tools}
    assert {"run_program", "program_status", "resume_program", "get_program_trace", "get_keyframe", "stop",
            "begin_episode", "end_episode", "pixel_to_table", "table_to_pixel", "get_state", "get_camera_frame"} <= names
    assert not names & {"move_joints", "move_pose", "set_gripper", "run_trajectory", "go_rest", "solve_ik"}


def test_episode_program_checkpoint_and_trace(agent):
    rd, runner, server, tmp = agent
    code = textwrap.dedent("""
        from robot_use import arm, estimate, checkpoint, log
        r = arm("right")
        b = estimate("block", [0.22, -0.17], size=(0.04, 0.04, 0.04), tol=0.03)
        r.open(80)
        checkpoint("hover", b, views=["ego"], need="centred")
        log("block " + str([round(v, 3) for v in b.xy]))
    """)
    ep, sub = _run(server, [("begin_episode", {"instruction": "pick up the red block"}),
                            ("run_program", {"code": code, "title": "t", "scene": [{"name": "block", "xy": [0.22, -0.17],
                                                                                     "size": [0.04, 0.04, 0.04]}]})])
    pid = _json(sub)["program_id"]
    assert Path(_json(sub)["dir"]).parent.name == _json(ep)["episode"], "filed under the episode"
    approve(rd, pid)
    wait(lambda: runner.programs[pid].status == "checkpoint", timeout=20)
    (st,) = _run(server, [("program_status", {"program_id": pid})])
    assert json.loads(st.content[0].text)["status"] == "checkpoint"
    assert any(c.type == "text" and "resume_program" in c.text for c in st.content)
    bad, good = _run(server, [("resume_program", {"program_id": pid, "decision": "refine", "updates": {"block": [0.40, -0.17]}}),
                              ("resume_program", {"program_id": pid, "decision": "refine", "updates": {"block": [0.23, -0.17]}})])
    assert bad.is_error and "tolerance" in bad.content[0].text
    assert not good.is_error
    wait(lambda: runner.programs[pid].status == "done", timeout=20)
    tr, end = _run(server, [("get_program_trace", {"program_id": pid}),
                            ("end_episode", {"outcome": "success", "notes": "test"})])
    trace = _json(tr)["trace"]
    assert any(r.get("type") == "log" and r.get("msg") == "block [0.23, -0.17]" for r in trace)
    assert [r["kind"] for r in trace if "kind" in r] == ["set_gripper"]
    epj = json.loads((Path(_json(sub)["dir"]).parent / "episode.json").read_text())
    assert epj["outcome"] == "success" and epj["programs"] == [pid] and epj["instruction"] == "pick up the red block"


def test_rejected_programs_are_tool_errors_with_the_reason(agent):
    _, _, server, _ = agent
    (r,) = _run(server, [("run_program", {"code": "import os\n"})])
    assert r.is_error and "not allowed" in r.content[0].text


def test_stop_ends_the_running_program(agent):
    rd, runner, server, _ = agent
    (sub,) = _run(server, [("run_program", {"code": "from robot_use import arm\narm('left').move_joints(shoulder_pan=60, speed=0.1)\n"})])
    pid = _json(sub)["program_id"]
    approve(rd, pid)
    wait(lambda: rd._active("left"), timeout=10)
    (r,) = _run(server, [("stop", {})])
    assert _json(r)["stopped_program"] == pid
    wait(lambda: runner.programs[pid].status == "stopped", timeout=15)


def test_geometry_tools_and_marked_frames(agent):
    _, _, server, _ = agent
    a, b = _run(server, [("table_to_pixel", {"xyz": [0.25, -0.15, 0.0]}), ("pixel_to_table", {"u": 700, "v": 400})])
    u, v = _json(a)["px"]
    (back,) = _run(server, [("pixel_to_table", {"u": u, "v": v})])
    assert np.allclose(_json(back)["xyz"], [0.25, -0.15, 0.0], atol=1e-3)
    assert len(_json(b)["xyz"]) == 3
    from stack.mcp_server.server import annotate
    import cv2
    blank = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()
    out = annotate(blank, "ego", grid=True, marks=[{"xyz": [0.25, -0.15, 0.0], "label": "block"}])
    img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    px = (int(u / 2), int(v / 2))
    assert img[px[1] - 3:px[1] + 4, px[0] - 3:px[0] + 4, 2].max() > 150, "the mark is drawn where the point projects"
    assert img[:, 50, :].mean() > 20, "grid line at full-res u=100 is at x=50 in a half-size frame"


def test_check_reach_is_offered_and_tool_calls_count_as_activity(agent):
    _, runner, server, _ = agent
    assert runner._last_client is None
    (tools, res) = _run(server, [("__list__", None),
                                 ("check_reach", {"arm": "right", "points": [{"xyz": [0.2, -0.08, 0.04]}]})])
    assert "check_reach" in {t.name for t in tools.tools}
    assert _json(res)["points"][0]["reachable"] is True
    assert runner._last_client is not None
