"""MCP server against a mock robotd, connected in-process (no network for the MCP side)."""

import asyncio
import base64
import json
import socket

import numpy as np
import pytest

from stack.mcp_server.server import RobotdClient, build_server
from stack.robotd.http import serve

from .test_robotd import robotd  # noqa: F401  (fixture)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def mcp_server(robotd):  # noqa: F811
    port = _free_port()
    srv = serve(robotd, "127.0.0.1", port, operator=False)
    yield build_server(RobotdClient(f"http://127.0.0.1:{port}"), public_url="http://jetson.test:8771", surface="debug")
    srv.shutdown()


def _run(server, calls):
    from mcp import Client

    async def go():
        async with Client(server) as c:
            return [await (c.list_tools() if name == "__list__" else c.call_tool(name, args)) for name, args in calls]
    return asyncio.run(go())


READ_TOOLS = {"get_state", "get_limits", "get_camera_frame", "get_camera_frames", "solve_ik",
              "validate_trajectory", "motion_status"}
MOVE_TOOLS = {"move_joints", "move_pose", "set_gripper", "run_trajectory", "stop", "go_rest"}


def test_tool_list_and_annotations(mcp_server):
    (tools,) = _run(mcp_server, [("__list__", None)])
    by = {t.name: t for t in tools.tools}
    assert set(by) == READ_TOOLS | MOVE_TOOLS
    assert all(by[n].annotations.read_only_hint for n in READ_TOOLS)
    assert not any(by[n].annotations.read_only_hint for n in MOVE_TOOLS)


@pytest.mark.parametrize("tool_set, absent", [("joint", {"move_pose", "solve_ik"}), ("cartesian", {"move_joints"})])
def test_tool_sets_for_phase_6b(robotd, tool_set, absent):  # noqa: F811
    (tools,) = _run(build_server(RobotdClient("http://127.0.0.1:9"), tool_set=tool_set, surface="debug"), [("__list__", None)])
    names = {t.name for t in tools.tools}
    assert not (names & absent) and "set_gripper" in names and "run_trajectory" in names
    with pytest.raises(ValueError):
        build_server(RobotdClient("http://127.0.0.1:9"), tool_set="everything", surface="debug")


def test_motion_disabled_is_a_readable_error(mcp_server):
    (r,) = _run(mcp_server, [("move_joints", {"arm": "left", "targets": {"elbow_flex": 50}})])
    assert r.is_error and "motion is disabled" in r.content[0].text


def test_motion_through_mcp(tmp_path):
    from .test_motion import make

    rd = make(tmp_path)
    port = _free_port()
    srv = serve(rd, "127.0.0.1", port, operator=False)
    try:
        server = build_server(RobotdClient(f"http://127.0.0.1:{port}"), surface="debug")
        (r,) = _run(server, [("move_joints", {"arm": "left", "targets": {"elbow_flex": 50}, "speed": 1.0})])
        out = json.loads(r.content[0].text)
        assert out["status"] == "pending_approval"
        rd.approve()                                          # the human presses ARM
        (st,) = _run(server, [("motion_status", {"motion_id": out["motion_id"], "wait_s": 10})])
        assert json.loads(st.content[0].text)["status"] == "done"
        bad, ik, val = _run(server, [
            ("move_joints", {"arm": "left", "targets": {"elbow_flex": 150}}),
            ("solve_ik", {"arm": "left", "xyz": [0.9, 0.2, 0.1]}),
            ("validate_trajectory", {"arm": "left", "waypoints": [{"t": 1.0, "joints": {"elbow_flex": 60}}]})])
        assert bad.is_error and "outside usable range" in bad.content[0].text
        assert json.loads(ik.content[0].text)["ok"] is False
        assert json.loads(val.content[0].text)["ok"] is True
        (stp,) = _run(server, [("stop", {})])
        assert json.loads(stp.content[0].text)["ok"] is True and rd.supervisor.armed is False
    finally:
        srv.shutdown()
        rd.stop()


def test_state_and_limits(mcp_server):
    st, lim = _run(mcp_server, [("get_state", {}), ("get_limits", {})])
    s = json.loads(st.content[0].text)
    assert s["motion_enabled"] is False
    assert "backend" not in s                                       # robotd's backend is for the runner, not the agent
    for arm in ("left", "right"):
        a = s["arms"][arm]
        assert a["fault"] is None and len(a["tip_xyz_m"]) == 3 and a["torque_on"] is True
    lim = json.loads(lim.content[0].text)
    assert lim["arms"]["left"]["rest_pose"]["shoulder_pan"] == 0


def test_camera_frame_is_a_jpeg_image_with_caption(mcp_server):
    import cv2

    (r,) = _run(mcp_server, [("get_camera_frame", {"camera": "wrist_right", "max_side": 200})])
    assert not r.is_error
    img, cap = r.content
    assert img.type == "image" and img.mime_type == "image/jpeg"
    px = cv2.imdecode(np.frombuffer(base64.b64decode(img.data), np.uint8), cv2.IMREAD_COLOR)
    assert max(px.shape[:2]) == 200
    assert cap.type == "text" and cap.text.startswith("wrist_right camera")


def test_multiple_frames_and_url_mode(mcp_server):
    many, url = _run(mcp_server, [
        ("get_camera_frames", {"cameras": ["ego", "wrist_left", "ego"]}),
        ("get_camera_frame", {"camera": "ego", "image_mode": "url", "max_side": 300})])
    assert [c.type for c in many.content] == ["image", "text", "image", "text"]      # duplicate dropped
    assert url.content[0].text == "ego: http://jetson.test:8771/frame/ego?max_side=300&quality=80&overlay=0"


@pytest.mark.parametrize("args, msg", [
    ({"camera": "wrist_left", "overlay": True}, "only available on the ego camera"),
    ({"camera": "nose"}, ""),
    ({"camera": "ego", "max_side": 5000}, ""),
])
def test_bad_arguments_are_errors(mcp_server, args, msg):
    (r,) = _run(mcp_server, [("get_camera_frame", args)])
    assert r.is_error and msg in r.content[0].text


def test_robotd_down_is_a_clear_error():
    server = build_server(RobotdClient(f"http://127.0.0.1:{_free_port()}", timeout_s=1.0), surface="debug")
    (r,) = _run(server, [("get_state", {})])
    assert r.is_error and "not reachable" in r.content[0].text


def test_url_mode_without_public_url_is_an_error(robotd):  # noqa: F811
    port = _free_port()
    srv = serve(robotd, "127.0.0.1", port, operator=False)
    try:
        (r,) = _run(build_server(RobotdClient(f"http://127.0.0.1:{port}"), surface="debug"),
                    [("get_camera_frame", {"camera": "ego", "image_mode": "url"})])
        assert r.is_error and "public URL" in r.content[0].text
    finally:
        srv.shutdown()
