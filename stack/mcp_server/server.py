"""MCP server for the bimanual SO-101 rig: a thin, stateless adapter over robotd's localhost API.

State, limits, camera frames and motion (docs/ARCHITECTURE.md). It never opens a device and
holds no safety logic: every motion is planned, checked, approved and executed by robotd, which
rejects anything unsafe with the reason. Restarting this server can never move an arm.

Built on the official ``mcp`` Python SDK 2.x (``MCPServer``, formerly FastMCP), streamable HTTP.
"""


import functools
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Literal

import sys
from pathlib import Path

import anyio
import numpy as np
from pydantic import Field

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import TextContent, ToolAnnotations

CameraName = Literal["ego", "wrist_left", "wrist_right"]
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
MOVES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)
ArmName = Literal["left", "right"]
ArmOrBoth = Literal["left", "right", "both"]
Speed = Annotated[float, Field(gt=0.0, le=1.0, description="fraction of the speed cap, 0..1")]
Xyz = Annotated[list[float], Field(min_length=3, max_length=3, description="[x, y, z] metres, rig frame")]
TOOL_SETS = ("all", "joint", "cartesian")
SURFACES = ("agent", "debug")
REPO = Path(__file__).resolve().parents[2]
if str(REPO / "lib") not in sys.path:
    sys.path.insert(0, str(REPO / "lib"))

INSTRUCTIONS = """\
You are connected to a physical bimanual robot: two SO-101 arms ("left" and "right", seen from the
robot) on a table, with three cameras. You can read state and limits, look through the cameras and
(when the robot's motion is enabled) move the arms. get_state tells you whether motion is enabled.

Coordinates: the rig frame has its origin on the work surface midway between the two arm bases;
+x points forward (away from the arms, toward the workspace), +y to the robot's left, +z up, in
metres. The arms stand on an 18 mm board. Joint angles are in degrees (0 = middle of the calibrated
range); the gripper is 0..100 (0 = closed). "pitch" is the gripper's approach angle in the arm's
vertical plane (0 = pointing forward horizontally, -90 = straight down).

Cameras: "ego" looks down at the workspace from 0.7 m above, between the arms; "wrist_left" and
"wrist_right" ride on each gripper. Request small frames (max_side 512 is usually enough) and only
the cameras you need: each image costs context. overlay=true on the ego camera draws the robot's
own kinematic model of both arms over the image; if the drawn arms do not lie on the real ones,
the model or the calibration is wrong and you should say so.

Call get_limits once before reasoning about reachable positions. Arm state includes "fault"; a
non-null fault means that arm cannot be trusted or used.

Moving: every motion is checked in full by the robot's safety layer before anything moves (joint
ranges, speed and acceleration caps, the table, the workspace box, the distance between the arms)
and is rejected with the reason if any check fails; nothing is ever silently clipped, so read the
reason and correct the command. Far targets are fine: the robot plans a smooth speed-capped path.
Motion tools return at once with a motion_id; call motion_status(motion_id, wait_s) to wait for it.
Statuses: pending_approval (a human must press ARM on the operator page; tell the user), running,
done, blocked (the arm hit something: it stopped, backed off and holds; look at the cameras before
trying again, and expect the next motion on that arm to need approval), stopped, rejected, faulted.
set_gripper with a low max_effort closes until it meets resistance: report.grasped means it is holding
something. Prefer validate_trajectory before run_trajectory for anything multi-step. Use stop() if
anything looks wrong.
"""


class RobotdUnavailable(ToolError):
    pass


class RobotdClient:
    """Minimal client for robotd's localhost JSON/JPEG API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8770", timeout_s: float = 5.0):
        self.base = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _get_sync(self, path: str) -> tuple[bytes, dict]:
        try:
            with urllib.request.urlopen(self.base + path, timeout=self.timeout_s) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise ToolError(f"robotd: {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            raise RobotdUnavailable(f"robotd is not reachable at {self.base} ({e}); is it running?") from None

    async def get(self, path: str) -> tuple[bytes, dict]:
        return await anyio.to_thread.run_sync(self._get_sync, path)

    async def _get_async_long(self, path: str, timeout_s: float) -> tuple[bytes, dict]:
        """A GET that may legitimately take ``timeout_s`` (motion_status waiting for a motion to end)."""
        return await anyio.to_thread.run_sync(RobotdClient(self.base, timeout_s)._get_sync, path)

    async def json(self, path: str):
        body, _ = await self.get(path)
        return json.loads(body)

    def _post_sync(self, path: str, body: dict, timeout_s: float):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise ToolError(f"robotd: {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            raise RobotdUnavailable(f"robotd is not reachable at {self.base} ({e}); is it running?") from None

    async def post(self, path: str, body: dict, timeout_s: float | None = None):
        return await anyio.to_thread.run_sync(self._post_sync, path, body, timeout_s or self.timeout_s)


AGENT_INSTRUCTIONS = """\
You are connected to a physical bimanual robot: two SO-101 arms ("left" and "right", seen from the
robot) at the back of a table, with three cameras. You act on it ONLY by writing Python programs
against the robot_use library and running them with run_program; every motion is checked by the
robot's safety layer and rejected with the reason (never silently clipped).

Coordinates: rig frame, metres. Origin on the table surface midway between the two shoulders; +x
forward over the table, +y to the robot's left, +z up. The left shoulder is at y = +0.186, the right
at y = -0.186. Joint angles in degrees; gripper 0..100 (0 closed).

Looking: get_camera_frame("ego") shows the whole workspace from 0.7 m above; a pixel grid labelled
in FULL-RESOLUTION coordinates (1280 x 720) is drawn by default, so you can read a pixel off it and
pass it straight to pixel_to_table(u, v, z) - z is the height of the point you mean (0 for the table,
an object's height for its top). Use marks=[{"xyz": [...], "label": "..."}] to see where a planned
point would appear before you act on it. The wrist cameras show the jaws at the bottom of the image.

Working: begin_episode(instruction) first. Write a program (the API is in the robot_use README in
your instructions), give run_program the scene you believe in (objects with xy and size): it is
previewed in a simulated twin first, then a human approves it on the operator page (tell the user
it is waiting). Poll program_status(id, wait_s). At a checkpoint you see the evidence images: resume
with continue, refine (move estimates within their tolerance) or abort. When something fails, read
get_program_trace and fix the program; a failed preview comes back with the reason. Finish with
end_episode(outcome, notes). stop() halts everything at once.
"""


def _frame_path(camera: str, max_side: int, quality: int, overlay: bool) -> str:
    q = urllib.parse.urlencode({"max_side": max_side, "quality": quality, "overlay": int(overlay)})
    return f"/frame/{camera}?{q}"


def annotate(jpeg: bytes, camera: str, grid: bool, marks=None, grid_step: int = 100) -> bytes:
    """Draw a pixel grid labelled in FULL-RESOLUTION coordinates, and marks (rig-frame points via the
    calibrated ego camera, or full-resolution pixels), on a (possibly downscaled) frame."""
    import cv2
    from robot_use import geometry as geo

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    full_w, full_h = geo.image_size("ego") if camera == "ego" else (1024, 576)
    k = w / float(full_w)
    if grid:
        for u in range(0, full_w + 1, grid_step):
            x = int(round(u * k))
            cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(u), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(u), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        for v in range(0, full_h + 1, grid_step):
            y = int(round(v * k))
            cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(v), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(v), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i, m in enumerate(marks or []):
        if "xyz" in m:
            if camera != "ego":
                raise ToolError("marks with xyz can only be drawn on the ego camera (the calibrated one)")
            u, v = geo.table_to_pixel(m["xyz"])
        elif "px" in m:
            u, v = (float(t) for t in m["px"])
        else:
            raise ToolError("a mark needs xyz (rig frame) or px (full-resolution pixel)")
        c = (int(round(u * k)), int(round(v * k)))
        cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(img, c, 10, (0, 0, 255), 1, cv2.LINE_AA)
        label = str(m.get("label", i + 1))
        cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else jpeg


def _motion_result(out: dict) -> dict:
    """Motion replies straight from robotd; a rejection is an error the model must read and act on."""
    if out.get("status") == "rejected":
        raise ToolError(out.get("reason") or "rejected")
    return out


def build_server(robotd: RobotdClient, public_url: str | None = None, tool_set: str = "all", surface: str = "agent",
                 runner=None, episodes_dir: Path | str | None = None) -> MCPServer:
    """``surface``: "agent" (default: observe, geometry, programs, episodes - the agent acts only through
    programs) or "debug" (observe + the direct motion tools, for people debugging the stack).
    ``runner``: the ProgramRunner (agent surface); created against robotd if not given.
    ``public_url``: this server's base URL as clients reach it, for frame URLs (image_mode='url').
    ``tool_set`` (debug surface): "all", or "joint" / "cartesian" to offer one way of commanding the arms."""
    if tool_set not in TOOL_SETS:
        raise ValueError(f"tool_set must be one of {TOOL_SETS}")
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}")
    debug = surface == "debug"
    joint_tools, cart_tools = debug and tool_set in ("all", "joint"), debug and tool_set in ("all", "cartesian")
    mcp = MCPServer(name="so101-bimanual", title="SO-101 bimanual rig",
                    instructions=INSTRUCTIONS if debug else AGENT_INSTRUCTIONS, version="0.3")
    # Every tool call is a sign of life from the client: the runner aborts a program left waiting at a
    # checkpoint after the client goes quiet (review 2026-10-03: closing the agent left one holding the lease).
    activity = {"fn": None}
    _register = mcp.tool

    def _tool(*a, **kw):
        reg = _register(*a, **kw)

        def deco(fn):
            @functools.wraps(fn)
            async def wrapped(*args, **kwargs):
                if activity["fn"] is not None:
                    activity["fn"]()
                return await fn(*args, **kwargs)
            return reg(wrapped)
        return deco
    mcp.tool = _tool
    mcp._robot_use_activity = activity

    @mcp.tool(annotations=READ_ONLY)
    async def get_state() -> dict:
        """Current state of both arms and the cameras.

        Per arm: joint angles (deg), gripper (0..100), gripper-tip position (m, rig frame), approach
        pitch and wrist roll (deg), per-joint load (% of max torque) and current (mA), torque state,
        fault. Also whether motion is enabled and the supervisor state.
        """
        s = await robotd.json("/state")
        arms = {}
        for name, a in s.get("arms", {}).items():
            arms[name] = {k: a.get(k) for k in ("fault", "joints", "tip_xyz_m", "pitch_deg", "roll_deg",
                                                  "load_pct", "current_ma", "torque", "age_s", "command")}
            arms[name]["torque_on"] = None if a.get("torque") is None else any(a["torque"])
        return {"motion_enabled": s.get("motion_enabled"), "supervisor": s.get("supervisor"), "arms": arms,
                "cameras": s.get("cameras"), "loop_hz": s.get("loop", {}).get("measured_hz")}

    @mcp.tool(annotations=READ_ONLY)
    async def get_limits() -> dict:
        """Units, the rig frame, per-joint usable ranges (calibrated range minus the safety margin),
        speed and acceleration caps, the workspace box, floor clearance, the minimum distance kept
        between the arms, and each arm's rest pose. Anything outside these will be rejected."""
        return await robotd.json("/limits")

    async def _one_frame(camera, max_side, quality, overlay, image_mode, grid=False, marks=None):
        if image_mode == "url":
            if not public_url:
                raise ToolError("image_mode='url' is not available: this server was started without a public URL")
            url = public_url.rstrip("/") + _frame_path(camera, max_side, quality, overlay)
            return [TextContent(type="text", text=f"{camera}: {url}")]
        body, headers = await robotd.get(_frame_path(camera, max_side, quality, overlay))
        age = headers.get("X-Frame-Age-S") or headers.get("x-frame-age-s")
        cap = f"{camera} camera, captured {float(age):.2f} s ago" if age else f"{camera} camera"
        if overlay:
            cap += "; the robot's kinematic model of both arms is drawn on top"
        if grid or marks:
            body = await anyio.to_thread.run_sync(annotate, body, camera, grid, marks)
            if grid:
                cap += "; grid lines every 100 px, labelled in full-resolution pixels (use them as-is)"
            if marks:
                cap += f"; {len(marks)} mark(s) drawn in red"
        return [Image(data=body, format="jpeg"), TextContent(type="text", text=cap)]

    @mcp.tool(annotations=READ_ONLY)
    async def get_camera_frame(
        camera: CameraName,
        max_side: Annotated[int, Field(ge=64, le=1280, description="longest side in pixels")] = 512,
        quality: Annotated[int, Field(ge=10, le=95, description="JPEG quality")] = 80,
        overlay: Annotated[bool, Field(description="draw the robot's own arm model (ego camera only)")] = False,
        image_mode: Annotated[Literal["image", "url"],
                              Field(description="'image' returns the picture; 'url' returns a link, for clients "
                                                "that cannot take images in tool results")] = "image",
        grid: Annotated[bool | None, Field(description="pixel grid in full-resolution coordinates "
                                                       "(default: on for ego on the agent surface)")] = None,
        marks: Annotated[list[dict] | None, Field(description='points to draw: [{"xyz": [x, y, z]} (ego only) or '
                                                              '{"px": [u, v]}, optional "label"}]')] = None,
    ) -> list[Image | TextContent]:
        """The newest frame from one camera, downscaled, as a JPEG image."""
        if overlay and camera != "ego":
            raise ToolError("overlay is only available on the ego camera (it is the calibrated one)")
        if grid is None:
            grid = (not debug) and camera == "ego"
        return await _one_frame(camera, max_side, quality, overlay, image_mode, grid, marks)

    @mcp.tool(annotations=READ_ONLY)
    async def get_camera_frames(
        cameras: Annotated[list[CameraName], Field(min_length=1, max_length=3)] = ["ego", "wrist_left", "wrist_right"],
        max_side: Annotated[int, Field(ge=64, le=1280)] = 384,
        quality: Annotated[int, Field(ge=10, le=95)] = 75,
        image_mode: Literal["image", "url"] = "image",
    ) -> list[Image | TextContent]:
        """Several cameras at once (captured within a few ms of each other), each followed by its caption."""
        out = []
        for cam in dict.fromkeys(cameras):          # de-duplicate, keep order
            out += await _one_frame(cam, max_side, quality, False, image_mode)
        return out

    if debug:
        # ------------------------------------------------------------------ motion
        if joint_tools:
            @mcp.tool(annotations=MOVES)
            async def move_joints(arm: ArmName, targets: Annotated[dict[str, float], Field(
                    description="joint name -> target degrees (gripper 0..100); joints not named keep their position. "
                                "Names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper")],
                    speed: Speed = 0.5) -> dict:
                """Move one arm's joints to target angles on a smooth, speed-capped path."""
                return _motion_result(await robotd.post("/motion/move_joints", {"arm": arm, "targets": targets, "speed": speed}))

        if cart_tools:
            @mcp.tool(annotations=MOVES)
            async def move_pose(arm: ArmName, xyz: Xyz,
                                pitch: Annotated[float | None, Field(description="approach pitch, deg (-90 = pointing down); "
                                                                               "omit to let the robot choose")] = None,
                                roll: Annotated[float | None, Field(description="wrist roll joint, deg; omit to keep")] = None,
                                linear: Annotated[bool, Field(description="move the tip along a straight line "
                                                                          "(slower; may be rejected where a line leaves reach)")] = False,
                                speed: Speed = 0.5) -> dict:
                """Move one arm's gripper tip to a position in the rig frame (inverse kinematics)."""
                return _motion_result(await robotd.post("/motion/move_pose", {
                    "arm": arm, "xyz": xyz, "pitch": pitch, "roll": roll, "linear": linear, "speed": speed}))

            @mcp.tool(annotations=READ_ONLY)
            async def solve_ik(arm: ArmName, xyz: Xyz, pitch: float | None = None, roll: float | None = None) -> dict:
                """Joint angles that would put the gripper tip at xyz (no motion), or why it is unreachable."""
                return await robotd.post("/motion/solve_ik", {"arm": arm, "xyz": xyz, "pitch": pitch, "roll": roll})

        @mcp.tool(annotations=MOVES)
        async def set_gripper(arm: ArmName, value: Annotated[float, Field(ge=0, le=100, description="0 closed .. 100 open")],
                              speed: Speed = 0.5,
                              max_effort: Annotated[float, Field(gt=0, le=1, description="grip force cap, 0..1; "
                                                                 "closing stops when it meets this resistance")] = 0.5) -> dict:
            """Open or close one gripper."""
            return _motion_result(await robotd.post("/motion/set_gripper", {
                "arm": arm, "value": value, "speed": speed, "max_effort": max_effort}))

        wp_desc = ("waypoints in time order: {t: seconds from start, joints: {name: deg}"
                   + (" | pose: {xyz, pitch?, roll?, linear?}" if cart_tools else "")
                   + ", gripper?: 0..100}. For arm='both' each waypoint is {t, left: {...}, right: {...}}")

        @mcp.tool(annotations=MOVES)
        async def run_trajectory(arm: ArmOrBoth, waypoints: Annotated[list[dict], Field(min_length=1, description=wp_desc)],
                                 time_scale: Annotated[Literal["reject", "auto"], Field(
                                     description="'auto' slows the timing down until the caps hold")] = "reject") -> dict:
            """Follow timed waypoints (one arm, or both arms on one clock)."""
            return _motion_result(await robotd.post("/motion/trajectory", {
                "arm": arm, "waypoints": waypoints, "time_scale": time_scale}, timeout_s=30))

        @mcp.tool(annotations=READ_ONLY)
        async def validate_trajectory(arm: ArmOrBoth, waypoints: Annotated[list[dict], Field(min_length=1, description=wp_desc)],
                                      time_scale: Literal["reject", "auto"] = "reject") -> dict:
            """Check a trajectory without moving: ok or the violations, duration, peak speeds, lowest tip height
            and closest approach between the arms."""
            return await robotd.post("/motion/validate_trajectory", {
                "arm": arm, "waypoints": waypoints, "time_scale": time_scale}, timeout_s=30)

        @mcp.tool(annotations=READ_ONLY)
        async def motion_status(motion_id: str,
                                wait_s: Annotated[float, Field(ge=0, le=30, description="wait up to this long for it to end")] = 0) -> dict:
            """Status of a motion; with wait_s, returns as soon as it ends (done, blocked, stopped, ...)."""
            q = urllib.parse.urlencode({"wait_s": wait_s})
            body, _ = await robotd._get_async_long(f"/motion/{urllib.parse.quote(motion_id)}?{q}", wait_s + 5)
            return json.loads(body)

        @mcp.tool(annotations=MOVES)
        async def stop(arm: ArmOrBoth = "both") -> dict:
            """Stop now: decelerate to a hold. Also disarms the session (the next motion needs approval)."""
            return await robotd.post("/motion/stop", {"arm": arm})

        @mcp.tool(annotations=MOVES)
        async def go_rest(arm: ArmOrBoth = "both") -> dict:
            """Move slowly to the folded rest pose."""
            return _motion_result(await robotd.post("/motion/go_rest", {"arm": arm}))


    else:
        _agent_tools(mcp, robotd, runner, episodes_dir)

    @mcp.custom_route("/frame/{camera}", methods=["GET"], include_in_schema=False)
    async def frame_route(request):
        """image_mode='url' target: proxies one JPEG from robotd (robotd itself is localhost only)."""
        from starlette.responses import JSONResponse, Response

        camera = request.path_params["camera"]
        if camera not in ("ego", "wrist_left", "wrist_right"):
            return JSONResponse({"error": f"unknown camera {camera}"}, status_code=404)
        try:
            max_side = int(request.query_params.get("max_side", 512))
            quality = int(request.query_params.get("quality", 80))
            overlay = request.query_params.get("overlay", "0") in ("1", "true")
            body, _ = await robotd.get(_frame_path(camera, max_side, quality, overlay))
        except ValueError:
            return JSONResponse({"error": "max_side and quality must be integers"}, status_code=400)
        except ToolError as e:
            return JSONResponse({"error": str(e)}, status_code=502)
        return Response(body, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return mcp



def _default_runner(robotd: RobotdClient, episodes_dir):
    from stack.runner.runner import ProgramRunner, default_preview
    try:
        from stack.twin_mujoco.scene import assets_dir
        from functools import partial
        from stack.twin_mujoco.preview import twin_preview
        assets_dir()
        preview = partial(twin_preview, robotd_url=robotd.base)   # robotd's own joint ranges
    except Exception:                                   # no MuJoCo assets: the human reviews the code alone
        preview = default_preview
    return ProgramRunner(robotd.base, episodes_dir, preview=preview, holder="mcp")


def _small_jpeg(path: Path, max_side: int = 512) -> bytes | None:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _agent_tools(mcp, robotd: RobotdClient, runner, episodes_dir):
    """The agent surface: geometry, programs, episodes. The agent never commands a motion directly."""
    import datetime as _dt

    from robot_use import geometry as geo

    runner = runner or _default_runner(robotd, episodes_dir)
    if hasattr(runner, "note_client_activity"):
        mcp._robot_use_activity["fn"] = runner.note_client_activity
    from stack import paths
    episodes = Path(episodes_dir).expanduser() if episodes_dir else paths.episodes()
    ep = {"id": None, "dir": None, "programs": []}

    def _ep_write(extra=None):
        if ep["dir"] is not None:
            f = ep["dir"] / "episode.json"
            d = json.loads(f.read_text()) if f.exists() else {}
            d.update({"programs": ep["programs"]} | (extra or {}))
            f.write_text(json.dumps(d, indent=1))

    async def call(fn, *a, **kw):
        return await anyio.to_thread.run_sync(lambda: fn(*a, **kw))

    # ---------------------------------------------------------------- geometry
    @mcp.tool(annotations=READ_ONLY)
    async def pixel_to_table(u: float, v: float, z: Annotated[float, Field(description="height of the point you mean: "
                             "0 for the table, an object's height for its top")] = 0.0) -> dict:
        """Where ego pixel (u, v) (full-resolution, 1280 x 720) meets the horizontal plane at height z,
        in the rig frame (metres)."""
        try:
            return {"xyz": [round(v_, 4) for v_ in geo.pixel_to_table(u, v, z)]}
        except ValueError as e:
            raise ToolError(str(e))

    @mcp.tool(annotations=READ_ONLY)
    async def check_reach(arm: ArmName, points: Annotated[list[dict], Field(
            min_length=1, max_length=20,
            description='[{"xyz": [x, y, z], "pitch": deg (optional), "roll": deg (optional)}]')]) -> dict:
        """Can this arm's gripper tip be at each point, and at which pitches? Instant (no motion, no preview).

        Per point: `reachable` at the pitch you give (or any pitch if none), the joint angles, `pose_ok` (the
        pose itself clears the table, the obstacles and the other arm where it is now), `pitch_ranges` (the
        pitches that work at that xyz), or the closest reachable point. It does not check the path between
        points - planning does that when the program runs. Use it to choose hover, grasp, lift and drop
        points before writing a program."""
        out = await robotd.post("/motion/check_reach", {"arm": arm, "points": points})
        if not out.get("ok"):
            raise ToolError(out.get("reason") or "check_reach failed")
        return out

    @mcp.tool(annotations=READ_ONLY)
    async def table_to_pixel(xyz: Xyz) -> dict:
        """Where a rig-frame point appears in the ego image (full-resolution pixels)."""
        try:
            return {"px": [round(v_, 1) for v_ in geo.table_to_pixel(xyz)]}
        except ValueError as e:
            raise ToolError(str(e))

    # ---------------------------------------------------------------- episodes
    @mcp.tool(annotations=MOVES)
    async def begin_episode(instruction: str) -> dict:
        """Start an episode for one human instruction; programs you run are filed under it."""
        eid = f"e{_dt.datetime.now():%Y%m%d-%H%M%S}"
        d = episodes / _dt.datetime.now().strftime("%Y-%m-%d") / eid
        d.mkdir(parents=True, exist_ok=True)
        ep.update({"id": eid, "dir": d, "programs": []})
        (d / "episode.json").write_text(json.dumps({"episode": eid, "instruction": instruction,
                                                    "started": _dt.datetime.now().isoformat()}, indent=1))
        return {"episode": eid, "dir": str(d)}

    @mcp.tool(annotations=MOVES)
    async def end_episode(outcome: Literal["success", "partial", "failure", "aborted"], notes: str = "") -> dict:
        """Close the current episode with its outcome and what you learned (it becomes training data)."""
        if ep["id"] is None:
            raise ToolError("no episode is open")
        _ep_write({"outcome": outcome, "notes": notes, "ended": _dt.datetime.now().isoformat()})
        out = {"episode": ep["id"], "outcome": outcome, "programs": ep["programs"]}
        ep.update({"id": None, "dir": None, "programs": []})
        return out

    # ---------------------------------------------------------------- programs
    @mcp.tool(annotations=MOVES)
    async def run_program(
        code: Annotated[str, Field(description="Python using robot_use (see the API); runs sandboxed")],
        title: str = "",
        scene: Annotated[list[dict] | None, Field(description='objects you believe are there, for the twin preview: '
                                                            '[{"name", "xy": [x, y], "size": [sx, sy, sz]}]')] = None,
        step_mode: Annotated[bool, Field(description="the human approves every step (slow; for risky first runs)")] = False,
        skip_preview: bool = False,
    ) -> dict:
        """Submit a program: static check, twin preview, then human approval, then it runs. Returns at once;
        follow it with program_status(program_id, wait_s)."""
        out = await call(runner.submit, code, title or "", step_mode, scene or [], skip_preview,
                         ep["dir"] if ep["dir"] is not None else None)
        if out.get("program_id") and ep["id"] is not None:
            ep["programs"].append(out["program_id"])
            _ep_write()
        if not out.get("ok"):
            raise ToolError(f"program not accepted: {out.get('reason')}")
        return {k: out.get(k) for k in ("program_id", "status", "dir")}

    @mcp.tool(annotations=READ_ONLY)
    async def program_status(program_id: str,
                             wait_s: Annotated[float, Field(ge=0, le=30, description="wait for a checkpoint or "
                                                                                     "the end")] = 0.0) -> list[Image | TextContent]:
        """A program's status: preview report, steps so far, the last step, a pending checkpoint (with its
        evidence images), the outcome or the error. Statuses: checking, previewing, awaiting_approval (a
        human must approve it on the operator page), running, checkpoint (decide with resume_program),
        done, failed, aborted, stopped, denied, rejected, expired."""
        st = await call(runner.status, program_id, wait_s)
        if not st.get("ok"):
            raise ToolError(st.get("reason", "unknown program"))
        keep = ("program_id", "title", "status", "reason", "steps", "motion_s", "last_step", "preview", "checkpoint",
                "outcome", "output_tail")
        summary = {k: st.get(k) for k in keep}
        content = [TextContent(type="text", text=json.dumps(summary, default=str, indent=1))]
        cp = st.get("checkpoint")
        if st.get("status") == "checkpoint" and cp:
            content.append(TextContent(type="text", text=f"checkpoint {cp['name']!r}: {cp.get('need', '')}. Evidence "
                                       "follows. Decide with resume_program(program_id, 'continue' | 'refine' | 'abort', "
                                       "updates={estimate: [x, y]})."))
            for rel in cp.get("evidence", []):
                jpg = await anyio.to_thread.run_sync(_small_jpeg, Path(st["dir"]) / rel)
                if jpg:
                    content += [Image(data=jpg, format="jpeg"), TextContent(type="text", text=rel)]
        return content

    @mcp.tool(annotations=MOVES)
    async def resume_program(program_id: str, decision: Literal["continue", "refine", "abort"],
                             updates: Annotated[dict[str, list[float]] | None, Field(
                                 description="for refine: {estimate name: [x, y]}, each within its tolerance")] = None) -> dict:
        """The decision at a checkpoint. A refinement beyond an estimate's tolerance is refused (revise the
        program instead)."""
        out = await call(runner.resume, program_id, decision, updates)
        if not out.get("ok"):
            raise ToolError(out.get("reason"))
        return out

    @mcp.tool(annotations=READ_ONLY)
    async def get_program_trace(program_id: str, last: Annotated[int, Field(ge=1, le=200)] = 30) -> dict:
        """What a program did, step by step: arguments, result status, reason, conditions met, grasps, and the
        keyframe files (fetch one with get_keyframe). Read it when something failed."""
        st = await call(runner.status, program_id, 0)
        if not st.get("ok"):
            raise ToolError(st.get("reason", "unknown program"))
        recs = []
        f = Path(st["dir"]) / "trace.jsonl"
        for line in (f.read_text().splitlines() if f.exists() else []):
            r = json.loads(line)
            if r["type"] == "step":
                res = r.get("result") or {}
                rep = res.get("report") or {}
                recs.append({"step": r["n"], "kind": r["kind"], "args": r["args"], "status": res.get("status"),
                             "reason": res.get("reason"), "condition_met": rep.get("condition_met"),
                             "grasped": rep.get("grasped"), "blocked": rep.get("blocked"),
                             "keyframes": r.get("keyframes")})
            elif r["type"] in ("log", "checkpoint", "resume", "status"):
                recs.append({k: v for k, v in r.items() if k not in ("state",)})
        return {"program_id": program_id, "status": st.get("status"), "trace": recs[-last:]}

    @mcp.tool(annotations=READ_ONLY)
    async def get_keyframe(program_id: str, name: str,
                           max_side: Annotated[int, Field(ge=64, le=1280)] = 512) -> list[Image | TextContent]:
        """One keyframe of a program (a path from get_program_trace, e.g. keyframes/step003_after_ego.jpg)."""
        st = await call(runner.status, program_id, 0)
        if not st.get("ok"):
            raise ToolError(st.get("reason", "unknown program"))
        base = Path(st["dir"]).resolve()
        f = (base / name).resolve()
        if base not in f.parents:
            raise ToolError("keyframes live inside the program's directory")
        jpg = await anyio.to_thread.run_sync(_small_jpeg, f, max_side)
        if not jpg:
            raise ToolError(f"no keyframe {name!r}")
        return [Image(data=jpg, format="jpeg"), TextContent(type="text", text=name)]

    @mcp.tool(annotations=MOVES)
    async def stop() -> dict:
        """Stop now: halts both arms (they decelerate and hold) and ends the running program."""
        pid = await call(runner.active)
        if pid:
            await call(runner.stop, pid, "stopped by the agent")
        return {"stopped_program": pid, **(await robotd.post("/motion/stop", {"arm": "both"}))}


def local_hosts() -> list[str]:
    """Host names and addresses this machine answers to, for the Host-header allowlist."""
    names = {"localhost", "127.0.0.1", "[::1]", socket.gethostname()}
    try:
        names.add(socket.getfqdn())
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            names.add(f"[{addr}]" if ":" in addr else addr)
    except OSError:
        pass
    try:                                              # the address used for the default route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))                   # TEST-NET, nothing is sent
        names.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(n for n in names if n)
