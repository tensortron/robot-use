"""robotd's two HTTP faces (stdlib only).

- **API** (localhost, default :8770): what the MCP server and agent code talk to. JSON state and
  limits, JPEG frames. No motion endpoints until step 3.
- **Operator** (LAN, default :8772): the human's page, live MJPEG with the model overlay, STOP and
  torque off. Served by robotd itself so it keeps working if the MCP server is down.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

OPERATOR_PAGE = Path(__file__).with_name("operator.html")
MAX_SIDE_LIMIT = 1280


def encode_jpeg(frame, max_side: int = 512, quality: int = 80) -> bytes:
    import cv2

    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _int(q, key, default, lo, hi):
    try:
        v = int(q.get(key, [default])[0])
    except ValueError:
        raise ValueError(f"{key} must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be in [{lo}, {hi}]")
    return v


class _Handler(BaseHTTPRequestHandler):
    robotd = None               # set per server class
    operator = False
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=HTTPStatus.OK):
        from .daemon import _jsonable

        self._send(code, json.dumps(obj, default=_jsonable).encode(), "application/json")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        parts = [p for p in url.path.split("/") if p]
        try:
            if url.path == "/state":
                return self._json(self.robotd.state())
            if url.path == "/limits":
                return self._json(self.robotd.limits())
            if url.path == "/health":
                return self._json({"ok": True})
            if url.path == "/config":
                return self._json(self.robotd.config_summary())
            if len(parts) == 2 and parts[0] == "motion":
                wait = float(q.get("wait_s", ["0"])[0])
                return self._json(self.robotd.motion_status(parts[1], wait))
            if len(parts) == 2 and parts[0] == "program":
                return self._json(self.robotd.program(parts[1], float(q.get("wait_s", ["0"])[0])))
            if self.operator and url.path == "/programs":
                return self._json(self.robotd.pending_programs())
            if self.operator and url.path == "/history":
                return self._send(HTTPStatus.OK, self.robotd.history_json(), "application/json")
            if self.operator and len(parts) == 2 and parts[0] == "history":
                return self._json(self.robotd.program_history(parts[1]))
            if len(parts) == 2 and parts[0] == "frame":
                return self._frame(parts[1], q)
            if self.operator and url.path == "/":
                return self._send(HTTPStatus.OK, OPERATOR_PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.operator and len(parts) == 2 and parts[0] == "stream":
                return self._stream(parts[1], q)
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except KeyError as e:
            return self._error(HTTPStatus.NOT_FOUND, f"unknown {e}")
        return self._error(HTTPStatus.NOT_FOUND, f"no route {url.path}")

    def _frame(self, camera, q):
        max_side = _int(q, "max_side", 512, 64, MAX_SIDE_LIMIT)
        quality = _int(q, "quality", 80, 10, 95)
        overlay = q.get("overlay", ["0"])[0] in ("1", "true")
        frame, ts = self.robotd.frame(camera, overlay)
        if frame is None:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"no frame yet from {camera}")
        self._send(HTTPStatus.OK, encode_jpeg(frame, max_side, quality), "image/jpeg",
                   {"X-Capture-Time": f"{ts:.3f}", "X-Frame-Age-S": f"{time.time() - ts:.3f}"})

    def _stream(self, camera, q):
        max_side = _int(q, "max_side", 960, 64, MAX_SIDE_LIMIT)
        overlay = q.get("overlay", ["0"])[0] in ("1", "true")
        fps = _int(q, "fps", 10, 1, 30)
        self.robotd.frame(camera)                       # KeyError -> 404 before headers go out
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                t0 = time.time()
                frame, _ = self.robotd.frame(camera, overlay)
                if frame is not None:
                    jpg = encode_jpeg(frame, max_side, 75)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        # Cross-site requests: a web page elsewhere can POST "simple" requests (form content types) to this
        # port without the browser asking first. Require JSON (which forces a CORS preflight robotd never
        # answers) and, when a browser names its origin, that it is this server.
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "POST bodies must be Content-Type: application/json")
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != (self.headers.get("Host") or ""):
            return self._error(HTTPStatus.FORBIDDEN, f"cross-origin request from {origin} refused")
        if not _host_allowed(self.headers.get("Host") or ""):
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header (use this machine's IP or name)")
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "body must be JSON")
        rd = self.robotd
        try:
            if not self.operator and len(parts) == 2 and parts[0] == "motion":
                return self._json(self._motion(parts[1], body))
            if not self.operator and len(parts) == 2 and parts[0] == "program":
                if parts[1] not in _PROGRAM_ARGS:
                    return self._error(HTTPStatus.NOT_FOUND, f"no program action {parts[1]!r}")
                unknown = set(body) - set(_PROGRAM_ARGS[parts[1]])
                if unknown:
                    raise ValueError(f"unknown fields {sorted(unknown)}")
                fn = {"request": rd.request_program, "start": rd.start_program, "end": rd.end_program,
                      "progress": rd.program_progress}[parts[1]]
                return self._json(fn(**body))
            if self.operator and len(parts) == 3 and parts[0] == "program" and parts[2] in ("approve", "deny"):
                return self._json(rd.decide_program(parts[1], parts[2] == "approve", body.get("step_mode")))
            if not self.operator and len(parts) == 2 and parts[0] == "lease":
                if parts[1] not in _LEASE_ARGS:
                    return self._error(HTTPStatus.NOT_FOUND, f"no lease action {parts[1]!r}")
                unknown = set(body) - set(_LEASE_ARGS[parts[1]])
                if unknown:
                    raise ValueError(f"unknown fields {sorted(unknown)}")
                fn = {"acquire": rd.acquire_lease, "renew": rd.renew_lease, "release": rd.release_lease}[parts[1]]
                return self._json(fn(**body))
            if self.operator and url.path == "/stop":
                return self._json(rd.halt("both"))
            if self.operator and url.path == "/arm":
                return self._json(rd.approve())
            if self.operator and url.path == "/mode":
                return self._json(rd.set_approval_mode(str(body.get("mode", ""))))
            if self.operator and len(parts) == 2 and parts[0] == "reset":
                return self._json(rd.reset_arm(parts[1]))
            if self.operator and len(parts) == 2 and parts[0] == "torque_off":
                if body.get("confirm") is not True:
                    return self._error(HTTPStatus.BAD_REQUEST, 'torque off drops the arm: send {"confirm": true}')
                try:
                    return self._json(rd.torque_off(parts[1]))
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, f"unknown arm {parts[1]}")
        except (TypeError, ValueError, KeyError) as e:
            return self._error(HTTPStatus.BAD_REQUEST, f"bad request: {e}")
        return self._error(HTTPStatus.NOT_FOUND, f"no route {url.path}")


_MOTION_ARGS = {
    "move_joints": ("arm", "targets", "speed", "until", "lease"),
    "set_gripper": ("arm", "value", "speed", "max_effort", "until", "lease"),
    "move_pose": ("arm", "xyz", "pitch", "roll", "linear", "speed", "until", "lease"),
    "trajectory": ("arm", "waypoints", "time_scale", "until", "lease"),
    "validate_trajectory": ("arm", "waypoints", "time_scale"),
    "solve_ik": ("arm", "xyz", "pitch", "roll"),
    "check_reach": ("arm", "points"),
    "view_pose": ("arm", "target", "distance", "elevations", "start"),
    "go_rest": ("arm", "lease"),
    "stop": ("arm",),                      # anyone may stop, lease or not
}
_LEASE_ARGS = {"acquire": ("holder", "ttl_s"), "renew": ("token", "ttl_s"), "release": ("token",)}
_PROGRAM_ARGS = {"request": ("program_id", "holder", "title", "code", "preview", "step_mode"),
                 "start": ("program_id", "lease"), "end": ("program_id", "lease", "outcome"),
                 "progress": ("program_id", "lease", "step", "kind", "checkpoint")}


def _call_motion(robotd, action: str, body: dict):
    if action not in _MOTION_ARGS:
        raise KeyError(f"unknown motion action {action!r}")
    unknown = set(body) - set(_MOTION_ARGS[action])
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)} for {action}; allowed {list(_MOTION_ARGS[action])}")
    fn = {"trajectory": robotd.run_trajectory, "stop": robotd.halt}.get(action) or getattr(robotd, action)
    return fn(**body)


_Handler._motion = lambda self, action, body: _call_motion(self.robotd, action, body)


def _host_allowed(host_header: str) -> bool:
    """DNS rebinding: a page on evil.example re-pointed at this machine sends Host (and Origin) evil.example.
    Accept an IP literal (it cannot be rebound), localhost, or this machine's own name."""
    import ipaddress
    import socket

    h = host_header.strip()
    if h.startswith("["):                                     # [v6]:port
        h = h[1:h.find("]")] if "]" in h else h
    elif h.count(":") == 1:
        h = h.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    name = socket.gethostname().lower()
    return h.lower() in {"localhost", name, name + ".local", name.split(".")[0]}


def serve(robotd, host: str, port: int, operator: bool) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"robotd": robotd, "operator": operator})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name=f"http-{'operator' if operator else 'api'}", daemon=True).start()
    return srv
