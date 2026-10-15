"""Live camera view with the capture grid, board detection and tilt, over HTTP.

The Orin is headless, so there is no imshow. This serves the camera as MJPEG
with the calibration overlay drawn on it: the 3x3 grid `capture_checkerboard.py`
reports cells against, and -- when the board is visible -- its centre cell,
square size and TILT.

Tilt is the reason this exists. A board slid around flat on the table reads the
same angle in every frame; the solver then cannot separate focal length from
distance and returns a confident wrong number with a small reprojection error.
The capture tool only checks that tilt reached 20 deg once, not that it VARIED,
so watch this and make each pose a different angle.

    ./.venv/bin/python tools/live_view.py --camera /dev/cam_ego --width 1280 --height 720
    # laptop:  ssh -L 8099:localhost:8099 <user>@<robot-host>   then open http://localhost:8099

Stop it before running capture_checkerboard.py: one process per camera.
"""

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from guided_capture import GuidedCapture

BOUNDARY = "frameboundary"

_latest = {"jpeg": None}
_lock = threading.Lock()


def grabber(a):
    cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        print(f"[FATAL] cannot open {a.camera} -- is anything else holding it?", flush=True)
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or a.width
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or a.height
    print(f"[live] {a.camera} at {W}x{H} -> http://0.0.0.0:{a.port}", flush=True)
    guided = GuidedCapture(a.cols, a.rows, a.squares_mm, a.capture_to, a.want, a.cooldown, a.min_sharpness,
                           log=lambda m: print(m, flush=True))
    if a.capture_to:
        print(f"[guided] capturing to {a.capture_to} (starting at {guided.kept}); "
              f"target {a.want} frames over {len(guided.slots)} cell/tilt slots", flush=True)
    announced = False
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = guided.step(frame)
        if guided.done and not announced:
            announced = True
            print(f"\n[guided] DONE: {guided.kept} frames, all {len(guided.slots)} slots covered. "
                  f"Ctrl-C, then:\n  ./.venv/bin/python tools/calibrate_camera.py "
                  f"--frames {a.capture_to} --squares-mm {a.squares_mm:g}\n", flush=True)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _latest["jpeg"] = buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = None
        try:
            while True:
                with _lock:
                    frame = _latest["jpeg"]
                if frame is not None and frame is not last:
                    last = frame
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                else:
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--camera", default="/dev/cam_ego")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--squares-mm", type=float, default=25.0)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--capture-to", help="directory: enables GUIDED capture (auto-saves a frame "
                                        "whenever a pose fills an uncovered cell/tilt slot)")
    p.add_argument("--want", type=int, default=27,
                   help="stop after this many frames (27 = one per cell/tilt slot; more tops up)")
    p.add_argument("--cooldown", type=float, default=0.8, help="seconds between auto-captures")
    p.add_argument("--min-sharpness", type=float, default=60.0,
                   help="reject frames blurrier than this (Laplacian variance)")
    a = p.parse_args()
    threading.Thread(target=grabber, args=(a,), daemon=True).start()
    time.sleep(1.5)
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
