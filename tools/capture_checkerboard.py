"""Capture checkerboard frames for camera calibration, on a headless Jetson.

Grabs frames from one camera, finds the checkerboard in each, and saves only the ones where it is
found, with the corners drawn so you can check them afterwards. Because the Jetson has no display,
the feedback is in the terminal: every grab prints whether the board was found and which cell of a
3x3 grid its centre fell in, and the run ends with a coverage map. A calibration is only as good as
its coverage -- the board must visit all nine cells, including the corners, and be tilted, not just
slid around flat.

Capture at the SAME resolution the camera is used at (ego 1280x720, wrists 1024x576): the intrinsics
are per-mode, because a different mode can crop or scale the sensor.

    ./.venv/bin/python tools/capture_checkerboard.py --camera /dev/cam_ego --width 1280 --height 720 \
        --out ~/calib/ego --cols 9 --rows 6
    # then, on any machine with the frames:
    ./.venv/bin/python tools/calibrate_camera.py --frames ~/calib/ego --squares-mm 25 --cols 9 --rows 6

`--cols`/`--rows` are INNER corners (a board of 10x7 squares has 9x6 inner corners). Default is
auto-grab every --interval seconds; `--manual` waits for Enter instead, which is easier alone.
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so the tools run from any cwd
from board_detect import find_board


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="/dev/cam_ego")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", required=True, help="directory for the accepted frames")
    p.add_argument("--cols", type=int, default=9, help="INNER corners across")
    p.add_argument("--rows", type=int, default=6, help="INNER corners down")
    p.add_argument("--want", type=int, default=30, help="stop after this many accepted frames")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between automatic grabs")
    p.add_argument("--manual", action="store_true", help="press Enter to grab instead of a timer")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # /dev/* is a real camera (V4L2, MJPG); anything else is a video file, used to test this tool
    live = a.camera.startswith("/dev/")
    cap = cv2.VideoCapture(a.camera, cv2.CAP_V4L2 if live else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"[FATAL] cannot open {a.camera} -- is it plugged in, and is anything else (a policy client) holding it?")
    if live: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height); cap.set(cv2.CAP_PROP_FPS, 30)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (a.width, a.height):
        print(f"[warn] asked for {a.width}x{a.height}, camera gave {w}x{h}: calibrate and use the SAME mode")
    if live:
        for _ in range(10): cap.read()                   # let exposure settle
    print(f"[capture] {a.camera} at {w}x{h} | board {a.cols}x{a.rows} inner corners | need {a.want} frames -> {a.out}")
    print("  Hold the board flat and rigid, fully in frame, and TILT it (20-45 deg) as well as moving it.")
    print("  Cover all nine cells of the frame, especially the corners: the lens distortion lives at the edges.\n")

    FLUSH = 6                                            # deeper than the V4L2 buffer (typically 4)
    # Continue the numbering rather than overwriting: a second session into the same directory used
    # to silently replace the first session's frames from frame_000 up.
    prior = [int(m.group(1)) for m in
             (re.match(r"frame_(\d+)\.png$", os.path.basename(f)) for f in glob.glob(os.path.join(a.out, "frame_*.png")))
             if m]
    existing = max(prior) + 1 if prior else 0        # the next free index, NOT the count: deleting a
                                                     # blurred frame must not make the next session
                                                     # overwrite a good one further up
    if existing: print(f"[capture] this directory already holds frames; numbering continues from frame_{existing:03d}\n")
    grid = np.zeros((3, 3), int); kept = 0; tried = 0; dead = 0
    try:
        while kept < a.want:
            if a.manual:
                try: input(f"[{kept}/{a.want}] position the board, then press Enter (Ctrl-C to stop)... ")
                except EOFError: break
            else:
                time.sleep(a.interval)
            if live:
                for _ in range(FLUSH): cap.grab()        # V4L2 queues frames: after a pause the
            ok, frame = cap.read(); tried += 1           # next read is stale, showing the board where it WAS
            if not ok:
                dead += 1
                print(f"  frame grab failed ({dead})")
                if dead >= 10: print("[FATAL] camera stopped delivering frames -- unplug/replug it and start again"); break
                continue
            dead = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners, how = find_board(gray, a.cols, a.rows)
            if not found:
                print(f"  [{tried}] board NOT found (whole board in frame? enough light? not blurred?)"); continue
            cx, cy = corners[:, 0, 0].mean(), corners[:, 0, 1].mean()
            gx, gy = min(2, int(cx / w * 3)), min(2, int(cy / h * 3)); grid[gy, gx] += 1
            vis = frame.copy(); cv2.drawChessboardCorners(vis, (a.cols, a.rows), corners, found)
            path = os.path.join(a.out, f"frame_{existing + kept:03d}.png")
            cv2.imwrite(path, frame)                                  # the raw frame is what the solver uses
            cv2.imwrite(os.path.join(a.out, f"vis_{existing + kept:03d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            kept += 1
            note = "" if how == "sb+accuracy" else f" | via {how}"
            print(f"  [{tried}] FOUND -> {os.path.basename(path)} | centre at grid cell ({gy},{gx}) | kept {kept}/{a.want}{note}")
    except KeyboardInterrupt:
        print("\n[capture] stopped")
    finally:
        cap.release()

    print("\ncoverage (frames whose board centre fell in each third of the image):")
    for row in range(3):
        print("   " + "  ".join(f"{grid[row, c]:3d}" for c in range(3)))
    thin = [(r, c) for r in range(3) for c in range(3) if grid[r, c] == 0]
    if thin: print(f"  EMPTY cells: {thin} -- capture more there before solving, or the distortion fit is unconstrained")
    print(f"\n[capture] {kept} frames in {a.out}. Next:")
    print(f"  ./.venv/bin/python tools/calibrate_camera.py --frames {a.out} --squares-mm <measured> --cols {a.cols} --rows {a.rows}")


if __name__ == "__main__":
    main()
