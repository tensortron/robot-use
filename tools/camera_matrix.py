"""Which resolutions can all cameras stream at the same time?

Three USB 2.0 cameras share one 480 Mbps bus and each reserves isochronous
bandwidth at stream start. Whether a combination fits is only knowable by
trying it: the reservation ladder is a firmware property and is not monotonic
in resolution (640x480 has failed where 1024x576 succeeded).

Runs every listed resolution on every camera simultaneously and reports which
combinations deliver frames on all cameras. Use it after moving cameras between
ports or hubs, or after swapping a camera, to see what changed.

    ./.venv/bin/python tools/camera_matrix.py                # default ladder
    ./.venv/bin/python tools/camera_matrix.py --res 1280x720 --res 1024x576
    ./.venv/bin/python tools/camera_matrix.py --sustain 10   # also measure fps

Stop teleop first: cameras cannot be opened by two processes.
"""

import argparse
import glob
import sys
import threading
import time

import cv2

DEFAULT_LADDER = ["1920x1080", "1280x720", "1024x576", "800x600", "640x480", "640x360"]


def find_cameras(pattern: str) -> list[str]:
    """Capture nodes only (index0); the index1 nodes are UVC metadata.

    udev publishes each camera under two by-path names (…usb-0:… and
    …usbv2-0:…) that resolve to the same /dev/videoN. Dedupe on the real
    device, or a wide glob opens every camera twice and the second open of
    each reports zero frames -- which looks exactly like starvation.
    """
    import os
    seen, out = set(), []
    for path in sorted(glob.glob(pattern)):
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        out.append(path)
    return out


def open_cam(path: str, w: int, h: int) -> cv2.VideoCapture:
    c = cv2.VideoCapture(path, cv2.CAP_V4L2)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    c.set(cv2.CAP_PROP_FPS, 30)
    return c


def trial(paths: list[str], w: int, h: int, probe_frames: int, sustain_s: float):
    caps = []
    for p in paths:
        caps.append(open_cam(p, w, h))
        time.sleep(0.7)   # let each isochronous reservation settle before the next
    time.sleep(1.0)
    counts = [sum(1 for _ in range(probe_frames) if c.read()[0]) for c in caps]
    actual = f"{int(caps[0].get(3))}x{int(caps[0].get(4))}"
    ok = min(counts) > probe_frames * 0.6
    fps = None
    if ok and sustain_s > 0:
        totals = [0] * len(caps)

        def pump(i, c):
            t0 = time.time()
            while time.time() - t0 < sustain_s:
                if c.read()[0]:
                    totals[i] += 1

        ts = [threading.Thread(target=pump, args=(i, c)) for i, c in enumerate(caps)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        fps = [n / sustain_s for n in totals]
    for c in caps:
        c.release()
    time.sleep(1.0)
    return actual, counts, ok, fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="/dev/v4l/by-path/*usb-0:4.1.*:1.0-video-index0",
                    help="which capture nodes to test (default: the three on hub port 4.1.x)")
    ap.add_argument("--res", action="append", help="WxH to try; repeatable. Default: a ladder")
    ap.add_argument("--probe", type=int, default=15, help="frames to read per camera per trial")
    ap.add_argument("--sustain", type=float, default=0,
                    help="seconds of concurrent streaming to measure fps on passing configs")
    args = ap.parse_args()

    paths = find_cameras(args.glob)
    if not paths:
        print(f"no cameras match {args.glob}")
        return 1
    print(f"{len(paths)} cameras:")
    for p in paths:
        print("  " + p.split("/")[-1])
    print()

    ladder = args.res or DEFAULT_LADDER
    print(f"{'requested':<11}{'actual':<11}{'frames':<22}result")
    for r in ladder:
        w, h = (int(x) for x in r.lower().split("x"))
        actual, counts, ok, fps = trial(paths, w, h, args.probe, args.sustain)
        verdict = "ALL OK" if ok else "STARVED (a camera got no frames)"
        if fps:
            verdict += "  " + " ".join(f"{f:.1f}fps" for f in fps)
        print(f"{r:<11}{actual:<11}{str(counts):<22}{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
