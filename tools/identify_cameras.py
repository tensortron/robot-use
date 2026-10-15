"""Identify which camera is which by frame motion, and print the USB path a udev rule needs for each.

Many cheap UVC cameras report the same serial, so /dev/v4l/by-id cannot tell them apart. This finds every camera,
labels it by its USB port path (the `KERNELS` value for udev/99-so101-cameras.rules, e.g. 1-4.1.3:1.0) and, when the udev
rules are installed already, by its /dev/cam_* name. Then move ONE follower arm by hand: the wrist camera bolted to
that arm shows a large frame delta, the others stay near zero. Cover the ego camera's lens to find it.

    .venv/bin/python tools/identify_cameras.py

Never address a camera by /dev/videoN: the numbers move when a camera re-enumerates. The streams here run at
640x480, small enough that all the cameras on one USB 2.0 bus can stream at once while you identify them.
"""

import glob
import os
import time

import cv2
import numpy as np

SIZE = (640, 480)


def cameras() -> dict:
    """{device: label} for every V4L2 capture node (index 0; index 1 is UVC metadata)."""
    names = {}
    for link in glob.glob("/dev/cam_*"):                               # names the udev rules already give
        names[os.path.realpath(link)] = os.path.basename(link)
    found = {}
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            if open(os.path.join(node, "index")).read().strip() != "0":
                continue
            usb = os.path.basename(os.path.realpath(os.path.join(node, "device")))   # KERNELS: 1-4.1.3:1.0
        except OSError:
            continue
        dev = "/dev/" + os.path.basename(node)
        found[dev] = f"{usb}" + (f" {names[dev]}" if dev in names else "")
    if not found:
        raise SystemExit("No camera found. Check the cables, then `ls /dev/video*`.")
    return found


def open_cam(dev, width, height):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def grab_gray(cap):
    ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) if ok else None


def main():
    # Stagger the opens: three streams on one USB 2.0 bus need each isochronous
    # reservation to settle before the next claims bandwidth, or the last one
    # silently gets nothing.
    cams = cameras()
    caps = {}
    for d in cams:
        caps[d] = open_cam(d, *SIZE)
        time.sleep(0.7)
    print("Warming up auto-exposure...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        for c in caps.values():
            c.read()

    prev = {d: grab_gray(c) for d, c in caps.items()}
    print("\nMove ONE follower arm by hand. Ctrl-C to stop.\n")
    print("USB path (KERNELS) for each column, then its /dev/cam_* name if the rules are installed:")
    print("  " + "".join(f"{cams[d]:>28}" for d in caps))

    try:
        while True:
            row = ""
            for d, c in caps.items():
                cur = grab_gray(c)
                if cur is None or prev[d] is None:
                    row += f"{'--':>28}"
                    continue
                delta = float(np.abs(cur - prev[d]).mean())
                prev[d] = cur
                row += f"{('>> ' if delta > 4 else '   ') + f'{delta:6.2f}':>28}"
            print("  " + row, end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")
    finally:
        for c in caps.values():
            c.release()


if __name__ == "__main__":
    main()
