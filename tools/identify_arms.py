#!/usr/bin/env python
"""Identify which SO-101 arm sits on which serial bus.

All four buses look identical at rest (6x STS3215, ids 1-6), so the only
reliable discriminator is motion. Run this, then move ONE arm by hand: the row
whose `moved` column jumps is that arm. Note its serial, repeat per arm.

    sg dialout -c "$PWD/.venv/bin/python tools/identify_arms.py"
"""

import glob
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS, 1)}


def read(bus):
    return list(bus.sync_read("Present_Position", normalize=False).values())


def main():
    buses = {}
    for port in sorted(glob.glob("/dev/serial/by-id/usb-1a86*")):
        serial = port.split("_")[-1].replace("-if00", "")
        bus = FeetechMotorsBus(port=port, motors=MOTORS)
        bus.connect(handshake=False)
        buses[serial] = bus

    if not buses:
        raise SystemExit("No /dev/serial/by-id/usb-1a86* devices found.")

    base = {s: read(b) for s, b in buses.items()}
    print(f"Watching {len(buses)} buses. Move ONE arm by hand; Ctrl-C to stop.\n")
    print("  " + "".join(f"{s:>14}" for s in buses))

    try:
        while True:
            deltas = {}
            for s, b in buses.items():
                try:
                    now = read(b)
                    deltas[s] = max(abs(a - c) for a, c in zip(base[s], now))
                except Exception:
                    deltas[s] = -1
            hot = max(deltas, key=deltas.get)
            row = "".join(
                f"{('>> ' if s == hot and deltas[s] > 25 else '   ') + str(deltas[s]):>14}"
                for s in buses
            )
            print("  " + row, end="\r", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n\nLargest movement seen per bus (baseline-relative).")
    finally:
        for b in buses.values():
            try:
                b.disconnect(disable_torque=False)  # read-only probe: never drop a live arm
            except Exception:
                pass


if __name__ == "__main__":
    main()
