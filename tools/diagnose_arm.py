"""Per-servo health probe for one SO-101 bus.

`broadcast_ping` only says "something answered". This pings each id in turn, so
the *pattern* of failure localises the fault:

  ids 1-6 all answer      bus healthy
  nothing answers         no 12 V to the driver board, or the first servo cable
                          is unseated (the adapter still enumerates on USB
                          because its chip is USB-powered)
  1..N answer, rest dead  daisy chain breaks after servo N -- suspect that
                          servo or the cable leaving it

Each ping is bounded by a watchdog thread: a silent Feetech bus otherwise
blocks for the full serial timeout per attempt, and a dead bus can wedge for
tens of seconds.

    sg dialout -c "./.venv/bin/python tools/diagnose_arm.py /dev/so101_follower_left"
"""

import argparse
import sys
import threading
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTORS = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(JOINTS, 1)}


def bounded(fn, timeout_s):
    """Run fn in a daemon thread; return (value, timed_out)."""
    box, err = [None], [None]

    def work():
        try:
            box[0] = fn()
        except BaseException as e:  # noqa: BLE001
            err[0] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, True
    if err[0] is not None:
        raise err[0]
    return box[0], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--rounds", type=int, default=1, help="repeat, to expose intermittency")
    ap.add_argument("--timeout", type=float, default=3.0, help="seconds per ping")
    args = ap.parse_args()

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    try:
        _, timed_out = bounded(lambda: bus.connect(handshake=False), 10)
        if timed_out:
            # The open is stuck on a thread that still holds the port; closing
            # it from here makes that thread's call raise and unwind.
            try:
                bus.port_handler.closePort()
            except Exception:
                pass
            print(f"{args.port}: could not even open the port (timed out)")
            return 1
    except Exception as e:
        print(f"{args.port}: open failed -> {type(e).__name__}: {e}")
        return 1

    print(f"{args.port}")
    ok_total = 0
    for r in range(args.rounds):
        row, t0 = [], time.time()
        for name, m in MOTORS.items():
            try:
                res, timed_out = bounded(lambda mid=m.id: bus.ping(mid, num_retry=0), args.timeout)
            except Exception:
                res, timed_out = None, False
            if timed_out:
                # A ping stuck in the SDK still owns the port. Close it under the
                # stuck thread so it errors out, then reopen for the next ping;
                # otherwise two readers would split the bus's bytes.
                try:
                    bus.port_handler.closePort()
                    bus.port_handler.openPort()
                except Exception:
                    pass
            alive = (not timed_out) and res is not None
            ok_total += alive
            row.append(f"{m.id}:{'ok' if alive else ('TMO' if timed_out else '--')}")
        print(f"  round {r + 1}: " + "  ".join(row) + f"   ({time.time() - t0:.1f}s)")

    expected = len(MOTORS) * args.rounds
    print(f"  {ok_total}/{expected} pings answered")
    try:
        bus.disconnect(disable_torque=False)
    except Exception:
        pass
    return 0 if ok_total == expected else 2


if __name__ == "__main__":
    sys.exit(main())
