#!/usr/bin/env python3
"""Disable torque on SO-101 arms left stiff by a killed lerobot process.

A lerobot process killed with SIGKILL (or a hung Ctrl-C) never runs
disconnect(), so the followers keep holding their last goal position.
This talks to the Feetech bus directly -- no lerobot calibration needed.

    python tools/torque_off.py --status          # read only, nothing moves
    python tools/torque_off.py                   # both followers go limp
    python tools/torque_off.py /dev/so101_leader_left

SUPPORT THE ARM FIRST: with torque off it falls under gravity.

Every port is bounded: it either reports within PORT_DEADLINE_S or is given up
on with a message, and the exit status is non-zero if any servo did not read
back the requested state. Why this is needed: scservo_sdk opens the port with no
write timeout, so a wedged USB adapter blocks the very first write forever
(docs/troubleshooting.md, "Servo transaction hangs").
"""
import argparse
import os
import signal
import sys
import time
import types

import scservo_sdk as scs
import serial

TORQUE_ENABLE = 40  # STS3215 control table
IDS = range(1, 7)   # shoulder_pan .. gripper
DEFAULT_PORTS = ["/dev/so101_follower_left", "/dev/so101_follower_right"]
WRITE_TIMEOUT_S = 0.3   # a packet takes ~0.1 ms at 1 Mbps; longer means the adapter is wedged
PORT_DEADLINE_S = 8.0   # opening a wedged adapter alone costs one 5 s USB control timeout
WEDGED = ("the USB adapter is not responding: power-cycle the arm hub "
          "(docs/troubleshooting.md, 'USB hub power')")


class PortDeadline(Exception):
    """Not an OSError, so pyserial's EINTR handling cannot swallow or rewrap it."""


def _setup_port(self, cflag_baud):
    """PortHandler.setupPort with a finite write_timeout (the SDK leaves it infinite)."""
    if self.is_open:
        self.closePort()
    self.ser = serial.Serial(port=self.port_name, baudrate=self.baudrate, bytesize=serial.EIGHTBITS,
                             timeout=0, write_timeout=WRITE_TIMEOUT_S)
    self.is_open = True
    self.ser.reset_input_buffer()
    self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
    return True


def _bounded_clear(self):
    """clearPort is ser.flush() == tcdrain(), unbounded on a wedged adapter."""
    deadline = time.monotonic() + WRITE_TIMEOUT_S
    while self.ser.out_waiting and time.monotonic() < deadline:
        time.sleep(0.001)
    if self.ser.out_waiting:
        self.ser.reset_output_buffer()


def txn(ph, fn, *args):
    """One SDK transaction; an I/O error becomes a result code and frees the SDK's busy flag."""
    try:
        return fn(*args)
    except serial.SerialException:        # includes SerialTimeoutException from write()
        ph.is_using = False
        try:
            ph.ser.reset_output_buffer()
        except Exception:
            pass
        return None


def do_port(port, status_only, baud):
    """Returns (ok, line)."""
    if not os.path.exists(port):
        return False, f"{port}: cannot open: no such device (unplugged, or udev rules missing)"
    ph = scs.PortHandler(port)
    ph.setupPort = types.MethodType(_setup_port, ph)
    ph.clearPort = types.MethodType(_bounded_clear, ph)
    pk = scs.PacketHandler(0)
    t0 = time.monotonic()
    try:
        ph.setBaudRate(baud)
    except (serial.SerialException, OSError) as e:
        return False, f"{port}: cannot open: {e}"
    if time.monotonic() - t0 > 1.0:
        print(f"{port}: opening took {time.monotonic() - t0:.1f} s (USB control timeout?)", flush=True)
    ok, states = True, []
    try:
        for i in IDS:
            if not status_only:
                for _ in range(3):
                    res = txn(ph, pk.write1ByteTxRx, ph, i, TORQUE_ENABLE, 0)
                    if res is None:
                        ph.closePort()
                        return False, f"{port}: write to id {i} timed out: {WEDGED}"
                    if res[0] == scs.COMM_SUCCESS:
                        break
            res = None
            for _ in range(3):
                res = txn(ph, pk.read1ByteTxRx, ph, i, TORQUE_ENABLE)
                if res is None or res[1] == scs.COMM_SUCCESS:
                    break
            if res is None:
                ph.closePort()
                return False, f"{port}: read of id {i} timed out: {WEDGED}"
            val, comm, _ = res
            if comm != scs.COMM_SUCCESS:
                states.append(f"{i}:??")
                ok = False
            else:
                states.append(f"{i}:{'ON' if val else 'off'}")
                ok &= status_only or val == 0
    finally:
        try:
            ph.ser.reset_output_buffer()
            ph.closePort()
        except Exception:
            pass
    line = f"{port}: torque {' '.join(states)}  ({time.monotonic() - t0:.2f} s)"
    if not ok:
        line += "  <- ?? = no answer (arm 12 V off, or cable)" if "??" in line else "  <- still ON"
    return ok, line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ports", nargs="*", default=DEFAULT_PORTS)
    ap.add_argument("--status", action="store_true", help="only report torque state")
    ap.add_argument("--baud", type=int, default=1_000_000)
    args = ap.parse_args()

    def deadline(signum, frame):
        raise PortDeadline()

    signal.signal(signal.SIGALRM, deadline)
    all_ok = True
    for port in args.ports:
        signal.setitimer(signal.ITIMER_REAL, PORT_DEADLINE_S)
        try:
            ok, line = do_port(port, args.status, args.baud)
        except PortDeadline:
            ok, line = False, f"{port}: no result within {PORT_DEADLINE_S:g} s: {WEDGED}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        print(line, flush=True)
        all_ok &= ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
