"""Bounded Feetech I/O.

The pty test checks the property robotd relies on: with the hardened port handler every transaction
on a port that is not draining returns within a bounded time and leaves the SDK's busy flag clear, so
the next transaction is not refused with COMM_PORT_BUSY. It does NOT reproduce the stock SDK's hang:
a pty keeps accepting small writes, unlike the wedged CH343 on 2026-09-27 (docs/troubleshooting.md,
"Servo transaction hangs"), where the stock write blocked in select() forever.
"""

import os
import pty
import time

import pytest

from stack.robotd import busio


_SLAVE_FD = {}


@pytest.fixture
def wedged_port():
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    _SLAVE_FD[path] = slave           # kept open: the ports are opened exclusively, so _fill writes through it
    yield path
    _SLAVE_FD.pop(path, None)
    os.close(master)
    os.close(slave)


def _fill(path):
    """Write into the pty until the kernel refuses more: the adapter has stopped taking data."""
    import fcntl

    fd = os.dup(_SLAVE_FD[path])
    fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    try:
        for chunk in (4096, 1):         # big chunks, then single bytes: no room left even for 8 bytes
            while True:
                try:
                    os.write(fd, b"\0" * chunk)
                except BlockingIOError:
                    break
    finally:
        os.close(fd)


def test_writes_are_bounded_and_busy_flag_released(wedged_port):
    scs = pytest.importorskip("scservo_sdk")
    try:
        ph, pk = busio.open_servo_port(wedged_port, baud=115200)   # a pty cannot take 1 Mbps
    except OSError as e:
        pytest.skip(f"pty rejected the port setup: {e}")
    try:
        for _ in range(3):
            _fill(wedged_port)
            t0 = time.monotonic()
            res, exc = busio.servo_txn(ph, pk.write1ByteTxRx, ph, 1, 40, 0)
            assert time.monotonic() - t0 < 3 * busio.WRITE_TIMEOUT_S, "write was not bounded"
            # a pty sometimes still accepts an 8-byte packet (then exc is None and the call times
            # out waiting for a reply): either way it must return and leave the port usable
            assert ph.is_using is False, "busy flag left set"
        # the output queue was dropped, so the next transaction goes out and simply gets no reply;
        # the old failure mode was COMM_PORT_BUSY here, returned instantly, forever
        res, exc = busio.servo_txn(ph, pk.ping, ph, 1)
        assert exc is None and res[1] != scs.COMM_PORT_BUSY
    finally:
        _fill(wedged_port)
        t2 = time.monotonic()
        busio.close_servo_port(ph)
        assert time.monotonic() - t2 < 2.0, "closing a wedged port blocked"


def test_harden_patches_a_lerobot_bus_port_handler():
    pytest.importorskip("lerobot")
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port="/dev/null", motors={"a": Motor(1, "sts3215", MotorNormMode.DEGREES)})
    busio.harden_port_handler(bus.port_handler)
    assert bus.port_handler.setupPort.__func__ is busio._setup_port
    assert bus.port_handler.clearPort.__func__ is busio._bounded_clear


class _NeverDrains:
    """A serial port whose output queue never empties (an adapter that stopped taking data)."""

    def __init__(self):
        self.reset_calls = 0

    @property
    def out_waiting(self):
        return 0 if self.reset_calls else 64

    def reset_output_buffer(self):
        self.reset_calls += 1


def test_bounded_clear_gives_up_and_drops_the_queue():
    class PH:
        ser = _NeverDrains()

    ph = PH()
    t0 = time.monotonic()
    busio._bounded_clear(ph)
    assert time.monotonic() - t0 < busio.WRITE_TIMEOUT_S + 0.2
    assert ph.ser.reset_calls == 1


@pytest.mark.hardware
def test_ports_are_opened_exclusively(wedged_port):
    """While robotd holds a port, nobody else can open it (lerobot, a stray script, bench_check)."""
    import errno

    ph, pk = busio.open_servo_port(wedged_port, baud=115200)
    try:
        with pytest.raises(OSError) as e:
            os.close(os.open(wedged_port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK))
        assert e.value.errno == errno.EBUSY
    finally:
        busio.close_servo_port(ph)
    # (the kernel clears the flag at the tty's last close; here the fixture still holds the pty open)
