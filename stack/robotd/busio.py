"""Feetech serial I/O that cannot block forever.

scservo_sdk opens the port with pyserial's default ``write_timeout=None``, so a
CH343 adapter that stops accepting USB OUT data blocks the first ``write()``
forever, and ``clearPort()`` (``tcdrain``) is just as unbounded. The SDK also
marks the port busy (``PortHandler.is_using``) at the start of ``txPacket`` and
clears it only at the end of ``rxPacket``, so one abandoned transaction makes
every later one fail with COMM_PORT_BUSY. See docs/troubleshooting.md,
"Servo transaction hangs".

Same fix as tools/bench_check.py and tools/torque_off.py, as a module robotd can
import. ``harden_port_handler`` works on a bare ``scs.PortHandler`` and on the
one inside a lerobot ``FeetechMotorsBus`` (patch before ``bus.connect()``).
"""

from __future__ import annotations

import time
import types

WRITE_TIMEOUT_S = 0.3   # a 14-byte packet takes ~0.1 ms at 1 Mbps; 0.3 s means the adapter is wedged
BAUD = 1_000_000


def _packet_timeout(self, packet_length):
    """lerobot's patch_setPacketTimeout (feetech.py): the SDK's own formula is wrong."""
    self.packet_start_time = self.getCurrentTime()
    self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50


def _setup_port(self, cflag_baud):
    """PortHandler.setupPort with a finite write_timeout (the SDK leaves it infinite), opened exclusively."""
    import serial

    if self.is_open:
        self.closePort()
    self.ser = serial.Serial(port=self.port_name, baudrate=self.baudrate, bytesize=serial.EIGHTBITS,
                             timeout=0, write_timeout=WRITE_TIMEOUT_S)
    make_exclusive(self.ser.fileno())
    self.is_open = True
    self.ser.reset_input_buffer()
    self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
    return True


def make_exclusive(fd: int) -> None:
    """TIOCEXCL: the kernel refuses every further open() of this tty (EBUSY) until it is closed, except for
    root. Unlike pyserial's exclusive=True (an advisory flock that lerobot or a stray script would simply
    ignore), this stops another process from opening an arm's port while robotd owns it."""
    import fcntl
    import termios

    fcntl.ioctl(fd, termios.TIOCEXCL)


def _bounded_clear(self):
    """clearPort is ser.flush() == tcdrain(): wait briefly for the output queue, then drop it."""
    deadline = time.monotonic() + WRITE_TIMEOUT_S
    while self.ser.out_waiting and time.monotonic() < deadline:
        time.sleep(0.001)
    if self.ser.out_waiting:
        self.ser.reset_output_buffer()


def harden_port_handler(ph):
    """Bound every blocking call on a scservo_sdk PortHandler. Call before the port is opened."""
    ph.setupPort = types.MethodType(_setup_port, ph)
    ph.setPacketTimeout = types.MethodType(_packet_timeout, ph)
    ph.clearPort = types.MethodType(_bounded_clear, ph)
    return ph


def release_busy(ph):
    """After an I/O error mid-transaction: free the SDK's busy flag and drop queued output."""
    ph.is_using = False
    try:
        ph.ser.reset_output_buffer()
    except Exception:
        pass


def open_servo_port(path, baud=BAUD):
    """(PortHandler, PacketHandler) for a Feetech bus, with every blocking call bounded."""
    import scservo_sdk as scs

    ph = harden_port_handler(scs.PortHandler(path))
    if not ph.setBaudRate(baud):
        raise OSError(f"unsupported baud rate {baud}")
    return ph, scs.PacketHandler(0)


def close_servo_port(ph):
    try:
        ph.ser.reset_output_buffer()   # nothing queued, so close() cannot wait out closing_wait
        ph.closePort()
    except Exception:
        pass


def servo_txn(ph, fn, *args):
    """Run one SDK transaction. Returns (result, exc); on an I/O error the busy flag is released."""
    import serial

    try:
        return fn(*args), None
    except (serial.SerialException, OSError) as e:   # SerialTimeoutException included
        release_busy(ph)
        return None, e
