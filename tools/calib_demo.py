"""A pretend SO-101 bus for `calibrate.py --demo` and the tests: no hardware, the same calls as lerobot's bus.

Its joints sweep slowly through a range, as if a person moved them. It keeps every register written, and it refuses
to turn torque on: the calibration must never do that.
"""

from __future__ import annotations

import math
import time

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class DemoBus:
    def __init__(self, port: str = "demo", sweep_ticks: float = 900.0, period_s: float = 4.0):
        self.port = port
        self.motors = {j: i + 1 for i, j in enumerate(JOINTS)}
        self.regs = {j: {"Homing_Offset": 100 + 10 * i, "Min_Position_Limit": 700 + i, "Max_Position_Limit": 3400 - i,
                         "Operating_Mode": 0, "Torque_Enable": 1, "Lock": 1} for i, j in enumerate(JOINTS)}
        self.writes = []                     # (register, joint, value) in order
        self.sweep_ticks, self.period_s = sweep_ticks, period_s
        self.t0 = time.monotonic()
        self.connected = False

    def connect(self, handshake: bool = True):
        self.connected = True

    def disconnect(self, disable_torque: bool = True):
        if disable_torque:
            self.disable_torque()
        self.connected = False

    def _actual(self, j) -> float:
        """Where the joint is, in raw encoder ticks before the homing offset (a slow sweep around 2048)."""
        i = JOINTS.index(j)
        return 2048 + self.sweep_ticks * math.sin((time.monotonic() - self.t0) * 2 * math.pi / self.period_s + i)

    def read(self, reg, j, normalize=True, num_retry=0):
        if reg == "Present_Position":
            return int(self._actual(j) - self.regs[j]["Homing_Offset"])
        return self.regs[j][reg]

    def write(self, reg, j, value, normalize=True, num_retry=0):
        self.writes.append((reg, j, int(value)))
        self.regs[j][reg] = int(value)

    def sync_read(self, reg, motors=None, normalize=True, num_retry=0):
        return {j: self.read(reg, j) for j in (motors or JOINTS)}

    def disable_torque(self, motors=None, num_retry=0):
        for j in (motors or JOINTS):
            self.write("Torque_Enable", j, 0)
            self.write("Lock", j, 0)

    def enable_torque(self, motors=None, num_retry=0):
        raise AssertionError("the calibration must never turn torque on")

    def set_half_turn_homings(self, motors=None):
        """lerobot's: homing offsets to zero and the full range first, then each offset so the present position reads
        half a turn (2047)."""
        names = list(motors or JOINTS)
        for j in names:
            self.write("Homing_Offset", j, 0)
            self.write("Min_Position_Limit", j, 0)
            self.write("Max_Position_Limit", j, 4095)
        out = {}
        for j in names:
            out[j] = int(self._actual(j)) - 2047
            self.write("Homing_Offset", j, out[j])
        return out
