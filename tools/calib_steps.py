"""The guided calibration's steps, apart from the web page (tools/calibrate.py): what each step reads and writes.

- `RigEditor`: changes named values in robot.yaml and keeps everything else (comments, order, the other values), each
  value with its `sources` line.
- `ArmCalibration`: lerobot's calibration of one SO-101 follower (the same file `lerobot-calibrate` writes), driven
  from a page instead of the terminal. It only turns torque OFF and reads; it never turns torque on.
- `State`: the tool's progress, so it resumes and skips what is done.
- `camera_entry_paths`, `portable`: where calibration files go, and the provenance strings written into them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FULL_TURN = "wrist_roll"                   # lerobot: its range is the whole turn, 0..4095, not recorded
MIN_RANGE_TICKS = 300                      # ~26 deg: less means the joint was not moved through its range


def today() -> str:
    return date.today().isoformat()


def portable(path) -> str:
    """A provenance path without this machine's home or user: under the data folder it reads $ROBOT_USE_HOME/...,
    in the repository it is relative, elsewhere ~/... ."""
    from stack import paths
    p = Path(path).expanduser().resolve()
    for root, label in ((paths.base(), "$ROBOT_USE_HOME"), (REPO, None), (Path.home().resolve(), "~")):
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        return str(rel) if label is None else f"{label}/{rel}"
    return str(p)


# --------------------------------------------------------------------------------------------- robot.yaml
class RigEditor:
    """Line edits of robot.yaml: only the lines of the named values change, so its comments, its wrapping and every
    other value stay exactly as they were. The YAML parser (ruamel) only finds where each key is; after every edit
    the text is parsed again, and `save` checks that every value set reads back as set."""

    WIDTH = 118

    def __init__(self, path):
        self.path = Path(path)
        text = self.path.read_text()
        self.lines = text.splitlines()
        import yaml as pyyaml
        self.raw0 = pyyaml.safe_load(text)       # every value not edited must read back exactly as this
        self.expect = []                         # (keys, value) to verify on save
        self._parse()

    def _parse(self):
        from ruamel.yaml import YAML
        self.data = YAML().load("\n".join(self.lines) + "\n")

    def get(self, *keys, default=None):
        d = self.data
        for k in keys:
            if not hasattr(d, "get") or k not in d:
                return default
            d = d[k]
        return d

    def _block_end(self, start: int, col: int) -> int:
        """The line after the block that starts at `start` (a key at column `col`): the first later line, not blank,
        indented at most `col`."""
        i = start + 1
        while i < len(self.lines):
            ln = self.lines[i]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= col:
                break
            i += 1
        while i > start + 1 and not self.lines[i - 1].strip():       # trailing blank lines belong to the gap
            i -= 1
        return i

    @staticmethod
    def _scalar(v) -> str:
        import yaml as pyyaml
        if isinstance(v, bool) or v is None:
            return pyyaml.safe_dump(v).split("\n")[0]
        if isinstance(v, float):                 # fixed point: YAML 1.1 (PyYAML) reads 4e-05 as a string
            text = f"{round(v, 6):.6f}".rstrip("0")
            return text + "0" if text.endswith(".") else text
        if isinstance(v, int):
            return str(v)
        text = pyyaml.safe_dump(str(v), width=10_000, default_style=None).rstrip("\n")
        return text[:-4].rstrip() if text.endswith("\n...") else text

    def _render(self, key, value, col: int, comment: str = "", comment_col: int = 0) -> list[str]:
        pad = " " * col
        if isinstance(value, dict):
            out = [f"{pad}{key}:"]
            for k, v in value.items():
                out += self._render(k, v, col + 2)
            return out
        if isinstance(value, (list, tuple)):
            line = f"{pad}{key}: [{', '.join(self._scalar(v) for v in value)}]"
        elif isinstance(value, str) and len(f"{pad}{key}: {self._scalar(value)}") > self.WIDTH \
                and self._scalar(value) == value:
            import textwrap                    # a long plain string: wrapped like the file's own `sources` lines
            parts = textwrap.wrap(value, self.WIDTH - col - 2 - len(str(key)), break_on_hyphens=False)
            return [f"{pad}{key}: {parts[0]}"] + textwrap.wrap(" ".join(parts[1:]), self.WIDTH - col - 2,
                                                               initial_indent=pad + "  ", subsequent_indent=pad + "  ",
                                                               break_on_hyphens=False)
        else:
            line = f"{pad}{key}: {self._scalar(value)}"
        if comment:
            at = comment_col or col + 39
            line = f"{line:<{at}}{comment}" if len(line) < at else f"{line}  {comment}"
        return [line]

    def set(self, keys: tuple, value, source: str | None = None, source_key: str | None = None,
            comment: str | None = None):
        """Set data[keys...] = value, and its `sources` line: under the first key's `sources`, named `source_key`
        (default the last key). A missing key is added at the end of its mapping (before its `sources`). The line's
        own comment stays, unless `comment` replaces it."""
        parent = self.get(*keys[:-1])
        if parent is None:
            raise KeyError(f"robot.yaml has no {'.'.join(keys[:-1])}")
        k = keys[-1]
        if k in parent:
            line, col = parent.lc.key(k)
            end = self._block_end(line, col)
            old = self.lines[line]
            old_comment, ccol = "", 0
            m = re.search(r"\s#", old[old.index(":"):])       # a comment starts at whitespace + '#', as in YAML
            if end == line + 1 and not isinstance(value, dict) and m:
                ccol = old.index(":") + m.start() + 1
                old_comment = old[ccol:]
            if comment is not None:
                old_comment = f"# {comment}" if comment else ""
            self.lines[line:end] = self._render(k, value, col, old_comment, ccol)
        else:
            keys_ = [x for x in parent if x != "sources"]
            if keys_:
                last = keys_[-1]
                line, col = parent.lc.key(last)
                at = self._block_end(line, col)
            else:
                pline, pcol = self.get(*keys[:-2]).lc.key(keys[-2])
                col, at = pcol + 2, pline + 1
            self.lines[at:at] = self._render(k, value, col, f"# {comment}" if comment else "")
        self._parse()
        keys = tuple(keys)                           # a new edit supersedes earlier ones at or inside it; an earlier
        self.expect = [(e, v) for e, v in self.expect if e[:len(keys)] != keys]    # block set stays (it replaced all)
        self.expect.append((keys, value))
        if source is not None:
            if "sources" not in self.data[keys[0]]:
                raise KeyError(f"robot.yaml's {keys[0]} has no sources")
            self.set((keys[0], "sources", source_key or k), source)

    def delete(self, keys: tuple):
        parent = self.get(*keys[:-1])
        if parent is None or keys[-1] not in parent:
            return
        line, col = parent.lc.key(keys[-1])
        del self.lines[line:self._block_end(line, col)]
        self._parse()
        self.expect = [(k, v) for k, v in self.expect if k[:len(keys)] != tuple(keys)]
        self.expect.append((tuple(keys), _DELETED))

    def save(self, backup: bool = True, backup_dir: Path | None = None):
        """Check, then write the file. The previous one is kept as `robot.yaml.bak-<time>` in `backup_dir` (default
        $ROBOT_USE_HOME/calibration/backups): not beside robot.yaml, where the operating agent could read the camera's
        pose and git would see it."""
        import copy

        import yaml as pyyaml

        from stack.rig.rig import Rig
        text = "\n".join(self.lines) + "\n"
        raw = pyyaml.safe_load(text)
        for keys, value in self.expect:              # every edit reads back as made
            d = raw
            for k in keys[:-1]:
                d = d[k]
            if value is _DELETED:
                if keys[-1] in d:
                    raise ValueError(f"robot.yaml: {'.'.join(map(str, keys))} was to be deleted but is still there")
                continue
            got = d[keys[-1]]
            if not _same(got, value):
                raise ValueError(f"robot.yaml edit of {'.'.join(map(str, keys))} reads back as {got!r}, not {value!r}")
        expected = copy.deepcopy(self.raw0)          # ... and nothing else changed
        for keys, value in self.expect:
            src, dst = raw, expected
            for k in keys[:-1]:
                src = src[k]
                if not isinstance(dst.get(k), dict):
                    dst[k] = {}
                dst = dst[k]
            if value is _DELETED:
                dst.pop(keys[-1], None)
            else:
                dst[keys[-1]] = src[keys[-1]]
        if expected != raw:
            raise ValueError("robot.yaml edit changed values it was not asked to change; nothing written")
        Rig(raw, self.path)                          # refuses an invalid rig before anything is written
        if backup and self.path.exists():
            from stack import paths
            bdir = Path(backup_dir) if backup_dir else paths.calibration() / "backups"
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, bdir / f"{self.path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, self.path)


_DELETED = object()


def _same(got, want) -> bool:
    """What PyYAML read back equals what was set (floats as written: to 6 decimals)."""
    if isinstance(want, dict):
        return isinstance(got, dict) and set(got) >= set(want) and all(_same(got[k], v) for k, v in want.items())
    if isinstance(want, (list, tuple)):
        return isinstance(got, list) and len(got) == len(want) and all(_same(g, w) for g, w in zip(got, want))
    if isinstance(want, float):
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - round(want, 6)) < 1e-9
    return type(got) is type(want) and got == want


def replace_file(tmp: Path, final: Path, backup_dir: Path | None = None):
    """Move a finished file into place, keeping the previous one in `backup_dir` (or beside it)."""
    final = Path(final)
    if final.exists():
        bdir = Path(backup_dir) if backup_dir else final.parent
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, bdir / f"{final.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(tmp, final)


def camera_entry_paths(calib_dir: Path, camera: str) -> tuple[Path, Path, str, str]:
    """(intrinsics file, extrinsics file, and how robot.yaml names them): relative to the repository when the files
    are in it, absolute otherwise (robot.yaml resolves relative paths against the repository)."""
    calib_dir = Path(calib_dir).resolve()
    intr, ext = calib_dir / f"{camera}_intrinsics.json", calib_dir / f"{camera}_extrinsics.json"

    def name(p):
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)
    return intr, ext, name(intr), name(ext)


# --------------------------------------------------------------------------------------------- the arms
def feetech_bus(port: str):
    """lerobot's bus for one SO-101 follower, without a calibration (the calibration is what we are making)."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    from stack.robotd import busio
    motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
              for i, j in enumerate(ARM_JOINTS)}
    bus = FeetechMotorsBus(port=port, motors=motors)
    busio.harden_port_handler(bus.port_handler)
    return bus


class ArmError(RuntimeError):
    pass


def _servo_values(d) -> dict:
    """{joint: {homing_offset, range_min, range_max}} for every joint, as ints, or ValueError."""
    out = {}
    for j in ARM_JOINTS:
        e = d[j]
        out[j] = {k: int(e[k]) for k in ("homing_offset", "range_min", "range_max")}
    return out


class ArmCalibration:
    """lerobot's SO-101 calibration (SOFollower.calibrate), one arm, step by step:

    1. `go_limp()`: remember what the servos hold now (homing offsets and limits), then torque OFF. The arm falls
       under gravity: the page asks the person to hold it first.
    2. `set_middle()`: with the arm in the middle of its range, lerobot's half-turn homing (written to the servos).
       From here the positions are sampled: move every joint except the wrist roll through its whole range.
    3. `finish()`: the ranges recorded; the calibration written to the servos and to the file lerobot reads.
       With `restore_after` (a trial), the file is written but the servos get back what they held before.

    `cancel()` puts back what the servos held, if homing had changed them. Torque is never turned on: torque comes on
    only through robotd's `hold_here` (present position as the goal first). `bus_factory(port)` makes the bus (a fake
    one in the tests)."""

    def __init__(self, name: str, port: str, calibration_id: str, out_path: Path, restore_after: bool = False,
                 bus_factory=feetech_bus, pending_path: Path | None = None):
        self.name, self.port, self.calibration_id = name, port, calibration_id
        self.out_path, self.restore_after, self.bus_factory = Path(out_path), restore_after, bus_factory
        self.bus = None
        self.phase = "idle"                    # idle -> limp -> ranging -> done | cancelled | error
        self.original = None                   # {joint: {homing_offset, range_min, range_max}} as the servos held
        self.touched = False                   # the servos' calibration differs (or may differ) from `original`
        self.message = ""
        # `original` also on disk while the servos may differ from it, so a restart (or a crash) can still restore it
        self.pending_path = Path(pending_path) if pending_path else None
        if self.pending_path and self.pending_path.exists():
            self.touched, self.phase = True, "error"
            try:
                self.original = _servo_values(json.loads(self.pending_path.read_text()))
                self.message = ("a previous run left this arm's servos with a changed calibration: Restore puts back "
                                "what they held before it")
            except (OSError, ValueError, TypeError, KeyError) as e:
                self.original = None
                self.message = (f"{self.pending_path} (what the servos held before a previous run) is unreadable "
                                f"({e}): fix or delete it by hand, restart this tool, then recalibrate this arm")
        self.homings = None
        self.mins = self.maxes = self.positions = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._bus_lock = threading.Lock()      # one transaction at a time: the sampler, and every write

    # -- helpers
    def _read_servo_calibration(self) -> dict:
        out = {}
        for j in ARM_JOINTS:
            out[j] = {k: int(self.bus.read(reg, j, normalize=False, num_retry=2)) for k, reg in
                      (("homing_offset", "Homing_Offset"), ("range_min", "Min_Position_Limit"),
                       ("range_max", "Max_Position_Limit"))}
        return out

    def _write_servo_calibration(self, cal: dict):
        for j in ARM_JOINTS:
            c = cal[j]
            self.bus.write("Homing_Offset", j, int(c["homing_offset"]), normalize=False, num_retry=2)
            self.bus.write("Min_Position_Limit", j, int(c["range_min"]), normalize=False, num_retry=2)
            self.bus.write("Max_Position_Limit", j, int(c["range_max"]), normalize=False, num_retry=2)

    def _sample(self):
        ranged = [j for j in ARM_JOINTS if j != FULL_TURN]
        while not self._stop.is_set():
            try:
                with self._bus_lock:
                    if self._stop.is_set():
                        break
                    pos = self.bus.sync_read("Present_Position", ranged, normalize=False, num_retry=3)
            except Exception as e:                   # one bad read must not end the recording
                self.message = f"read: {e}"
                time.sleep(0.05)
                continue
            with self._lock:
                self.positions = {j: int(pos[j]) for j in ranged}
                self.mins = {j: min(self.mins.get(j, pos[j]), int(pos[j])) for j in ranged}
                self.maxes = {j: max(self.maxes.get(j, pos[j]), int(pos[j])) for j in ranged}
            time.sleep(0.02)

    # -- the steps
    def go_limp(self):
        if self.phase not in ("idle", "cancelled", "error"):
            raise ArmError(f"{self.name}: already {self.phase}")
        if self.touched:
            raise ArmError(f"{self.name}: the servos have not got their previous calibration back yet: Restore "
                           f"(Cancel) first, after checking the 12 V and the USB")
        self._connect()
        self.original = self._read_servo_calibration()
        self.bus.disable_torque(num_retry=2)
        for j in ARM_JOINTS:                         # lerobot sets position mode; write only what differs
            if int(self.bus.read("Operating_Mode", j, normalize=False, num_retry=2)) != 0:
                self.bus.write("Operating_Mode", j, 0, normalize=False, num_retry=2)
        self.phase, self.message = "limp", "torque off"

    def _connect(self):
        if self.bus is None:
            bus = self.bus_factory(self.port)
            bus.connect(handshake=True)
            self.bus = bus

    def _mark_touched(self):
        if self.pending_path:                    # on disk before the first servo write
            self.pending_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.pending_path.with_name(self.pending_path.name + ".tmp")
            tmp.write_text(json.dumps(self.original, indent=1))
            os.replace(tmp, self.pending_path)
        self.touched = True

    def _untouched(self):
        if self.pending_path:
            self.pending_path.unlink(missing_ok=True)
        self.touched = False

    def set_middle(self):
        if self.phase != "limp":
            raise ArmError(f"{self.name}: turn torque off first (now {self.phase})")
        self._mark_touched()                   # from the first write on: a failure part-way still needs a restore
        try:
            self.homings = {j: int(v) for j, v in self.bus.set_half_turn_homings().items()}
        except Exception as e:
            self.phase = "error"
            self._restore(f"homing failed ({e})")
            raise ArmError(f"{self.name}: {self.message}") from e
        self.mins, self.maxes, self.positions = {}, {}, {}
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, name=f"calib-{self.name}", daemon=True)
        self._thread.start()
        self.phase, self.message = "ranging", "recording: move every joint through its whole range"

    def snapshot(self) -> dict:
        with self._lock:
            return {"phase": self.phase, "message": self.message, "touched": self.touched, "positions": dict(self.positions or {}),
                    "mins": dict(self.mins or {}), "maxes": dict(self.maxes or {}),
                    "short": [j for j in ARM_JOINTS if j != FULL_TURN and (j not in (self.mins or {}) or
                              self.maxes[j] - self.mins[j] < MIN_RANGE_TICKS)]}

    def _stop_sampling(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)       # its reads are bounded (busio): a stuck one ends in seconds
            if self._thread.is_alive():
                raise ArmError(f"{self.name}: the position reader did not stop; nothing written")
            self._thread = None

    def _restore(self, why: str):
        """Give the servos back what they held before `set_middle`. On failure `touched` stays set: the next attempt
        refuses until a restore succeeds."""
        try:
            self._connect()                      # the link may have been closed after a failed attempt
            with self._bus_lock:
                self._write_servo_calibration(self.original)
            self._untouched()
            self.message = f"{why}: the servos have their previous calibration back (torque stays off)"
        except Exception as e:
            self.message = (f"{why}; restoring the previous calibration FAILED ({e}): check the 12 V and the USB, "
                            f"then Restore again; until then the arm's zero may be wrong")

    def finish(self) -> dict:
        if self.phase != "ranging":
            raise ArmError(f"{self.name}: nothing recorded (now {self.phase})")
        snap = self.snapshot()
        if snap["short"]:
            raise ArmError(f"{self.name}: move {', '.join(snap['short'])} through the whole range (each joint at "
                           f"least {MIN_RANGE_TICKS} ticks, about 26 deg); still recording")
        self._stop_sampling()
        ids = {j: i + 1 for i, j in enumerate(ARM_JOINTS)}
        cal = {}
        for j in ARM_JOINTS:
            lo, hi = (0, 4095) if j == FULL_TURN else (snap["mins"][j], snap["maxes"][j])
            cal[j] = {"id": ids[j], "drive_mode": 0, "homing_offset": self.homings[j], "range_min": int(lo),
                      "range_max": int(hi)}
        with self._bus_lock:
            if self.restore_after:
                self._write_servo_calibration(self.original)   # a trial: the servos keep what they had
            else:
                self._write_servo_calibration(cal)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.out_path.exists():
            shutil.copy2(self.out_path, self.out_path.with_name(
                f"{self.out_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
        tmp = self.out_path.with_name(self.out_path.name + ".tmp")
        tmp.write_text(json.dumps(cal, indent=4) + "\n")
        os.replace(tmp, self.out_path)
        self._untouched()                        # the servos hold the file's calibration (or, a trial, their own)
        self.phase = "done"
        self.message = (f"written {self.out_path}" + (" (trial: the servos got their previous calibration back)"
                                                       if self.restore_after else ""))
        return cal

    def cancel(self):
        self._stop_sampling()
        if self.touched and self.original is not None:
            self._restore("cancelled")
        elif self.touched:
            pass                                   # nothing known to restore: the message says what to do
        else:
            self.message = "cancelled (torque stays off)"
        self.homings = None
        self.phase = "cancelled" if not self.touched else "error"

    def close(self):
        if self.phase == "ranging" or (self.touched and self.bus is not None):   # what this run changed goes back
            self.cancel()
        self._stop_sampling()
        if self.bus is not None:
            try:
                self.bus.disconnect(disable_torque=False)       # it is off already; write nothing more
            except Exception:
                pass
            self.bus = None


# --------------------------------------------------------------------------------------------- progress
class State:
    """The tool's progress: {step: {"done": date, ...}}, in a JSON file beside the captures."""

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def mark(self, step: str, **info):
        self.data[step] = {"done": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
        self.save()

    def clear(self, step: str):
        self.data.pop(step, None)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)
