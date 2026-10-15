"""lerobot calibration files, read without importing lerobot (so the stack runs without the hardware extra).

The file is lerobot's: ``$HF_LEROBOT_CALIBRATION/robots/so_follower/<id>.json``, where the directory defaults as in
``lerobot.utils.constants`` (``$HF_LEROBOT_HOME/calibration``, ``$HF_LEROBOT_HOME`` = ``$HF_HOME/lerobot``,
``$HF_HOME`` = ``$XDG_CACHE_HOME/huggingface`` or ``~/.cache/huggingface``).

Joint limits mirror ``MotorsBus._normalize`` for DEGREES mode: ``(raw - mid) * 360 / (resolution - 1)``, so the
calibrated travel is symmetric about zero, +/- (range_max - range_min) / 2 * 360 / 4095 for the sts3215. The gripper
is RANGE_0_100: 0..100 by definition. (The same rule as webslider's ``limits.read_joint_limits``.)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

STS3215_MAX_RES = 4096 - 1
GRIPPER = "gripper"


def calibration_dir() -> Path:
    """lerobot's HF_LEROBOT_CALIBRATION, resolved by the same rules (an empty variable counts as set, as there)."""
    cal = os.getenv("HF_LEROBOT_CALIBRATION")
    if cal is not None:
        return Path(cal).expanduser()
    home = os.getenv("HF_LEROBOT_HOME")
    if home is None:
        hf = os.getenv("HF_HOME")
        if hf is None:                                            # huggingface_hub.constants
            hf = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface")
        home = os.path.join(os.path.expandvars(os.path.expanduser(hf)), "lerobot")
    return Path(home).expanduser() / "calibration"


def calibration_path(arm_id: str, directory: Path | None = None) -> Path:
    if directory is not None:
        return Path(directory) / f"{arm_id}.json"
    return calibration_dir() / "robots" / "so_follower" / f"{arm_id}.json"


def read_joint_limits(arm_id: str, directory: Path | None = None) -> dict[str, tuple[float, float]] | None:
    """``{joint: (min, max)}`` in degrees (gripper 0..100), or None if the file is missing or unreadable."""
    path = calibration_path(arm_id, directory)
    if not path.is_file():
        logger.warning("no calibration at %s", path)
        return None
    try:
        cal = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("could not parse %s (%s)", path, e)
        return None
    limits: dict[str, tuple[float, float]] = {}
    for joint, entry in cal.items():
        try:
            lo, hi = float(entry["range_min"]), float(entry["range_max"])
        except (KeyError, TypeError, ValueError):
            continue
        if hi <= lo:
            continue
        if joint == GRIPPER:
            limits[joint] = (0.0, 100.0)
        else:
            half = (hi - lo) / 2 * 360 / STS3215_MAX_RES
            limits[joint] = (-half, half)
    return limits or None
