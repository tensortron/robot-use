"""Shared fixtures. Hermetic: limits are the measured calibration ranges (2026-09-28), not read from disk."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))

from stack.rig.kinematics import ArmKinematics, default_mount  # noqa: E402
from stack.robotd.safety import ArmLimits, ArmModel, SafetyConfig  # noqa: E402

CALIBRATED = {
    "left": {"shoulder_pan": (-122.5, 122.5), "shoulder_lift": (-103.9, 103.9), "elbow_flex": (-97.2, 97.2),
             "wrist_flex": (-102.8, 102.8), "wrist_roll": (-180.0, 180.0), "gripper": (0.0, 100.0)},
    "right": {"shoulder_pan": (-120.8, 120.8), "shoulder_lift": (-105.3, 105.3), "elbow_flex": (-97.7, 97.7),
              "wrist_flex": (-103.1, 103.1), "wrist_roll": (-180.0, 180.0), "gripper": (0.0, 100.0)},
}


def _hardware_extra_missing() -> str | None:
    import importlib.util
    missing = [m for m in ("lerobot", "scservo_sdk", "serial.tools") if importlib.util.find_spec(m.split(".")[0]) is None
               or importlib.util.find_spec(m) is None]
    return ", ".join(missing) or None


def pytest_collection_modifyitems(config, items):
    """Tests marked `hardware` need the hardware extra (lerobot, the Feetech SDK, pyserial); none opens a real port."""
    missing = _hardware_extra_missing()
    if missing:
        skip = pytest.mark.skip(reason=f"needs the hardware extra (missing: {missing})")
        for item in items:
            if item.get_closest_marker("hardware"):
                item.add_marker(skip)


@pytest.fixture(scope="session")
def cfg():
    return SafetyConfig()


@pytest.fixture(scope="session")
def models():
    return {a: ArmModel(a, ArmKinematics(default_mount(a)), ArmLimits(dict(CALIBRATED[a]))) for a in ("left", "right")}


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def reachable_sample(model, cfg, rng, n):
    """Random joint vectors inside the usable range whose tip is above the floor and in front of the base."""
    from stack.robotd.safety import _floor_violations, _workspace_violations, _keypoints

    lo, hi = model.limits.usable(cfg)
    out = []
    while len(out) < n:
        q = rng.uniform(lo, hi)
        q[5] = 50.0
        t = np.array([0.0])
        kp = _keypoints(model, q[None, :])
        if not _floor_violations(model, cfg, t, q[None, :], kp) and not _workspace_violations(model, cfg, t, kp):
            out.append(q)
    return np.array(out)
