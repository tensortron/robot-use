"""Where the stack keeps what it records, outside the repository: robotd's event logs, the agent's programs and
episodes (code, traces, keyframes), and the simulated rig's own logs and episodes.

All beneath one folder: $ROBOT_USE_HOME if set, else ~/.tensortron/robot-use: agent_logs/, episodes/, sim/. Outside the agent's
working folder on purpose: the episodes are the record of what the agent did (and training data), so it must not be
able to rewrite them; it sees its own programs through its tools (get_program_trace, get_keyframe). A config file or
a command-line option that names a folder still wins. tools/setup_agent_ws.py denies the operating agent reads and edits of these
folders, wherever they are.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "ROBOT_USE_HOME"
DEFAULT = "~/.tensortron/robot-use"


def base() -> Path:
    """Absolute, so the agent's deny rules and every process agree on it (a relative value would resolve against
    each process's own folder)."""
    return Path(os.environ.get(ENV) or DEFAULT).expanduser().resolve()


def agent_logs() -> Path:
    return base() / "agent_logs"


def episodes() -> Path:
    return base() / "episodes"


def sim_home() -> Path:
    return base() / "sim"


def calibration() -> Path:
    """tools/calibrate.py: its captures, progress, trials and the backups of robot.yaml (which hold the camera pose)."""
    return base() / "calibration"


def all_dirs() -> list[Path]:
    return [agent_logs(), episodes(), sim_home(), calibration()]
