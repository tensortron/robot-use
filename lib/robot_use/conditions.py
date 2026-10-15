"""Step conditions ("move until ..."), evaluated by robotd every 20 ms tick (docs/ARCHITECTURE.md).

Pass one, or a list (any of them), as `until=` to a motion. Meeting one ends the step with status
`condition_met` and `result.condition_met` set; a touch holds where it touched, the others decelerate. If the
motion ends without meeting it, the step is `done` with `result.condition_met` None.
"""

from __future__ import annotations


def contact(level: str = "light", joints=None) -> dict:
    """The arm touches something. level "light" (first touch) or "firm"; joints limits which joints count."""
    return {"type": "contact", "level": level, **({"joints": list(joints)} if joints else {})}


def gripper_stalled(effort: float = 0.3) -> dict:
    """The gripper meets resistance at `effort` (0..1) of its full grip current."""
    return {"type": "gripper_stalled", "effort": float(effort)}


def joint_crosses(joint: str, value: float, direction: int) -> dict:
    """A joint passes `value` (degrees, or 0..100 for the gripper) moving in `direction` (+1 or -1)."""
    return {"type": "joint_crosses", "joint": joint, "value": float(value), "direction": int(direction)}


def elapsed(seconds: float) -> dict:
    """The step has run for `seconds`."""
    return {"type": "time", "s": float(seconds)}
