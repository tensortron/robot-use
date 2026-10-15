"""Scene files for the simulated rig (`scenes/*.yaml`): which objects are on the table, and where they may be.

A file names its objects in the twin's object format (`SceneObject`: name, shape, size, colour, mass) and gives each
one either a fixed `xy` or a range (`x: [lo, hi]`, `y: [lo, hi]`) in the rig frame; `yaw_deg` is fixed or a range.
`load_scene_file(path, seed)` draws one layout from a seed: the same seed gives the same layout. Objects stand on
the table (z = 0), off the board, with at least `min_gap_m` between their footprints (circles round each object).
A layout that cannot be drawn in `MAX_TRIES` is an error, not a silently crowded scene. Placement is greedy, in file
order (no backtracking), and keeps clear of robot.yaml's board, so a layout depends on both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .scene import CUP_SEGMENTS, SCENES_DIR, SceneObject

MAX_TRIES = 2000
OBJECT_KEYS = {"name", "shape", "size", "xy", "x", "y", "yaw_deg", "mass", "rgba", "friction", "wall_m"}
FILE_KEYS = {"name", "description", "min_gap_m", "objects"}


@dataclass
class Layout:
    name: str
    seed: int
    objects: list                            # [SceneObject], placed


def scene_path(name_or_path) -> Path:
    """A scene by name (`block_cup`, from `scenes/`) or by the path of a .yaml file."""
    p = Path(name_or_path).expanduser()
    return p if p.suffix in (".yaml", ".yml") else SCENES_DIR / f"{name_or_path}.yaml"


def footprint_radius(o: SceneObject) -> float:
    """The radius of a circle round the object's footprint, whatever its yaw (a cup's wall boxes reach slightly
    past its nominal radius, at the polygon's corners)."""
    if o.shape == "cup":
        return o.size[0] / 2 / math.cos(math.pi / CUP_SEGMENTS)
    return math.hypot(o.size[0], o.size[1]) / 2


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what}: expected a number, got {v!r}")
    return float(v)


def _range(spec, what: str) -> tuple[float, float]:
    """A number (fixed) or [lo, hi]."""
    if not isinstance(spec, (list, tuple)):
        v = _num(spec, what)
        return v, v
    if len(spec) != 2:
        raise ValueError(f"{what}: expected a number or [lo, hi], got {spec!r}")
    lo, hi = (_num(v, what) for v in spec)
    if hi < lo:
        raise ValueError(f"{what}: range [{lo}, {hi}] is empty")
    return lo, hi


def load_scene_file(name_or_path, seed: int | None = None, board_front_x: float | None = None) -> Layout:
    """Draw a layout. `seed` None picks one (reported in Layout.seed, so a run can be repeated). Objects keep clear
    of the board's front edge (`board_front_x`, default from robot.yaml)."""
    path = scene_path(name_or_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("objects", []), list):
        raise ValueError(f"{path.name}: expected a mapping with an `objects` list")
    if unknown := set(raw) - FILE_KEYS:
        raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
    if board_front_x is None:
        from stack.rig.rig import load_rig
        board_front_x = load_rig().board.front_x_m
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**31)
    rng = np.random.default_rng(seed)
    gap = _num(raw.get("min_gap_m", 0.03), f"{path.name}: min_gap_m")
    names, placed = set(), []
    for d in raw.get("objects") or []:
        if not isinstance(d, dict) or "name" not in d:
            raise ValueError(f"{path.name}: each object is a mapping with a name, got {d!r}")
        if unknown := set(d) - OBJECT_KEYS:
            raise ValueError(f"{path.name}: object {d.get('name')!r}: unknown keys {sorted(unknown)}")
        name = str(d["name"])
        if name in names:
            raise ValueError(f"{path.name}: two objects named {name!r}")
        names.add(name)
        if "xy" in d and ("x" in d or "y" in d):
            raise ValueError(f"{path.name}: object {name!r}: give xy or x/y ranges, not both")
        if "xy" in d:
            if not isinstance(d["xy"], (list, tuple)) or len(d["xy"]) != 2:
                raise ValueError(f"{path.name}: object {name!r}: xy is [x, y], got {d['xy']!r}")
            xr, yr = _range(d["xy"][0], f"{name} x"), _range(d["xy"][1], f"{name} y")
        elif "x" not in d or "y" not in d:
            raise ValueError(f"{path.name}: object {name!r}: give xy, or x and y ranges")
        else:
            xr, yr = _range(d["x"], f"{name} x"), _range(d["y"], f"{name} y")
        yawr = _range(d.get("yaw_deg", 0.0), f"{name} yaw_deg")
        proto = SceneObject.from_dict({**{k: v for k, v in d.items() if k not in ("x", "y", "xy", "yaw_deg")},
                                       "xy": [0, 0]})
        r = footprint_radius(proto)
        for _ in range(MAX_TRIES):
            x, y = rng.uniform(*xr), rng.uniform(*yr)
            if x - r < board_front_x + gap:
                continue
            if all(math.hypot(x - o.xyz[0], y - o.xyz[1]) >= r + footprint_radius(o) + gap for o in placed):
                break
        else:
            raise ValueError(f"{path.name}: could not place {name!r} clear of the board and the other objects "
                             f"in {MAX_TRIES} tries (seed {seed}): widen its range or lower min_gap_m")
        yaw = rng.uniform(*yawr)
        placed.append(SceneObject(name, proto.size, (float(x), float(y), proto.size[2] / 2), float(yaw), proto.mass,
                                  proto.rgba, proto.friction, proto.shape, proto.wall_m))
    return Layout(str(raw.get("name", path.stem)), seed, placed)
