"""Estimates: positions and sizes the program believes, with how far it trusts them (docs/ARCHITECTURE.md).

A pick-and-place program is a chain of estimates (where the block is, how tall, where the cup is), first
read off the ego image to a few centimetres, then tightened as the arm closes in. An Estimate carries its
tolerance and where each value came from, and checkpoints can refine it within that tolerance.

    block = estimate("red block", xy=geo.pixel_to_table(612, 401, z=0.04)[:2], size=(0.04, 0.04, 0.04),
                     tol=0.03, source="vlm/ego pixel (612, 401)")
    block.top          -> [x, y, height]            the top centre
    block.centre       -> [x, y, height / 2]
    block.above(0.08)  -> [x, y, height + 0.08]     a hover point
    block.at(z)        -> [x, y, z]
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Estimate:
    name: str
    xy: list
    size: tuple = (0.0, 0.0, 0.0)            # x, y, z extent in metres; z is the height above the table
    tol: float = 0.03                        # how far (m) a checkpoint may move it without re-approval
    source: str = "unspecified"
    base_z: float = 0.0                      # the surface it rests on (the table is 0)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.xy = [float(v) for v in self.xy][:2]
        if len(self.xy) != 2:
            raise ValueError("xy needs two values")
        self.size = tuple(float(v) for v in self.size)
        self.history.append({"xy": list(self.xy), "source": self.source})

    @property
    def height(self) -> float:
        return self.size[2] if len(self.size) > 2 else 0.0

    @property
    def top(self) -> list:
        return [self.xy[0], self.xy[1], self.base_z + self.height]

    @property
    def centre(self) -> list:
        return [self.xy[0], self.xy[1], self.base_z + self.height / 2]

    def above(self, dz: float) -> list:
        return [self.xy[0], self.xy[1], self.base_z + self.height + float(dz)]

    def at(self, z: float) -> list:
        return [self.xy[0], self.xy[1], float(z)]

    def refine(self, xy=None, dxy=None, source: str = "refined"):
        """Move the estimate (absolute xy, or by dxy). Checkpoint refinements are bounded by the runner."""
        if xy is not None:
            self.xy = [float(v) for v in xy][:2]
        elif dxy is not None:
            self.xy = [self.xy[0] + float(dxy[0]), self.xy[1] + float(dxy[1])]
        self.source = source
        self.history.append({"xy": list(self.xy), "source": source})
        return self

    def as_checkpoint(self) -> dict:
        return {"value": list(self.xy), "tol": self.tol, "source": self.source, "size": list(self.size)}


def estimate(name: str, xy, size=(0.0, 0.0, 0.0), tol: float = 0.03, source: str = "unspecified",
             base_z: float = 0.0) -> Estimate:
    return Estimate(name, list(xy), tuple(size), float(tol), source, float(base_z))
