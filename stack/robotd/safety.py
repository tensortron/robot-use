"""Level 1 of the safety layer: every path is checked here before the first tick moves an arm.

See docs/ARCHITECTURE.md, "What stops a bad motion". A path is a sampled joint trajectory
(times, 6 values per sample: five joints in lerobot degrees + gripper 0..100). Checks never clip:
they return ``Violation``s whose message names the rule, the value and the allowed range, so the
model that sent the command can correct it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path as FsPath

import numpy as np

from ..rig.kinematics import ARM_JOINTS, JOINTS, ArmKinematics

GRIPPER = "gripper"


@dataclass
class SafetyConfig:
    margin_deg: float = 5.0                  # usable range = calibrated range minus this, each end
    rest_margin_deg: float = 3.0             # go_rest only
    # The rest pose's wrist roll (sprint 2026-10-05, D9-a). 0 today; -90 once the wrist cable is checked at the
    # rig: in the model it keeps the folded jaws / camera 6+ mm from the arm's own shoulder (1.1 mm at 0, left arm).
    rest_roll_deg: float = 0.0
    # Gripper 0 = the calibration's closed end. No margin there: the jaws touch at 1-2 (rig 2026-10-11), so a goal
    # of 0 only presses them together (the servo's grip force grows with goal minus position). User 2026-10-11.
    gripper_usable: tuple[float, float] = (0.0, 98.0)
    rest_gripper: float = 2.0                # the rest pose's gripper: jaws touching, not pressing for hours at rest
    vmax: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 60.0, "shoulder_lift": 60.0, "elbow_flex": 60.0, "wrist_flex": 60.0,
        "wrist_roll": 90.0, "gripper": 150.0})             # deg/s, gripper %/s
    amax: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 200.0, "shoulder_lift": 200.0, "elbow_flex": 200.0, "wrist_flex": 200.0,
        "wrist_roll": 300.0, "gripper": 600.0})            # deg/s^2, gripper %/s^2
    tick_s: float = 0.02
    cap_tolerance: float = 1.01              # numerical slack on the speed/acceleration caps
    # Floors (rig frame): the arms stand on an 18 mm board placed on the work surface (the table, z = 0).
    # The board top is floor inside its outline, the work surface outside it. Unset outline values mean
    # "board everywhere" in that direction (conservative: it forbids the last 18 mm to the work surface).
    work_surface_z: float = 0.0
    board_top_z: float = 0.018
    board_front_x: float | None = None        # board front edge, m ahead of the pan-axis line
    board_half_width_y: float | None = None   # board side edges at +/- this, m from the midline
    floor_samples_per_link: int = 5           # points checked along each link, between its joints
    # Fixed obstacles: axis-aligned boxes in the rig frame, {"name", "min": [x, y, z], "max": [x, y, z]}.
    # Every link (as a capsule of capsule_radius_m) must stay obstacle_clearance_m clear of every box.
    obstacles: tuple = ()
    obstacle_clearance_m: float = 0.020
    obstacle_samples_per_link: int = 8
    tip_clearance_m: float = 0.020
    link_clearance_m: float = 0.020
    workspace_min: tuple[float, float, float] = (-0.15, -0.55, 0.0)
    workspace_max: tuple[float, float, float] = (0.55, 0.55, 0.45)
    arm_clearance_m: float = 0.040           # between link surfaces of the two arms
    capsule_radius_m: float = 0.025          # link thickness model (an STS3215 is 45 x 24 x 32 mm)
    start_tolerance_deg: float = 1.0         # a path must start where the arm is
    # Convex-hull checks of the gripper end (robotd/hulls.py, sprint 2026-10-05 decision D5), surface to surface.
    # floor: camera 15 mm = 5 mm + 10 mm for the ~1 cm elbow sag below the commanded pose; jaws 2 mm (grasps need
    # the fixed jaw near the table; a light touch is harmless; the tip rule above still applies); gripper 10 mm.
    hull_checks: bool = True
    hull_margins_m: dict = field(default_factory=lambda: {
        "floor": {"camera": 0.015, "jaws": 0.002, "gripper": 0.010},
        "obstacle": {"camera": 0.020, "jaws": 0.020, "gripper": 0.020},
        # wrist: enforced against the other arm only (sprint 2026-10-08), where it replaces the gripper-end capsules
        "other_arm": {"camera": 0.020, "jaws": 0.020, "gripper": 0.020, "wrist": 0.020},
        # the arm's own shoulder, base and upper arm (task 12): the -90 rest keeps 6+ mm, the old roll-0 rest 1.1 mm
        "own_arm": {"camera": 0.005, "jaws": 0.005, "gripper": 0.005}})

    def floor_z(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        on_board = np.ones(x.shape, bool)
        if self.board_front_x is not None:
            on_board &= x <= self.board_front_x
        if self.board_half_width_y is not None:
            on_board &= np.abs(y) <= self.board_half_width_y
        return np.where(on_board, self.board_top_z, self.work_surface_z)


@dataclass
class ArmLimits:
    """Calibrated (hard) ranges per joint, degrees; gripper 0..100."""

    hard: dict[str, tuple[float, float]]

    def usable(self, cfg: SafetyConfig, rest: bool = False) -> tuple[np.ndarray, np.ndarray]:
        m = cfg.rest_margin_deg if rest else cfg.margin_deg
        lo = [self.hard[j][0] + m for j in JOINTS] + [cfg.gripper_usable[0]]
        hi = [self.hard[j][1] - m for j in JOINTS] + [cfg.gripper_usable[1]]
        return np.array(lo), np.array(hi)


def load_arm_limits(arm_id: str, calibration_dir: FsPath | None = None) -> ArmLimits:
    """From the lerobot calibration file (e.g. arm_id 'bi_follower_left'); raises if unavailable."""
    from .calibration import read_joint_limits

    lim = read_joint_limits(arm_id, calibration_dir)
    missing = [j for j in ARM_JOINTS if not lim or j not in lim]
    if missing:
        raise RuntimeError(f"no calibrated limits for {arm_id}: missing {missing}; refusing to guess")
    return ArmLimits({j: lim[j] for j in ARM_JOINTS})


@dataclass
class ArmModel:
    """Everything the checks need about one arm."""

    name: str                                # "left" / "right"
    kin: ArmKinematics
    limits: ArmLimits


@dataclass
class Violation:
    rule: str
    arm: str
    joint: str | None
    t: float
    value: float
    allowed: str
    message: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "arm": self.arm, "joint": self.joint, "t": round(self.t, 3),
                "value": round(self.value, 4), "allowed": self.allowed, "message": self.message}


@dataclass
class Path:
    t: np.ndarray                            # (N,) seconds from start, t[0] == 0
    q: np.ndarray                            # (N, 6)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def at(self, t: float) -> np.ndarray:
        """Joint vector at time t (held at the ends), linear between samples."""
        if t <= self.t[0]:
            return self.q[0]
        if t >= self.t[-1]:
            return self.q[-1]
        return np.array([np.interp(t, self.t, self.q[:, i]) for i in range(self.q.shape[1])])


# --------------------------------------------------------------------------- rest pose
def rest_pose(model: ArmModel, cfg: SafetyConfig, wrist_pref_deg: float = 70.0) -> np.ndarray:
    """Folded rest derived from calibration (design doc, "Rest pose").

    pan 0, roll ``cfg.rest_roll_deg`` (0 or -90), lift 4 deg above its lower limit, elbow 4 deg below its upper
    limit, gripper ``cfg.rest_gripper`` (jaws touching, no press); wrist_flex is the value nearest ``wrist_pref_deg`` that keeps every link and the tip
    clear of the floor (70 deg, the recording plan's proposal, puts the tip into the board).
    """
    h = model.limits.hard
    lo, hi = model.limits.usable(cfg, rest=True)
    q = np.array([0.0, h["shoulder_lift"][0] + 4.0, h["elbow_flex"][1] - 4.0,
                  0.0, float(cfg.rest_roll_deg), cfg.rest_gripper])
    q[:5] = np.clip(q[:5], lo[:5], hi[:5])
    for w in sorted(np.arange(math.ceil(lo[3]), math.floor(hi[3]) + 1, 1.0), key=lambda v: abs(v - wrist_pref_deg)):
        q[3] = w
        if not _floor_violations(model, cfg, np.array([0.0]), q[None, :]):
            return q
    raise RuntimeError(f"{model.name}: no wrist_flex keeps the folded rest pose clear of the floor")


# --------------------------------------------------------------------------- checks
def _fmt_range(lo, hi, unit="deg"):
    return f"[{lo:.1f}, {hi:.1f}] {unit}"


def _unit(j):
    return "%" if j == GRIPPER else "deg"


def _range_violations(model, cfg, t, q, rest):
    out = []
    lo, hi = model.limits.usable(cfg, rest)
    for i, j in enumerate(ARM_JOINTS):
        # How far outside the usable range each sample is. A path may START outside it (an arm resting at the
        # edge, a gripper settling a hair under its minimum) as long as no later sample goes further out than
        # the start already was: moving back in, or holding, is always allowed; moving further out is not.
        excess = np.maximum(np.maximum(lo[i] - q[:, i], q[:, i] - hi[i]), 0.0)
        allowed = excess[0] + 1e-9 if len(excess) else 1e-9
        bad = np.nonzero(excess > allowed)[0]
        if len(bad):
            k = bad[0]
            hard = model.limits.hard[j]
            out.append(Violation("joint_range", model.name, j, float(t[k]), float(q[k, i]), _fmt_range(lo[i], hi[i], _unit(j)),
                                 f"{model.name}.{j} {'target' if k == len(q) - 1 else 'path'} {q[k, i]:.1f} outside usable range "
                                 f"{_fmt_range(lo[i], hi[i], _unit(j))}"
                                 + ("" if j == GRIPPER else f" (hard limit {hard[0]:.1f}..{hard[1]:.1f}, margin "
                                    f"{cfg.rest_margin_deg if rest else cfg.margin_deg:.1f})")))
    return out


def _rate_violations(model, cfg, t, q):
    out = []
    if len(t) < 2:
        return out
    dt = np.diff(t)
    v = np.diff(q, axis=0) / dt[:, None]
    for i, j in enumerate(ARM_JOINTS):
        cap = cfg.vmax[j] * cfg.cap_tolerance
        bad = np.nonzero(np.abs(v[:, i]) > cap)[0]
        if len(bad):
            k = bad[0]
            out.append(Violation("speed", model.name, j, float(t[k]), float(abs(v[k, i])), f"<= {cfg.vmax[j]:.0f} {_unit(j)}/s",
                                 f"{model.name}.{j} speed {abs(v[k, i]):.0f} {_unit(j)}/s at t={t[k]:.2f}s exceeds the cap "
                                 f"{cfg.vmax[j]:.0f} {_unit(j)}/s; slow the trajectory down or use time_scale='auto'"))
    if len(t) < 3:
        return out
    a = np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
    for i, j in enumerate(ARM_JOINTS):
        cap = cfg.amax[j] * cfg.cap_tolerance
        bad = np.nonzero(np.abs(a[:, i]) > cap)[0]
        if len(bad):
            k = bad[0]
            out.append(Violation("acceleration", model.name, j, float(t[k + 1]), float(abs(a[k, i])),
                                 f"<= {cfg.amax[j]:.0f} {_unit(j)}/s^2",
                                 f"{model.name}.{j} acceleration {abs(a[k, i]):.0f} {_unit(j)}/s^2 at t={t[k + 1]:.2f}s exceeds "
                                 f"the cap {cfg.amax[j]:.0f}; space the waypoints further apart in time or use time_scale='auto'"))
    return out


KEYPOINT_NAMES = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "tip")


def _keypoints(model, q):
    """(N, 5, 3) moving keypoints (lift, elbow, wrist_flex, wrist_roll origins, tip); pan origin is fixed."""
    return np.array([model.kin.keypoints(qq)[1:] for qq in q])


def _floor_points(kp, n):
    """Keypoints plus ``n`` points along each link between them: (N, P, 3) points and their names.

    Links: lift->elbow (upper arm), elbow->wrist_flex (forearm), wrist_flex->wrist_roll, wrist_roll->tip.
    Checking only the joints would let a forearm crossing the board's front corner dip into it.
    """
    names, pts = [], []
    for i, nm in enumerate(KEYPOINT_NAMES):
        pts.append(kp[:, i]); names.append(nm)
    links = (("upper arm", 0, 1), ("forearm", 1, 2), ("wrist", 2, 3), ("gripper", 3, 4))
    for nm, a, b in links:
        for f in np.linspace(0, 1, n + 2)[1:-1]:
            pts.append(kp[:, a] + f * (kp[:, b] - kp[:, a])); names.append(nm)
    return np.stack(pts, axis=1), names


def _floor_violations(model, cfg, t, q, kp=None):
    kp = _keypoints(model, q) if kp is None else kp
    pts, names = _floor_points(kp, cfg.floor_samples_per_link)
    floor = cfg.floor_z(pts[..., 0], pts[..., 1])
    need = np.array([cfg.tip_clearance_m if nm == "tip" else cfg.link_clearance_m for nm in names])
    gap = pts[..., 2] - floor - need
    # A path may START closer to the floor than the clearance (the folded arms rest with their jaws on the
    # table, tip ~7 mm up): on the way, no point may get any closer than it started, and the path must END
    # fully clear. So the arm can lift off from rest, never go further down, and never finish below the
    # clearance. A single pose (one sample) is checked absolutely.
    allowed = np.repeat(np.minimum(gap[:1], 0.0), len(gap), axis=0)
    allowed[-1] = 0.0
    bad = np.argwhere(gap < allowed - 1e-6)
    if not len(bad):
        return []
    k = int(bad[0][0])
    p = int(np.argmin(gap[k]))
    z_above = pts[k, p, 2] - floor[k, p]
    surface = "board top" if floor[k, p] > cfg.work_surface_z else "work surface"
    return [Violation("floor", model.name, None, float(t[k]), float(z_above), f">= {need[p] * 1000:.0f} mm above the {surface}",
                      f"{model.name} {names[p]} would be {z_above * 1000:.0f} mm above the {surface} at t={t[k]:.2f}s "
                      f"(at x={pts[k, p, 0]:.3f}, y={pts[k, p, 1]:.3f}); minimum {need[p] * 1000:.0f} mm")]


OBSTACLE_LINKS = (("shoulder", 0, 1), ("upper arm", 1, 2), ("forearm", 2, 3), ("wrist", 3, 4), ("gripper", 4, 5))


def _obstacle_violations(model, cfg, t, kp_full):
    """Links (capsules) against the fixed obstacle boxes. ``kp_full``: (N, 6, 3) incl. the pan origin."""
    if not cfg.obstacles:
        return []
    pts, names = [], []
    for nm, a, b in OBSTACLE_LINKS:
        for f in np.linspace(0, 1, cfg.obstacle_samples_per_link + 2):
            pts.append(kp_full[:, a] + f * (kp_full[:, b] - kp_full[:, a])); names.append(nm)
    pts = np.stack(pts, axis=1)                                         # (N, P, 3)
    need = cfg.capsule_radius_m + cfg.obstacle_clearance_m
    for ob in cfg.obstacles:
        lo, hi = np.asarray(ob["min"], float), np.asarray(ob["max"], float)
        d = np.linalg.norm(np.maximum(np.maximum(lo - pts, 0.0), pts - hi), axis=-1)   # point-to-box distance
        bad = np.argwhere(d < need - 1e-9)
        if len(bad):
            k = int(bad[0][0])
            p = int(np.argmin(d[k]))
            gap = d[k, p] - cfg.capsule_radius_m
            return [Violation("obstacle", model.name, None, float(t[k]), float(gap),
                              f">= {cfg.obstacle_clearance_m * 1000:.0f} mm from {ob['name']}",
                              f"{model.name} {names[p]} would come within {max(gap, 0) * 1000:.0f} mm of the "
                              f"{ob['name']} at t={t[k]:.2f}s (at x={pts[k, p, 0]:.3f}, y={pts[k, p, 1]:.3f}, "
                              f"z={pts[k, p, 2]:.3f}); minimum {cfg.obstacle_clearance_m * 1000:.0f} mm")]
    return []


def _workspace_violations(model, cfg, t, kp):
    tip = kp[:, -1, :]
    lo, hi = np.array(cfg.workspace_min), np.array(cfg.workspace_max)
    bad = np.argwhere((tip < lo - 1e-9) | (tip > hi + 1e-9))
    if not len(bad):
        return []
    k, ax = bad[0]
    name = "xyz"[ax]
    return [Violation("workspace", model.name, None, float(t[k]), float(tip[k, ax]),
                      f"{name} in [{lo[ax]:.2f}, {hi[ax]:.2f}] m",
                      f"{model.name} gripper tip {name}={tip[k, ax]:.3f} m at t={t[k]:.2f}s is outside the workspace box "
                      f"{name} in [{lo[ax]:.2f}, {hi[ax]:.2f}] m")]


def _segments(model, q):
    """(N, 4, 2, 3) link segments: pan->lift, lift->elbow, elbow->wrist_flex, wrist_flex->tip."""
    pts = np.array([model.kin.keypoints(qq) for qq in q])            # (N, 6, 3)
    idx = [(0, 1), (1, 2), (2, 3), (3, 5)]
    return np.stack([np.stack([pts[:, a], pts[:, b]], axis=1) for a, b in idx], axis=1)


def segment_distance(p1, q1, p2, q2):
    """Vectorised closest distance between segments p1-q1 and p2-q2 (arrays (..., 3))."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    denom = a * e - b * b
    eps = 1e-12
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, eps)
    s = np.where(t < 0, np.clip(-c / np.maximum(a, eps), 0, 1), np.where(t > 1, np.clip((b - c) / np.maximum(a, eps), 0, 1), s))
    t = np.clip(t, 0, 1)
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


SEGMENT_NAMES = ("shoulder", "upper_arm", "forearm", "wrist+gripper")


GRIP = SEGMENT_NAMES.index("wrist+gripper")


def _clearance_violations(models, cfg, t, qs, hulls: bool = False):
    """qs: {arm: (N, 6)} on the same timeline, for both arms."""
    (na, ma), (nb, mb) = list(models.items())
    return _clearance_from_segments(na, cfg, t, _segments(ma, qs[na]), _segments(mb, qs[nb]), hulls)


def _clearance_from_segments(na, cfg, t, sa, sb, hulls: bool = False):
    """sa, sb: (N, 4, 2, 3) link segments of the first and second arm on the timeline t.

    hulls: the gripper-end hull checks run on these paths too (robotd/hulls.py). The two arms' wrist+gripper
    capsules are then not compared: the hulls check that pair exactly (camera, jaws, gripper and wrist against the
    other arm's), and the 25 mm capsules with their 40 mm gap kept the two wrist-to-tip lines 90 mm apart - more
    than two-arm work needs (sprint 2026-10-08). Every other pair keeps its capsules, also at launch, where the hull
    check only looks where the two wrist+gripper segments are near each other."""
    A = sa[:, :, None]                                               # (N, 4, 1, 2, 3)
    B = sb[:, None, :]                                               # (N, 1, 4, 2, 3)
    d = segment_distance(A[..., 0, :], A[..., 1, :], B[..., 0, :], B[..., 1, :])   # (N, 4, 4)
    gap = d - 2 * cfg.capsule_radius_m
    if hulls:
        gap[:, GRIP, GRIP] = np.inf
    k, i, j = np.unravel_index(np.argmin(gap), gap.shape)
    if gap[k, i, j] >= cfg.arm_clearance_m - 1e-9:
        return []
    return [Violation("arm_clearance", "both", None, float(t[k]), float(gap[k, i, j]),
                      f">= {cfg.arm_clearance_m * 1000:.0f} mm",
                      f"left {SEGMENT_NAMES[i] if na == 'left' else SEGMENT_NAMES[j]} and right "
                      f"{SEGMENT_NAMES[j] if na == 'left' else SEGMENT_NAMES[i]} would come within "
                      f"{max(gap[k, i, j], 0) * 1000:.0f} mm at t={t[k]:.2f}s; minimum {cfg.arm_clearance_m * 1000:.0f} mm")]


def check_paths(models: dict[str, ArmModel], cfg: SafetyConfig, paths: dict[str, Path],
                current: dict[str, np.ndarray], rest: bool = False,
                others: dict[str, Path] | None = None, hulls: bool = False) -> list[Violation]:
    """Check planned paths for some arms; arms without a path hold ``current``, unless they are moving:
    ``others`` gives such an arm's remaining path (t = 0 is now), used for the arm-to-arm clearance only
    (it was checked when it was planned). ``hulls``: the caller also runs the gripper-end hull checks on these
    paths (see _clearance_from_segments); leave False when it does not.

    Returns every distinct violation (first occurrence per rule/arm/joint), empty if the motion is safe.
    """
    out: list[Violation] = []
    for arm, p in paths.items():
        m = models[arm]
        if not np.all(np.isfinite(p.q)) or not np.all(np.isfinite(p.t)):
            return [Violation("not_finite", arm, None, 0.0, float("nan"), "finite numbers",
                              f"{arm}: the path contains NaN or infinite values")]
        if p.q.ndim != 2 or p.q.shape[1] != len(ARM_JOINTS) or len(p.t) != len(p.q) or len(p.t) == 0:
            return [Violation("malformed", arm, None, 0.0, 0.0, f"(N, {len(ARM_JOINTS)})", f"{arm}: malformed path")]
        jump = np.abs(p.q[0] - current[arm])
        if np.max(jump) > cfg.start_tolerance_deg:
            j = ARM_JOINTS[int(np.argmax(jump))]
            out.append(Violation("start", arm, j, 0.0, float(np.max(jump)), f"<= {cfg.start_tolerance_deg} deg",
                                 f"{arm}.{j}: path starts {np.max(jump):.1f} away from the arm's current position"))
        out += _range_violations(m, cfg, p.t, p.q, rest)
        out += _rate_violations(m, cfg, p.t, p.q)
        kp_full = np.array([m.kin.keypoints(qq) for qq in p.q])
        kp = kp_full[:, 1:]
        out += _floor_violations(m, cfg, p.t, p.q, kp)
        out += _obstacle_violations(m, cfg, p.t, kp_full)
        out += _workspace_violations(m, cfg, p.t, kp)
    out += check_clearance(models, cfg, paths, current, others, hulls=hulls)
    return out


def check_clearance(models: dict[str, ArmModel], cfg: SafetyConfig, paths: dict[str, Path],
                    current: dict[str, np.ndarray], others: dict[str, Path] | None = None,
                    window_s: float = 0.0, hulls: bool = False) -> list[Violation]:
    """The arm-to-arm clearance part of check_paths alone (robotd re-runs it at launch).

    ``window_s``: a moving other arm (``others``) may be anywhere from 0 to ``window_s`` further along its path
    than t = 0 says (robotd launches a little after it checks); every tick-sized shift in that window is
    checked. Kinematics run once per sample (the cost), the shifted distances are cheap numpy.
    """
    if len(models) != 2:
        return []
    others = {a: p for a, p in (others or {}).items() if a not in paths}
    tick = cfg.tick_s
    T = max([p.duration for p in paths.values()] + [p.duration for p in others.values()], default=0.0)
    n = int(math.ceil(T / tick)) + 1
    grid = np.arange(n) * tick
    K = int(math.ceil(window_s / tick)) if others else 0
    (na, ma), (nb, mb) = list(models.items())
    segs = {}
    for a, m in ((na, ma), (nb, mb)):
        src = paths.get(a) or others.get(a)
        if src is None:
            q = np.repeat(np.asarray(current[a], float)[None, :], 1, 0)
            segs[a] = np.repeat(_segments(m, q), n + K, 0)
        else:
            tt = np.arange(n + K) * tick if a in others else grid
            segs[a] = _segments(m, np.array([src.at(x) for x in tt]))
    worst = None
    for k in range(K + 1):
        sa = segs[na][k:k + n] if na in others else segs[na][:n]
        sb = segs[nb][k:k + n] if nb in others else segs[nb][:n]
        v = _clearance_from_segments(na, cfg, grid, sa, sb, hulls)
        if v and (worst is None or v[0].value < worst[0].value):
            worst = v
    return worst or []
