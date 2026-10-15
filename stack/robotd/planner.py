"""Turn goals into sampled joint paths under the speed and acceleration caps.

The planner decides *how fast*; ``safety.check_paths`` decides *whether*. Every path produced here
is still checked before it runs (design doc: "Large moves are planned, not rejected").

- ``plan_joint_move``: rest-to-rest, all joints synchronised on one trapezoidal speed profile.
- ``plan_waypoints``: a timed trajectory through waypoints (clamped cubic spline, rest to rest);
  ``time_scale="auto"`` stretches time until the caps hold, ``"reject"`` leaves it to the checks.
- ``plan_linear``: the gripper tip on a straight line, IK every tick, then retimed to the joint caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..rig.kinematics import ARM_JOINTS
from .safety import ArmModel, Path, SafetyConfig


class PlanError(ValueError):
    """The goal cannot be turned into a path (unreachable pose, bad waypoint times...)."""


def _trapezoid(T_total_min: float, vs: float, as_: float) -> tuple[float, float]:
    """Duration and cruise speed of a rest-to-rest profile over s in [0, 1]."""
    if vs * vs / as_ >= 1.0:          # never reaches cruise: triangular
        return 2.0 * math.sqrt(1.0 / as_), math.sqrt(as_)
    return 1.0 / vs + vs / as_, vs


def _s_of_t(t: np.ndarray, T: float, vc: float, as_: float) -> np.ndarray:
    ta = vc / as_
    s = np.where(t < ta, 0.5 * as_ * t * t,
                 np.where(t <= T - ta, 0.5 * as_ * ta * ta + vc * (t - ta),
                          1.0 - 0.5 * as_ * (T - t) ** 2))
    return np.clip(s, 0.0, 1.0)


def _grid(T: float, tick: float) -> np.ndarray:
    n = max(int(math.ceil(T / tick - 1e-9)), 1)
    return np.linspace(0.0, T, n + 1)


def _speed(speed: float) -> float:
    if not (0.0 < speed <= 1.0) or not math.isfinite(speed):
        raise PlanError(f"speed must be in (0, 1], got {speed}")
    return speed


def plan_joint_move(q0, q1, cfg: SafetyConfig, speed: float = 0.5) -> Path:
    """All six values move together and finish together, each under speed x its cap."""
    speed = _speed(speed)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = q1 - q0
    moving = np.abs(d) > 1e-9
    if not np.any(moving):
        return Path(np.array([0.0]), q0[None, :].copy())
    vmax = np.array([cfg.vmax[j] for j in ARM_JOINTS]) * speed
    amax = np.array([cfg.amax[j] for j in ARM_JOINTS]) * speed
    vs = float(np.min(vmax[moving] / np.abs(d[moving])))
    as_ = float(np.min(amax[moving] / np.abs(d[moving])))
    T, vc = _trapezoid(0.0, vs, as_)
    t = _grid(T, cfg.tick_s)
    s = _s_of_t(t, T, vc, as_)
    return Path(t, q0[None, :] + s[:, None] * d[None, :])


def plan_waypoints(q0, waypoints: list[tuple[float, np.ndarray]], cfg: SafetyConfig,
                   time_scale: str = "reject", max_scale: float = 20.0) -> tuple[Path, float]:
    """Clamped cubic spline through (t, q) waypoints, starting at rest at q0. Returns (path, scale)."""
    from scipy.interpolate import CubicSpline

    if time_scale not in ("reject", "auto"):
        raise PlanError(f"time_scale must be 'reject' or 'auto', got {time_scale!r}")
    if not waypoints:
        raise PlanError("no waypoints")
    ts = np.array([0.0] + [float(t) for t, _ in waypoints])
    if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
        raise PlanError("waypoint times must be finite, > 0 and strictly increasing")
    qs = np.vstack([np.asarray(q0, float)] + [np.asarray(q, float) for _, q in waypoints])
    if not np.all(np.isfinite(qs)):
        raise PlanError("waypoints contain NaN or infinite values")
    vmax = np.array([cfg.vmax[j] for j in ARM_JOINTS])
    amax = np.array([cfg.amax[j] for j in ARM_JOINTS])
    scale = 1.0
    for _ in range(8):
        spline = CubicSpline(ts * scale, qs, axis=0, bc_type="clamped")
        t = _grid(ts[-1] * scale, cfg.tick_s)
        path = Path(t, spline(t))
        if time_scale == "reject":
            return path, scale
        v = np.max(np.abs(spline(t, 1)), axis=0) / vmax
        a = np.max(np.abs(spline(t, 2)), axis=0) / amax
        need = max(float(np.max(v)), math.sqrt(float(np.max(a))))
        if need <= 0.98:
            return path, scale
        scale *= need * 1.03
        if scale > max_scale:
            raise PlanError(f"trajectory would need to be slowed down more than {max_scale:.0f}x to fit the caps")
    return path, scale


@dataclass
class LinearPlan:
    path: Path
    ik_failures: int


def plan_linear(model: ArmModel, cfg: SafetyConfig, q0, xyz, pitch_deg: float | None = None,
                roll_deg: float | None = None, speed: float = 0.5, cart_vmax: float = 0.15,
                cart_amax: float = 0.5, rest: bool = False) -> Path:
    """Tip along a straight line to xyz; pitch and roll interpolate from the current values.

    Geometry first: the line is sampled uniformly (every ~2 mm / 1 deg of pitch or roll) and solved with IK
    seeded by the previous sample, so the arm never switches branch mid-path; an unsolvable sample, or a jump
    between samples, rejects the move. Then timing along the path: at every sample the speed along the line
    is limited by the Cartesian cap, each joint's speed cap (joint rate = dq/ds x ds/dt) and, where the joint
    path bends sharply (dq/ds changes fast, e.g. near a joint limit), by the acceleration that bend needs; a
    forward and a backward pass add the acceleration caps. So the arm slows only where it has to.

    (Review 2026-10-03: the old uniform retime slowed a 7 cm lift near the wrist limit to 198 s, because one
    sharp bend at the end set the pace for the whole line.) A line that would still take far longer than
    its Cartesian length needs is rejected with the reason, rather than crawled.
    """
    speed = _speed(speed)
    q0 = np.asarray(q0, float)
    start = model.kin.pose(q0)
    p0, p1 = start["xyz"], np.asarray(xyz, float)
    pitch1 = start["pitch"] if pitch_deg is None else float(pitch_deg)
    roll1 = start["roll"] if roll_deg is None else float(roll_deg)
    dpitch = math.degrees(math.remainder(math.radians(pitch1 - start["pitch"]), math.tau))
    droll = roll1 - start["roll"]
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-9 and abs(dpitch) < 1e-9 and abs(droll) < 1e-9:
        return Path(np.array([0.0]), q0[None, :].copy())
    # caps on the path parameter s in [0, 1] from the Cartesian and wrist (pitch / roll) caps
    vs, as_ = math.inf, math.inf
    if L > 1e-9:
        vs, as_ = min(vs, cart_vmax * speed / L), min(as_, cart_amax * speed / L)
    if abs(dpitch) > 1e-9:
        vs, as_ = min(vs, cfg.vmax["wrist_flex"] * speed / abs(dpitch)), min(as_, cfg.amax["wrist_flex"] * speed / abs(dpitch))
    if abs(droll) > 1e-9:
        vs, as_ = min(vs, cfg.vmax["wrist_roll"] * speed / abs(droll)), min(as_, cfg.amax["wrist_roll"] * speed / abs(droll))
    T_nominal, _ = _trapezoid(0.0, vs, as_)

    # ---- geometry, uniform in s
    n = int(max(30, math.ceil(L / 0.002), math.ceil(abs(dpitch)), math.ceil(abs(droll))))
    s = np.linspace(0.0, 1.0, n + 1)
    lo, hi = model.limits.usable(cfg, rest)
    q = np.empty((n + 1, 6))
    q[0] = q0
    prev = q0.copy()

    def ik_at(sv, seed5):
        return model.kin.solve_ik(p0 + sv * (p1 - p0), start["pitch"] + sv * dpitch, start["roll"] + sv * droll,
                                  seed_deg=seed5, lower_deg=lo[:5], upper_deg=hi[:5])

    def walk(sa, qa, sb, depth):
        """IK at sb, seeded from qa at sa; if that fails or jumps, go there in halves (re-seeding each time),
        as the old per-tick sampling did: a distant seed can pick the other branch."""
        r = ik_at(sb, qa)
        if r.ok and np.max(np.abs(r.q - qa)) <= 5.0:
            return r
        if depth == 0:
            return r
        mid = (sa + sb) / 2.0
        rm = walk(sa, qa, mid, depth - 1)
        if not rm.ok or np.max(np.abs(rm.q - qa)) > 20.0:
            return rm
        return walk(mid, rm.q, sb, depth - 1)

    for k in range(1, n + 1):
        r = walk(s[k - 1], prev[:5], s[k], 6)
        if not r.ok:
            target = p0 + s[k] * (p1 - p0)
            raise PlanError(f"straight line leaves the reachable space at {s[k] * 100:.0f}% of the way "
                            f"(tip {np.round(target, 3).tolist()}): {r.message}")
        step = np.max(np.abs(r.q - prev[:5]))
        if step > 20.0:
            raise PlanError(f"straight line passes through a singularity or branch switch at {s[k] * 100:.0f}% "
                            f"of the way (joint jump {step:.0f} deg in one tick); use linear=false")
        q[k, :5] = r.q
        q[k, 5] = q0[5]
        prev = q[k]

    # ---- timing along the path
    vmax = np.array([cfg.vmax[j] for j in ARM_JOINTS]) * speed
    amax = np.array([cfg.amax[j] for j in ARM_JOINTS]) * speed
    ds = np.diff(s)                                                    # (n,)
    dq = np.diff(q, axis=0) / ds[:, None]                              # dq/ds per segment, (n, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_v = np.min(np.where(np.abs(dq) > 1e-9, vmax / np.abs(dq), np.inf), axis=1)       # ds/dt cap per segment
        seg_a = np.min(np.where(np.abs(dq) > 1e-9, amax / np.abs(dq), np.inf), axis=1)       # d2s/dt2 cap per segment
    seg_v = np.minimum(seg_v, vs)
    seg_a = np.minimum(seg_a, as_)
    node_v = np.full(n + 1, vs)                                        # ds/dt cap at each sample
    node_v[1:] = np.minimum(node_v[1:], seg_v)
    node_v[:-1] = np.minimum(node_v[:-1], seg_v)
    if n >= 2:                                                         # bends: |d2q/ds2| (ds/dt)^2 <= amax / 2
        ddq = np.diff(dq, axis=0) / ((ds[:-1] + ds[1:]) / 2)[:, None]  # (n-1, 6) at interior samples
        with np.errstate(divide="ignore", invalid="ignore"):
            bend = np.min(np.where(np.abs(ddq) > 1e-9, np.sqrt(0.5 * amax / np.abs(ddq)), np.inf), axis=1)
        node_v[1:-1] = np.minimum(node_v[1:-1], bend)
    sd = np.minimum(node_v, np.inf)
    sd[0] = sd[-1] = 0.0
    for k in range(n):                                                 # forward: accelerate within the caps
        sd[k + 1] = min(sd[k + 1], math.sqrt(sd[k] ** 2 + 2.0 * seg_a[k] * ds[k]))
    for k in range(n - 1, -1, -1):                                     # backward: brake within the caps
        sd[k] = min(sd[k], math.sqrt(sd[k + 1] ** 2 + 2.0 * seg_a[k] * ds[k]))
    avg = (sd[:-1] + sd[1:]) / 2.0
    if np.any(avg <= 0):
        raise PlanError("straight line could not be timed (a sample where no joint may move); use linear=false")
    tk = np.concatenate([[0.0], np.cumsum(ds / avg)])

    from scipy.interpolate import PchipInterpolator

    q_of_s = PchipInterpolator(s, q, axis=0)            # smooth along the path (no overshoot), so joint
    acc = (sd[1:] ** 2 - sd[:-1] ** 2) / (2.0 * ds)     # velocity is continuous; constant d2s/dt2 per segment

    def resample(tt_nodes):
        """The path on the tick grid: s(t) exactly from the per-segment constant acceleration, then q(s)."""
        f_ = tt_nodes[-1] / tk[-1]
        tg = _grid(tt_nodes[-1], cfg.tick_s)
        tl = tg / f_                                     # unstretched time
        k = np.clip(np.searchsorted(tk, tl, side="right") - 1, 0, n - 1)
        tau = tl - tk[k]
        sg = np.clip(s[k] + sd[k] * tau + 0.5 * acc[k] * tau * tau, 0.0, 1.0)
        sg[-1] = 1.0
        return tg, q_of_s(sg)

    def excess(tt, qq):
        if len(tt) < 3:
            return 1.0
        dt = np.diff(tt)
        v = np.diff(qq, axis=0) / dt[:, None]
        a = np.diff(v, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
        return max(float(np.max(np.abs(v) / vmax)), math.sqrt(float(np.max(np.abs(a) / amax))))

    limit_s = max(8.0, 5.0 * T_nominal)
    f = 1.0
    for _ in range(12):                    # the tick grid adds a little: stretch uniformly until it fits
        t2, q2 = resample(tk * f)
        if t2[-1] > limit_s:
            k = int(np.argmin(node_v[1:-1])) + 1 if n >= 2 else 0
            raise PlanError(f"straight line would take {t2[-1]:.0f} s: near {s[k] * 100:.0f}% of the way the "
                            f"joints must move very fast for little tip motion (close to a joint limit or "
                            f"singularity); use linear=false, or a different pitch or end point")
        e = excess(t2, q2)
        if e <= 1.0:
            return Path(t2, q2)
        f *= e * 1.02
    raise PlanError("straight line could not be timed within the joint caps; use linear=false")
