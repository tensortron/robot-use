"""SO-101 forward and inverse kinematics in the rig frame, numpy only.

Model: ``stack/assets/so101_new_calib.urdf`` (TheRobotStudio/SO-ARM100, "new
calibration": every joint's zero is the middle of its calibrated range, which is
also lerobot's DEGREES zero). lerobot feeds degrees straight into this URDF as
radians (``lerobot/model/kinematics.py``), and so do we. Each joint still has a
configurable sign and offset in ``ArmMount`` so a mismatch found on the rig is a
config change, not a code change.

Rig frame (docs/camera-calibration.md): origin on the work surface midway between
the two arm bases, +x forward away from the arms, +y to the robot's left, +z up.
The arm base plates sit on an 18 mm board, and the shoulder-pan axes are 371.5 mm
apart. In the URDF's base frame the arm reaches along +x at zero, which is the
rig's forward, so both mounts default to yaw 0.

Cartesian targets are gripper-tip position + approach pitch + wrist roll (the arm
has five joints before the gripper, so a full 6-DOF pose is not reachable in
general):

- ``pitch``: angle of the approach axis in the arm's vertical plane, degrees:
  0 = pointing forward horizontally, -90 = straight down, +90 = straight up.
- ``roll``: the wrist_roll joint angle itself, degrees. It turns the gripper about
  its approach axis and barely moves the tip, so IK fixes it directly.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARM_JOINTS = JOINTS + ("gripper",)
IK_JOINTS = 4                       # pan, lift, elbow, wrist_flex solve; wrist_roll is set directly
TIP_LINK = "gripper_frame_link"
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "assets" / "so101_new_calib_camera.urdf"



def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q):
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    t = 1.0 - c
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def _homogeneous(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


@dataclass(frozen=True)
class ArmMount:
    """Where an arm's URDF base sits in the rig frame, and how lerobot degrees map to URDF angles."""

    xyz: tuple[float, float, float]
    yaw_deg: float = 0.0
    signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)


def default_mount(arm: str) -> ArmMount:
    """The arm's mount as robot.yaml describes it (pan axis at (0, +/-sep/2), base on the board top)."""
    from .rig import load_rig

    return load_rig().arm_mount(arm)


@dataclass
class IKResult:
    ok: bool
    q: np.ndarray                    # 5 joint angles, lerobot degrees (best found, even if not ok)
    pos_err_m: float
    pitch_err_deg: float
    reached_xyz: np.ndarray
    reached_pitch_deg: float
    iterations: int
    message: str = ""


@dataclass
class _Joint:
    name: str
    origin: np.ndarray               # 4x4, parent frame -> joint frame
    axis: np.ndarray | None          # unit axis in the joint frame, None if fixed
    index: int | None                # index into JOINTS, None if fixed


class ArmKinematics:
    """FK, Jacobians and IK for one SO-101 arm, in the rig frame."""

    def __init__(self, mount: ArmMount, urdf_path: Path | str = DEFAULT_URDF, tip_link: str = TIP_LINK):
        self.mount = mount
        self._signs = np.asarray(mount.signs, float)
        self._offsets = np.deg2rad(np.asarray(mount.offsets_deg, float))
        self._base = _homogeneous(_axis_angle((0, 0, 1), math.radians(mount.yaw_deg)), mount.xyz)
        self._chain = self._parse(Path(urdf_path), tip_link)
        # Frames of interest, by chain position: the joint frames of the five joints, then the tip.
        self._joint_pos = [i for i, j in enumerate(self._chain) if j.index is not None]
        zero = self._frames_rad(np.zeros(len(JOINTS)))
        tip, wrist = zero[-1], zero[self._joint_pos[4]]
        # Approach axis: the tip-frame column that points from the wrist toward the tip.
        d = tip[:3, 3] - wrist[:3, 3]
        self._approach_col = int(np.argmax(np.abs(tip[:3, :3].T @ d)))
        self._approach_sign = float(np.sign(tip[:3, self._approach_col] @ d))
        # Forward direction of the arm plane, fixed in the shoulder (post-pan) frame.
        shoulder = zero[self._joint_pos[0]]
        base_x = self._base[:3, 0]
        self._fwd_in_shoulder = shoulder[:3, :3].T @ base_x
        # Pan sign: which way positive pan swings the tip, for the IK pan seed.
        dq = np.zeros(len(JOINTS))
        dq[0] = 1e-3
        p0, p1 = tip[:3, 3], self._frames_rad(dq)[-1][:3, 3]
        base_y = self._base[:3, 1]
        self._pan_dir = float(np.sign((p1 - p0) @ base_y)) or 1.0

    # ------------------------------------------------------------------ model
    @staticmethod
    def _parse(path: Path, tip_link: str) -> list[_Joint]:
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip_link
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        chain.reverse()
        joints = []
        for j in chain:
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            name = j.get("name")
            if j.get("type") == "revolute":
                if name not in JOINTS:
                    raise ValueError(f"unexpected revolute joint {name!r} on the chain to {tip_link}")
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                joints.append(_Joint(name, _homogeneous(_rpy(*rpy), xyz), axis / np.linalg.norm(axis),
                                     JOINTS.index(name)))
            else:
                joints.append(_Joint(name, _homogeneous(_rpy(*rpy), xyz), None, None))
        if sorted(j.index for j in joints if j.index is not None) != list(range(len(JOINTS))):
            raise ValueError(f"chain to {tip_link} does not contain exactly {JOINTS}")
        return joints

    def to_urdf_rad(self, q_deg) -> np.ndarray:
        q = np.asarray(q_deg, float)[: len(JOINTS)]
        return self._signs * np.deg2rad(q) + self._offsets

    def _frames_rad(self, q_rad) -> list[np.ndarray]:
        """World frame after each chain element (joint rotation applied)."""
        T = self._base.copy()
        out = []
        for j in self._chain:
            T = T @ j.origin
            if j.axis is not None:
                T = T @ _homogeneous(_axis_angle(j.axis, q_rad[j.index]))
            out.append(T)
        return out

    # ------------------------------------------------------------------ FK
    def frames(self, q_deg) -> list[np.ndarray]:
        return self._frames_rad(self.to_urdf_rad(q_deg))

    def fk(self, q_deg) -> np.ndarray:
        """4x4 tip pose in the rig frame."""
        return self.frames(q_deg)[-1]

    def keypoints(self, q_deg) -> np.ndarray:
        """(6, 3): pan, lift, elbow, wrist_flex, wrist_roll joint origins, then the tip."""
        f = self.frames(q_deg)
        return np.array([f[i][:3, 3] for i in self._joint_pos] + [f[-1][:3, 3]])

    def _approach(self, tip: np.ndarray) -> np.ndarray:
        return self._approach_sign * tip[:3, self._approach_col]

    def _plane(self, frames) -> tuple[np.ndarray, np.ndarray]:
        """(forward, pitch axis) of the arm plane at this pan angle."""
        fwd = frames[self._joint_pos[0]][:3, :3] @ self._fwd_in_shoulder
        fwd = fwd - fwd[2] * np.array([0, 0, 1.0])
        fwd /= np.linalg.norm(fwd)
        return fwd, np.cross(fwd, [0, 0, 1.0])

    def pitch_deg(self, q_deg) -> float:
        f = self.frames(q_deg)
        a = self._approach(f[-1])
        fwd, _ = self._plane(f)
        return math.degrees(math.atan2(a[2], a @ fwd))

    def pose(self, q_deg) -> dict:
        """Tip position (m), approach pitch and wrist roll (deg): the Cartesian command's coordinates."""
        return {"xyz": self.fk(q_deg)[:3, 3].copy(), "pitch": self.pitch_deg(q_deg),
                "roll": float(np.asarray(q_deg, float)[4])}

    # ------------------------------------------------------------------ IK
    def _jacobian(self, frames) -> tuple[np.ndarray, np.ndarray, float]:
        """Position Jacobian (3x4) and pitch gradient (4,) w.r.t. lerobot radians, plus pitch."""
        tip = frames[-1][:3, 3]
        a = self._approach(frames[-1])
        fwd, n = self._plane(frames)
        Jp = np.zeros((3, IK_JOINTS))
        Jr = np.zeros(IK_JOINTS)
        for k in range(IK_JOINTS):
            F = frames[self._joint_pos[k]]
            w = F[:3, :3] @ self._chain[self._joint_pos[k]].axis * self._signs[k]
            Jp[:, k] = np.cross(w, tip - F[:3, 3])
            Jr[k] = w @ n
        return Jp, Jr, math.atan2(a[2], a @ fwd)

    def solve_ik(self, xyz, pitch_deg: float | None = None, roll_deg: float | None = None, *,
                 seed_deg, lower_deg, upper_deg, tol_m: float = 0.002, tol_pitch_deg: float = 2.0,
                 max_iter: int = 60, extra_seeds=()) -> IKResult:
        """Closed-form IK, polished by damped least squares; joint limits always respected.

        With the wrist roll fixed, the lift, elbow and wrist_flex axes are parallel and the tip
        stays in one vertical plane through the pan axis (offset sideways by the roll, measured
        < 0.02 mm off-plane), so the solution is exact: 2 pan branches (reach forward / back over)
        x 2 elbow branches. Without a pitch, pitch is scanned and the valid solution nearest the
        seed wins. If nothing is within limits, damped least squares from the seeds finds the
        closest reachable point for the rejection message.
        """
        target = np.asarray(xyz, float)
        lo = np.asarray(lower_deg, float)[: len(JOINTS)]
        hi = np.asarray(upper_deg, float)[: len(JOINTS)]
        seed = np.clip(np.asarray(seed_deg, float)[: len(JOINTS)].copy(), lo, hi)
        if roll_deg is not None:
            if not lo[4] <= roll_deg <= hi[4]:
                return IKResult(False, seed, math.inf, math.inf, self.fk(seed)[:3, 3], self.pitch_deg(seed), 0,
                                f"roll {roll_deg:.1f} outside usable range [{lo[4]:.1f}, {hi[4]:.1f}]")
            seed[4] = roll_deg

        pitches = (np.array([pitch_deg], float) if pitch_deg is not None
                   else np.arange(-180.0, 180.0, 1.0))
        cands = self._closed_form(target, np.deg2rad(pitches), seed[4])       # (n, 5) degrees
        if len(cands):
            inside = np.all((cands >= lo - 1e-9) & (cands <= hi + 1e-9), axis=1)
            cands = cands[inside]
        if len(cands):
            w = np.array([1.0, 1.0, 1.0, 0.5, 0.0])     # prefer staying close to the seed, wrist counts less
            order = np.argsort(np.sum(w * (cands - seed) ** 2, axis=1))
            for i in order[:4]:
                r = self._dls(target, pitch_deg, cands[i], lo, hi, tol_m, tol_pitch_deg, 8)
                if r.ok:
                    return r

        aimed = seed.copy()
        rel = np.linalg.inv(self._base) @ np.append(target, 1.0)
        aimed[0] = np.clip(self._pan_dir * math.degrees(math.atan2(rel[1], rel[0])), lo[0], hi[0])
        seeds = [seed, aimed] + [np.clip(np.r_[np.asarray(s, float)[:4], seed[4]], lo, hi) for s in extra_seeds]
        best = None
        for s in seeds:
            r = self._dls(target, pitch_deg, s, lo, hi, tol_m, tol_pitch_deg, max_iter)
            if r.ok:
                return r
            if best is None or (r.pos_err_m + 0.001 * r.pitch_err_deg) < (best.pos_err_m + 0.001 * best.pitch_err_deg):
                best = r
        best.message = (f"unreachable within joint limits: closest reachable tip "
                        f"{np.round(best.reached_xyz, 3).tolist()} is {best.pos_err_m * 1000:.0f} mm from the target"
                        + (f", pitch off by {best.pitch_err_deg:.1f} deg" if pitch_deg is not None else ""))
        return best

    def _planar(self, roll_deg: float) -> dict:
        """Arm-plane geometry at this wrist roll, in the base frame with pan = 0 (cached)."""
        key = round(float(roll_deg), 6)
        cache = self.__dict__.setdefault("_planar_cache", {})
        if key in cache:
            return cache[key]
        base_inv = np.linalg.inv(self._base)
        def pts(q):
            f = self.frames(q)
            return [(base_inv @ f[i])[:3, 3] for i in self._joint_pos] + [(base_inv @ f[-1])[:3, 3]]
        q0 = np.array([0, 0, 0, 0, roll_deg], float)
        P, L, E, W, _, T = pts(q0)
        uv = lambda p: np.array([p[0] - P[0], p[2]])
        g = {"P": P[:2], "delta": T[1] - P[1], "L": uv(L), "a1": uv(E) - uv(L), "a2": uv(W) - uv(E),
             "a3": uv(T) - uv(W), "pitch0": math.radians(self.pitch_deg(q0))}
        # Rotation sense of each planar joint in the (u, v) plane, per lerobot radian.
        sig = []
        for k, (a, b) in zip((1, 2, 3), ((L, T), (E, T), (W, T))):
            dq = q0.copy()
            dq[k] += 1e-3
            p1 = pts(dq)[-1]
            r0, r1 = uv(T) - uv(a), uv(p1) - uv(a)
            sig.append(float(np.sign(r0[0] * r1[1] - r0[1] * r1[0])))
        g["sigma"] = np.array(sig)
        cache[key] = g
        return g

    def reachable_pitches(self, xyz, roll_deg: float, lower_deg, upper_deg, pitches_deg) -> list[float]:
        """The pitches (from ``pitches_deg``) at which the tip can be at xyz with an exact IK solution inside the
        limits. Closed form only: fast enough to scan a whole range."""
        lo = np.asarray(lower_deg, float)[: len(JOINTS)]
        hi = np.asarray(upper_deg, float)[: len(JOINTS)]
        out = []
        for p in pitches_deg:
            c = self._closed_form(np.asarray(xyz, float), np.deg2rad([float(p)]), float(roll_deg))
            if len(c) and np.any(np.all((c >= lo - 1e-9) & (c <= hi + 1e-9), axis=1)):
                out.append(float(p))
        return out

    def _closed_form(self, target, pitches_rad, roll_deg) -> np.ndarray:
        """All exact solutions (n, 5) in lerobot degrees for the given pitches; limits not applied."""
        g = self._planar(roll_deg)
        tb = (np.linalg.inv(self._base) @ np.append(target, 1.0))[:3]
        dx, dy = tb[0] - g["P"][0], tb[1] - g["P"][1]
        rho2 = dx * dx + dy * dy
        if rho2 < g["delta"] ** 2 - 1e-8:              # closer to the pan axis than the tip's side offset
            return np.zeros((0, 5))
        u_abs = math.sqrt(max(rho2 - g["delta"] ** 2, 0.0))
        a1, a2, a3, s = g["a1"], g["a2"], g["a3"], g["sigma"]
        n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
        gamma = math.atan2(a2[1], a2[0]) - math.atan2(a1[1], a1[0])
        out = []
        for u_t in (u_abs, -u_abs):
            psi = math.atan2(dy, dx) - math.atan2(g["delta"], u_t)
            q_pan = self._pan_dir * math.degrees(psi)
            q_pan = (q_pan + 180.0) % 360.0 - 180.0
            total = pitches_rad - g["pitch0"]                          # alpha1 + alpha2 + alpha3
            c, sn = np.cos(total), np.sin(total)
            wu = u_t - (c * a3[0] - sn * a3[1])
            wv = tb[2] - (sn * a3[0] + c * a3[1])
            du, dv = wu - g["L"][0], wv - g["L"][1]
            d2 = du * du + dv * dv
            cosv = (d2 - n1 * n1 - n2 * n2) / (2 * n1 * n2)
            ok = np.abs(cosv) <= 1.0
            if not np.any(ok):
                continue
            du, dv, total_ok = du[ok], dv[ok], total[ok]
            base_ang = np.arccos(np.clip(cosv[ok], -1, 1))
            for br in (1.0, -1.0):
                al2 = br * base_ang - gamma
                vu = a1[0] + np.cos(al2) * a2[0] - np.sin(al2) * a2[1]
                vv = a1[1] + np.sin(al2) * a2[0] + np.cos(al2) * a2[1]
                al1 = np.arctan2(dv, du) - np.arctan2(vv, vu)
                al3 = total_ok - al1 - al2
                qs = np.degrees(np.stack([al1 / s[0], al2 / s[1], al3 / s[2]], axis=1))
                qs = (qs + 180.0) % 360.0 - 180.0
                n = len(qs)
                out.append(np.column_stack([np.full(n, q_pan), qs, np.full(n, float(roll_deg))]))
        return np.vstack(out) if out else np.zeros((0, 5))

    def _dls(self, target, pitch_deg, q0, lo, hi, tol_m, tol_pitch_deg, max_iter) -> IKResult:
        W_PITCH = 0.1          # metres per radian: a 10 deg pitch error weighs like 17 mm
        LAMBDA = 0.02
        MAX_STEP = math.radians(15)
        q = q0.copy()
        lo_r, hi_r = np.deg2rad(lo[:IK_JOINTS]), np.deg2rad(hi[:IK_JOINTS])
        seed_r = np.deg2rad(q0[:IK_JOINTS])
        it = 0
        for it in range(1, max_iter + 1):
            frames = self.frames(q)
            Jp, Jr, pitch = self._jacobian(frames)
            e_pos = target - frames[-1][:3, 3]
            if pitch_deg is None:
                J, e = Jp, e_pos
            else:
                e_pitch = math.remainder(math.radians(pitch_deg) - pitch, math.tau)
                J, e = np.vstack([Jp, W_PITCH * Jr]), np.r_[e_pos, W_PITCH * e_pitch]
            pe = np.linalg.norm(e_pos)
            if pe <= tol_m and (pitch_deg is None or abs(math.degrees(e_pitch)) <= tol_pitch_deg):
                break
            JJt = J @ J.T + (LAMBDA ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            if pitch_deg is None:   # redundant: pull gently toward the seed in the null space
                Jpinv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(3))
                dq += (np.eye(IK_JOINTS) - Jpinv @ J) @ (0.1 * (seed_r - np.deg2rad(q[:IK_JOINTS])))
            n = np.max(np.abs(dq))
            if n > MAX_STEP:
                dq *= MAX_STEP / n
            qr = np.clip(np.deg2rad(q[:IK_JOINTS]) + dq, lo_r, hi_r)
            q[:IK_JOINTS] = np.rad2deg(qr)
        frames = self.frames(q)
        reached = frames[-1][:3, 3].copy()
        a = self._approach(frames[-1])
        fwd, _ = self._plane(frames)
        rp = math.degrees(math.atan2(a[2], a @ fwd))
        pe = float(np.linalg.norm(target - reached))
        ae = 0.0 if pitch_deg is None else abs(math.degrees(math.remainder(math.radians(pitch_deg - rp), math.tau)))
        ok = pe <= tol_m and ae <= tol_pitch_deg
        return IKResult(ok, q, pe, ae, reached, rp, it)
