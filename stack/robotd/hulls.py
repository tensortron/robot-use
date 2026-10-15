"""Convex-hull clearance checks for the gripper end (sprint 2026-10-05, task 4).

robotd's original checks model each arm as joint-to-joint segments (points for the floor, capsules for
obstacles and the other arm) plus the tip point. They do not know the wrist camera exists (it sticks out
~73 mm from the wrist axis; at roll +90 it hangs underneath) nor the jaws' real extent, and a review found
poses passing every check with gripper parts in the table. This module checks the real shapes.

Geometry: the twin's scene (stack/twin_mujoco/scene.py) - the upstream SO-ARM100 camera model of both arms, the
work surface, the board and the fixed obstacles - so the twin and the safety check see the same thing. MuJoCo
collides mesh geoms by their convex hulls (the Isaac default for non-jaw parts, sprint decision D3). The jaws use
their convex decomposition (32 pieces each, CoACD, stack/assets/gripper_collision/so101_jaws.json from
tools/build_jaw_collision.py - covers every jaw vertex, ~1.15x the jaw's volume vs ~2.4x for one hull); without that
file, one hull per jaw. For the floor a convex hull is exact either way: its lowest point is the part's.

Each sample sets both arms' joints, runs MuJoCo's kinematics and collision (broad + narrow phase in C) with
every relevant geom given a detection margin, and keeps the closest distance per (arm, part, target). Rules, as
the floor rule: a path may START inside a margin but must not get closer than it started, and must END clear; a
single pose is checked absolutely. Margins (decision D5) are SafetyConfig.hull_margins_m.

Enforced parts: camera (mount + module), jaws (fixed jaw body + moving jaw), gripper (the wrist-roll servo on
the gripper); against the floor, obstacles, the other arm and the arm's own shoulder, base and upper arm (task 12).
All of it runs only with SafetyConfig.hull_checks on and both arms configured (robotd logs a warning otherwise);
then only the segment checks and the folded-roll rule (compute.py) remain. Report-only (task 4b, decision D6): wrist, forearm, upper arm - their clearances are returned for
logging, never as violations. Shoulder and base are not checked (they stand on the board).
"""

from __future__ import annotations

import functools
import threading

import numpy as np

ENFORCED = ("camera", "jaws", "gripper")
# Against the other arm the wrist is enforced too (sprint 2026-10-08): with hull checks on, the two arms'
# wrist+gripper capsules are no longer compared (safety._clearance_from_segments), so the hulls must cover the whole
# segment - the wrist body as well as the camera, jaws and gripper.
OTHER_ARM_ENFORCED = ENFORCED + ("wrist",)
GRIP_REACH_M = 0.13        # every enforced part lies within this of the wrist+gripper segment (measured: 115 mm)
STILL_ARM_DEG = 0.1        # an arm whose five joints change less than this along a path is "not moving" (gripper only)
REST_DIP_M = 0.002          # rest / back-off paths starting inside a margin may get this much closer before lifting
REPORTED = ("wrist", "forearm", "upper_arm")
TARGETS = ("floor", "obstacle", "other_arm")

# MuJoCo contype / conaffinity bits: each arm collides with the environment and the other arm, not itself;
# the environment does not collide with itself.
_BIT = {"left": 1, "right": 2, "env": 4}
OWN_LINKS = ("shoulder", "base", "upper_arm")   # what an arm's gripper end is checked against on the arm itself


def _part_of(body: str, mesh: str) -> str | None:
    b, m = body.split("_", 1)[1] if "_" in body else body, mesh
    if b.startswith("wrist_camera"):
        return "camera"
    if b.startswith("moving_jaw") or (b == "gripper" and "wrist_roll_follower" in m):
        return "jaws"
    if b == "gripper":
        return "gripper"
    if b == "wrist":
        return "wrist"
    if b == "lower_arm":
        return "forearm"
    if b == "upper_arm":
        return "upper_arm"
    return None                                    # shoulder, base: not checked


def _moving(paths: dict) -> set:
    """The arms whose five joints change along their path (>= STILL_ARM_DEG; NaN counts as moving)."""
    out = set()
    for a, p in paths.items():
        q = np.asarray(p.q, float)[:, :5]
        if not np.all(np.abs(q - q[0]) < STILL_ARM_DEG):
            out.add(a)
    return out


def _locked(fn):
    """The checker's MjModel / MjData are shared: every method that touches them holds its (re-entrant) lock, so
    no caller can race another (a background pose check crashed MuJoCo against an unlocked caller, 2026-10-07)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)
    return wrapper


class HullCheckIncomplete(RuntimeError):
    """MuJoCo dropped contacts: the distances are not complete, so nothing may be concluded from them."""


class HullChecker:
    def __init__(self, rig, safety, objects=()):
        import mujoco

        from ..twin_mujoco.scene import build_scene, load_jaw_hulls
        from ..twin_mujoco.sim import GRIP_I, gripper_real_to_rad

        self._mj = mujoco
        self.lock = threading.RLock()
        self.cfg = safety
        self._grip = (GRIP_I, gripper_real_to_rad)
        self.jaw_hulls = load_jaw_hulls()               # convex pieces per jaw (task 3), or single hulls if absent
        self.scene = build_scene(rig, objects, jaw_hulls=self.jaw_hulls)
        m = self.model = self.scene.model
        self.data = mujoco.MjData(m)
        margins = safety.hull_margins_m
        broad = self.broad_m = max(max(v.values()) for v in margins.values()) + 0.005
        self.geom_arm, self.geom_part, self.geom_target = {}, {}, {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_margin[g] = 0.0
            if body == "world":
                if name in ("work_surface", "board"):
                    self.geom_target[g] = ("floor", "the board top" if name == "board" else "the table")
                elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    self.geom_target[g] = ("obstacle", name.replace("_", " "))
                else:
                    continue
                m.geom_contype[g], m.geom_conaffinity[g] = _BIT["env"], _BIT["left"] | _BIT["right"]
                continue
            arm = "left" if body.startswith("left_") else "right" if body.startswith("right_") else None
            if arm is None or m.geom_group[g] != 3 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue                                # visual geoms and objects are not checked
            mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""
            part = _part_of(body, mesh)
            if part is None:
                continue
            other = "right" if arm == "left" else "left"
            self.geom_arm[g], self.geom_part[g] = arm, part
            m.geom_contype[g] = _BIT[arm]
            m.geom_conaffinity[g] = _BIT["env"] | _BIT[other]
            m.geom_margin[g] = broad
        # (Tried 2026-10-05: colliding each jaw's single hull first and its pieces only near something - slower:
        # a single hull of the 14k-vertex jaw mesh costs MuJoCo more than its 32 small pieces.)
        # Own arm (task 12, pulled in 2026-10-06): an arm's camera / jaws / gripper against its own shoulder, base
        # and upper arm - folded at roll +90 the camera is inside the shoulder, and some +90 poses near the base put
        # it inside the upper arm. Same rules, but measured outside mj_collision: with the shared 25 mm detection
        # margin the jaw pieces against the shoulder's 9-13k-vertex meshes made ~800 contacts a sample (20x the
        # plan time). Here: mj_geomDistance capped at OWN_DIST_CAP_M, pairs pre-filtered by bounding spheres.
        self.own_link: dict = {}                        # geom -> (arm, link)
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            arm, _, link = body.partition("_")
            if arm in ("left", "right") and link in OWN_LINKS and m.geom_group[g] == 3 \
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                self.own_link[g] = (arm, link)
        self.own_links = {a: [g for g, (x, _) in self.own_link.items() if x == a] for a in ("left", "right")}
        if not all(self.own_links.values()):
            raise RuntimeError("hull checker: the arms' shoulder / base / upper arm geoms were not found")
        self.own_cap = max(margins.get("own_arm", {}).values(), default=0.0) + 0.010
        self.own_end = {a: [g for g, p in self.geom_part.items() if self.geom_arm[g] == a and p in ENFORCED]
                        for a in ("left", "right")}
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)   # nothing here is a parent pair
        # Mid-phase culls geom pairs with per-body bounding volumes built at compile time, without the detection
        # margins set above: it silently dropped jaw pieces 17 mm from the table (found 2026-10-05). Off: every
        # candidate pair is tested with its margin-inflated bounding box (cheap at this size).
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)

    # ------------------------------------------------------------------ one pose
    @_locked
    def _set(self, q_by_arm):
        GRIP_I, to_rad = self._grip
        for arm, q in q_by_arm.items():
            info, mnt = self.scene.arms[arm], self.scene.mounts[arm]
            q = np.asarray(q, float)
            for i in range(5):
                self.data.qpos[info["qpos"][i]] = mnt.signs[i] * np.radians(q[i]) + np.radians(mnt.offsets_deg[i])
            self.data.qpos[info["qpos"][GRIP_I]] = to_rad(q[GRIP_I])
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_collision(self.model, self.data)

    @_locked
    def distances(self, q_by_arm, env: bool = True) -> dict:
        """{(arm, part, target_kind, target_name): closest distance (m)} for pairs within the detection margin.
        The table and the board top are one target, "the floor" (as the segment floor rule's floor_z). An arm missing
        from q_by_arm (no reading) is left out of the scene for this call rather than kept at a stale pose.
        env=False: arm-to-arm pairs only. Raises HullCheckIncomplete if MuJoCo's contact buffer overflowed (it then
        drops pairs silently - review 2026-10-05)."""
        missing = [a for a in self.scene.arms if a not in q_by_arm]
        saved = {}
        for g in (*self.geom_arm, *self.geom_target):
            if self.geom_arm.get(g) in missing or (not env and g in self.geom_target):
                saved[g] = (self.model.geom_contype[g], self.model.geom_conaffinity[g])
                self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 0
        full = self._mj.mjtWarning.mjWARN_CONTACTFULL
        self.data.warning[full].number = 0
        try:
            self._set(q_by_arm)
        finally:
            for g, (ct, ca) in saved.items():
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = ct, ca
        if self.data.warning[full].number:
            raise HullCheckIncomplete(f"contact buffer full ({self.data.ncon} contacts)")
        out = {}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a not in self.geom_part:
                    continue
                if b in self.geom_target:
                    kind, name = self.geom_target[b]
                    if kind == "floor":
                        name = "the floor"                  # table or board top: one surface for the rules
                elif b in self.geom_part and self.geom_arm[b] != self.geom_arm[a]:
                    kind, name = "other_arm", f"the {self.geom_arm[b]} arm's {self.geom_part[b].replace('_', ' ')}"
                else:
                    continue
                key = (self.geom_arm[a], self.geom_part[a], kind, name)
                d = float(c.dist)
                if d < out.get(key, np.inf):
                    out[key] = d
        if env:                                         # own arm (not at launch: that re-check is arm-to-arm)
            self._own_distances(q_by_arm, out)
        return out

    def _own_distances(self, q_by_arm, out):
        mj, m, d = self._mj, self.model, self.data
        cap = self.own_cap
        for arm in q_by_arm:
            ends, links = self.own_end[arm], self.own_links[arm]
            for o in links:
                near = np.linalg.norm(d.geom_xpos[ends] - d.geom_xpos[o], axis=1) - m.geom_rbound[ends] - m.geom_rbound[o]
                for g in np.asarray(ends)[near < cap]:
                    dist = mj.mj_geomDistance(m, d, int(g), o, cap, None)
                    if dist < cap:
                        key = (arm, self.geom_part[int(g)], "own_arm", f"its own {self.own_link[o][1].replace('_', ' ')}")
                        if dist < out.get(key, np.inf):
                            out[key] = float(dist)

    @_locked
    def own_clearance(self, arm, q_by_arm) -> tuple[float, str]:
        """(closest distance, "part vs link") of this arm's camera, jaws and gripper to its own shoulder, base and
        upper arm, at one pose, without a detection margin limit (a diagnostic; the check itself is the
        "own_arm" target of distances / check)."""
        self._set(q_by_arm)
        mine = [g for g, p in self.geom_part.items() if self.geom_arm[g] == arm and p in ENFORCED]
        best, what = np.inf, ""
        for g in mine:
            for o in self.own_links[arm]:
                d = self._mj.mj_geomDistance(self.model, self.data, g, o, 0.05, None)
                if d < best:
                    body = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[o])
                    best, what = d, f"{self.geom_part[g]} vs its own {body.split('_', 1)[1].replace('_', ' ')}"
        return float(best), what

    # ------------------------------------------------------------------ paths
    @_locked
    def check(self, paths: dict, current: dict, others: dict | None = None, rest: bool = False):
        """Violations (enforced parts) and the report-only clearances, for paths of some arms; the other arm
        follows `others` (its remaining path) or holds `current`."""
        others = {a: p for a, p in (others or {}).items() if a not in paths}
        grid = self._grid(paths, others)
        try:
            series = self._series(paths, current, others, grid, shift=0, env=True)
        except HullCheckIncomplete as e:
            return [self._incomplete(paths, e)], {}
        return self._judge(series, grid, rest, moving=_moving({**paths, **others}))

    @_locked
    def arm_check(self, paths: dict, current: dict, others: dict | None = None, window_s: float = 0.0,
                  near=None) -> list:
        """The arm-to-arm part of check() alone, at launch: a moving other arm (`others`) may be anywhere from 0
        to window_s further along its path (as safety.check_clearance); every tick shift is checked.
        near: optional (shifts, samples) bool array - False where the two wrist+gripper segments are more than
        2 * GRIP_REACH_M + the detection margin apart (no gripper part can be within it): those samples are skipped."""
        others = {a: p for a, p in (others or {}).items() if a not in paths}
        grid = self._grid(paths, others)
        K = int(np.ceil(window_s / self.cfg.tick_s)) if others else 0
        worst = []
        for k in range(K + 1):
            if near is not None and not near[k].any():
                continue
            try:
                series = self._series(paths, current, others, grid, shift=k, env=False,
                                      only=None if near is None else near[k])
            except HullCheckIncomplete as e:
                return [self._incomplete(paths, e)]
            v, _ = self._judge(series, grid, rest=False, moving=_moving({**paths, **others}))
            if v and (not worst or v[0].value < worst[0].value):
                worst = v
        return worst

    # ------------------------------------------------------------------ internals
    def _grid(self, paths, others):
        T = max([p.duration for p in paths.values()] + [p.duration for p in others.values()], default=0.0)
        return np.arange(int(np.ceil(T / self.cfg.tick_s)) + 1) * self.cfg.tick_s

    def _series(self, paths, current, others, grid, shift, env, only=None):
        """{key: distance per grid sample (inf = beyond the detection margin)}; `others` run `shift` ticks ahead;
        `only`: a bool mask of the samples to compute (the rest are known to be beyond the margin)."""
        moving = set(paths) | set(others)
        n, dt = len(grid), shift * self.cfg.tick_s
        series: dict = {}
        for k, t in enumerate(grid):
            if only is not None and not only[k]:
                continue
            q = {}
            for a in self.scene.arms:
                if a in paths:
                    q[a] = paths[a].at(t)
                elif a in others:
                    q[a] = others[a].at(t + dt)
                elif current.get(a) is not None:
                    q[a] = current[a]
            for key, d in self.distances(q, env=env).items():
                arm, part, kind, name = key
                if arm not in paths and not (kind == "other_arm" and moving):
                    continue                            # a still arm's parts vs the table: not this motion's doing
                series.setdefault(key, np.full(n, np.inf))[k] = d
        return series

    def _incomplete(self, paths, e):
        from .safety import Violation
        arm = next(iter(paths)) if len(paths) == 1 else "both"
        return Violation("hull_check_incomplete", arm, None, 0.0, 0.0, "a complete check",
                         f"the gripper clearance check could not complete ({e}); nothing was moved - send it again "
                         f"or move the arms apart first")

    def _judge(self, series, grid, rest, moving=None):
        """Apply the rules (start inside a margin allowed, never closer, end clear) to sampled distances.

        `moving`: the arms whose five joints move along their path (None: all). For a part of an arm that does not
        move (a gripper-only motion), "end clear" is dropped: the motion brought the arm nowhere, so refusing it for
        ending where it started would make nothing safer (sprint 2026-10-07: open() on a torque-off arm settled
        4.5 mm from its own shoulder was refused). "Never closer than at the start" still applies, per part: the
        part's closest point may not come nearer than it started (for the jaws, the nearer of the two jaws - a
        moving jaw may swing in, but no closer than the jaws' starting gap). Against the other arm this holds only
        if that arm is not moving either. A still arm gets no rest dip."""
        from .safety import Violation

        n = len(grid)
        margins = self.cfg.hull_margins_m
        violations, report = [], {}
        for key, d in series.items():
            arm, part, kind, name = key
            if part not in (OTHER_ARM_ENFORCED if kind == "other_arm" else ENFORCED):
                report[key] = float(np.min(d))
                continue
            need = margins[kind][part]
            if kind == "other_arm":                      # a pair takes the larger of its two parts' margins; a
                m = margins[kind]                        # report-only part (forearm, upper arm) counts as the largest
                other = name.rsplit("arm's ", 1)[-1].replace(" ", "_")
                need = max(need, m[other] if other in OTHER_ARM_ENFORCED else max(m.values()))
            allowed = np.full(n, need)
            if n > 1:                                    # may start inside; never closer; must end clear
                start = d[0]
                still = moving is not None and arm not in moving and not (kind == "other_arm" and moving)
                if rest and start < need and not still:  # rest / back-off out of a low pose may dip a little
                    start -= REST_DIP_M                  # first (the joint-space path swings before it lifts)
                allowed = np.minimum(np.full(n, need), start)
                if not still:
                    allowed[-1] = need
            bad = np.nonzero(d < allowed - 1e-6)[0]
            if not len(bad):
                continue
            k = int(bad[0])
            what = {"camera": "wrist camera", "jaws": "jaws", "gripper": "gripper", "wrist": "wrist"}[part]
            gap = max(d[k], -1.0)
            touch = "would touch" if gap <= 0 else f"would be {gap * 1000:.1f} mm from"
            violations.append(Violation(
                f"{part}_clearance", arm, None, float(grid[k]), float(gap), f">= {need * 1000:.0f} mm from {name}",
                f"{arm} {what} {touch} {name} at t={grid[k]:.2f}s; minimum {need * 1000:.0f} mm"
                + (" (a camera underneath the gripper: roll toward -90 puts it on top)" if part == "camera" and kind == "floor" else "")))
        violations.sort(key=lambda v: v.t)
        return violations, report
