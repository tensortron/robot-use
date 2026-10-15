"""robotd's simulated rig (`--sim`): the MuJoCo twin's arms, a scene file's objects and rendered cameras, behind the
same daemon, safety checks, API and operator page as the real arms.

Nothing here opens a serial port or a camera device: the arms are `SimArmIO` on a `SimWorld` stepping in real time,
the cameras are `SimCameraIO` on one `SimRenderer`. Motion is enabled by construction, since nothing physical moves.

What the agent must not learn from the sim is where the objects really are: the drawn layout (and its seed) goes to
robotd's console only, never into the API, the event log or the operator page. The agent finds the objects the way
it does on the rig, with the cameras.

The joint limits are nominal (`NOMINAL_RANGE_DEG`, about this rig's calibrated travel), since a simulated arm has no
calibration file; the same ranges as `--mock`.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

NOMINAL_RANGE_DEG = {"shoulder_pan": 120.0, "shoulder_lift": 104.0, "elbow_flex": 97.0, "wrist_flex": 102.0,
                     "wrist_roll": 180.0}
DEFAULT_SCENE = "block_cup"


def nominal_limits(arms) -> dict:
    """{arm: ArmLimits} with the nominal ranges, for arms that have no calibration file (mock and sim)."""
    from .safety import ArmLimits

    return {n: ArmLimits({**{j: (-v, v) for j, v in NOMINAL_RANGE_DEG.items()}, "gripper": (0.0, 100.0)})
            for n in arms}


def choose_gl() -> str | None:
    """Set MUJOCO_GL (and what it needs) to a setting that renders here, before mujoco is first imported. Returns
    a description for the log, or None if nothing renders (the cameras will then say so in their frames)."""
    if os.environ.get("MUJOCO_GL"):
        return f"MUJOCO_GL={os.environ['MUJOCO_GL']} (given)"
    if "mujoco" in sys.modules:                                   # too late to choose: say so rather than pretend
        log.warning("mujoco was imported before a GL backend was chosen; the cameras use its default")
        return "mujoco's default"
    from stack.twin_mujoco.render import find_gl_env

    log.info("probing which OpenGL setting renders here (a few seconds; MUJOCO_GL skips this)...")
    env = find_gl_env()
    if env is None:
        return None
    os.environ.update(env)
    return " ".join(f"{k}={v}" for k, v in env.items()) or "the default"


@dataclass
class SimRig:
    arm_ios: dict
    cam_ios: dict
    limits: dict
    world: object                         # SimWorld: the objects' true poses live here, never in robotd's API
    layout: object                        # scene_file.Layout: name, seed, objects

    def stop(self):
        for io in self.cam_ios.values():
            io.stop()
        self.world.stop()


def build_sim(cfg, scene: str = DEFAULT_SCENE, seed: int | None = None) -> SimRig:
    """The simulated rig for robotd's config `cfg`: its arms (cfg.arms) at rest, the scene's objects drawn from
    `seed`, and its cameras (cfg.cameras; a wrist camera only for a configured arm). Call choose_gl() first."""
    from stack.rig.kinematics import ArmKinematics
    from stack.twin_mujoco.render import SimCameraIO, SimRenderer, sim_cameras
    from stack.twin_mujoco.scene import build_scene
    from stack.twin_mujoco.scene_file import load_scene_file
    from stack.twin_mujoco.sim import SimArmIO, SimWorld

    from .safety import ArmModel, rest_pose

    rig = cfg.rig
    layout = load_scene_file(scene, seed, board_front_x=rig.board.front_x_m)
    sizes = {n: (c.width, c.height) for n, c in cfg.cameras.items()}
    cams = sim_cameras(rig, sizes, arms=tuple(cfg.arms))
    if dropped := sorted(set(sizes) - {c.name for c in cams}):
        log.info("sim: no %s (its arm is not configured)", ", ".join(dropped))
    scene_obj = build_scene(rig, layout.objects, cameras=cams)
    # every arm in the scene starts folded at rest, configured or not (an unconfigured one just stays there)
    limits_all = nominal_limits(scene_obj.arms)
    start = {a: rest_pose(ArmModel(a, ArmKinematics(rig.arm_mount(a)), limits_all[a]), cfg.safety)
             for a in scene_obj.arms}
    world = SimWorld(scene_obj, start=start)
    world.start()
    renderer = SimRenderer(world, cams)
    arm_ios = {n: SimArmIO(world, n) for n in cfg.arms}
    cam_ios = {c.name: SimCameraIO(renderer, c.name) for c in cams}
    return SimRig(arm_ios, cam_ios, {n: limits_all[n] for n in cfg.arms}, world, layout)
