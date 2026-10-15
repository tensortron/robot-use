"""Run robotd.

    .venv/bin/python -m stack.robotd                      # real rig, stack/robotd.yaml
    .venv/bin/python -m stack.robotd --mock               # no hardware: mock arms and cameras
    .venv/bin/python -m stack.robotd --sim [--scene block_cup] [--seed N]
                                                          # no hardware: the simulated rig (physics, rendered cameras)

Operator page: http://<robot-host>:8772/   API (localhost only): http://127.0.0.1:8770/state
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time

from .config import DEFAULT_CONFIG, load_config
from .daemon import RobotD
from .hardware import ArmIO, CameraIO, MockArmIO, MockCameraIO, device_holders
from .http import serve


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--mock", action="store_true", help="mock arms and cameras (no hardware)")
    ap.add_argument("--sim", action="store_true",
                    help="the simulated rig: MuJoCo arms, a scene's objects, rendered cameras (no hardware; motion on)")
    ap.add_argument("--scene", default=None, help="--sim: a scene name (stack/twin_mujoco/scenes) or a .yaml path")
    ap.add_argument("--seed", type=int, default=None, help="--sim: the layout's seed (default: drawn and printed)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--enable-motion", action="store_true",
                    help="allow motion for this run (robotd.yaml keeps motion_enabled: false in the repository)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.sim and (a.mock or a.enable_motion):
        ap.error("--sim runs alone: it is never combined with --mock, and its motion is on by construction "
                 "(--enable-motion is for the real arms)")
    if not a.sim and (a.scene or a.seed is not None):
        ap.error("--scene and --seed are for --sim")
    return a


def backends(a, cfg):
    """(arm_ios, cam_ios, limits, sim) for the mode asked for. Only the real-rig branch touches devices: --sim and
    --mock build no ArmIO or CameraIO (test_robotd_sim checks it). Returns None if a device is held by another
    process, or the simulated rig cannot be built."""
    if a.sim:
        from .sim import DEFAULT_SCENE, build_sim, choose_gl
        gl = choose_gl()                       # before anything imports mujoco
        logging.info("sim rendering: %s", gl or "NO GL BACKEND RENDERS HERE: the cameras will say so")
        cfg.motion_enabled = True              # nothing physical moves
        cfg.backend = "sim"
        cfg.log_dir = cfg.log_dir / "sim"      # keep the sim's events apart from the rig's
        try:
            sim = build_sim(cfg, a.scene or DEFAULT_SCENE, a.seed)
        except (ValueError, OSError) as e:          # a bad scene file, a camera the sim cannot place
            logging.error("cannot build the simulated rig: %s", e)
            return None
        # the console only: the agent must find the objects with the cameras, as on the rig
        logging.info("sim scene %s, seed %d (repeat it with --seed %d): %s", sim.layout.name, sim.layout.seed,
                     sim.layout.seed, ", ".join(o.name for o in sim.layout.objects))
        return sim.arm_ios, sim.cam_ios, sim.limits, sim
    if a.mock:
        from .sim import nominal_limits
        cfg.backend = "mock"
        return ({n: MockArmIO(n) for n in cfg.arms}, {n: MockCameraIO(n, c.width, c.height) for n, c in cfg.cameras.items()},
                nominal_limits(cfg.arms), None)
    held = device_holders([ac.port for ac in cfg.arms.values()] + [c.path for c in cfg.cameras.values()])
    if held:
        for path, pids in held.items():
            logging.error("%s is held by pid %s", path, ", ".join(map(str, pids)))
        logging.error("robotd must be the only process on the arms and cameras (stop teleop/record/policy first)")
        return None
    return ({n: ArmIO(n, ac.port, ac.calibration_id) for n, ac in cfg.arms.items()},
            {n: CameraIO(n, c.path, c.width, c.height, c.fps) for n, c in cfg.cameras.items()}, None, None)


def main() -> int:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    done = threading.Event()               # first: a stop during start-up still shuts down in order (planning workers)
    running = threading.Event()            # set once start-up is over and robotd serves

    def on_signal(*_):
        if done.is_set() and not running.is_set():
            os._exit(130)                  # a second Ctrl-C during start-up (one that hangs): exit now, nothing moves yet
        done.set()                         # once running, further signals never cut a halt's deceleration short
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    cfg = load_config(a.config)
    if a.enable_motion:
        cfg.motion_enabled = True
    if a.no_cameras:
        cfg.cameras = {}
    b = backends(a, cfg)
    if b is None:
        return 2
    arm_ios, cam_ios, limits, sim = b

    cam_model = None
    if cfg.overlay and cfg.overlay.camera in cam_ios:
        try:
            cam_model, src = cfg.rig.camera_model(cfg.overlay.camera)
            if "estimated" in src.values():
                logging.warning("%s camera: %s (robot.yaml's numbers, no calibration file): positions from it are "
                                "approximate (docs/CALIBRATION.md)", cfg.overlay.camera,
                                ", ".join(f"{k} {v}" for k, v in src.items()))
        except Exception as e:
            logging.warning("overlay disabled: %s", e)

    rd = RobotD(cfg, arm_ios, cam_ios, cam_model, limits=limits, compute_workers=2)   # planning off the loop's process
    try:                                   # bind first: a port clash must fail before any arm is touched
        api = serve(rd, cfg.api.host, cfg.api.port, operator=False)
        op = serve(rd, cfg.operator.host, cfg.operator.port, operator=True)
    except OSError as e:
        logging.error("cannot bind the robotd ports (%s): is another robotd running?", e)
        if sim is not None:
            sim.stop()
        return 2
    rd.start()
    logging.info("robotd %s: operator http://%s:%d/  api http://%s:%d/state  log %s",
                 ("MOCK" if a.mock else "SIM" if a.sim else "") + (" motion ENABLED" if cfg.motion_enabled else " read-only"),
                 cfg.operator.host, cfg.operator.port,
                 cfg.api.host, cfg.api.port, rd.events.path)
    for name, st in rd.arms.items():
        logging.info("%s: connected=%s fault=%s torque=%s", name, st.connected, st.fault, st.torque)

    running.set()
    done.wait()
    def moving() -> bool:                                  # running, or starting (stiffening, its launch check)
        with rd._lock:                                     # the loop changes the motions: read them under its lock
            return any(m.status in ("running", "starting") for m in rd._active())
    with rd._lock:                                         # no new motion from here on (the halt's own deceleration
        rd.cfg.motion_enabled = False                      # is launched without that check)
    if moving():                                           # a motion running: stop it as the operator's STOP does,
        logging.info("halting the running motion before stopping")   # so the arm holds and the log says so
        rd.events.write("shutdown_halt", reason="robotd is stopping")
        rd.halt("both")
        t0 = time.monotonic()
        while moving() and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
    logging.info("stopping (the arms keep their torque: they hold where they are)")
    api.shutdown(); op.shutdown()
    rd.stop()
    if sim is not None:
        sim.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
