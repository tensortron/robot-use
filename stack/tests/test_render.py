"""The simulated rig's rendered cameras (stack/twin_mujoco/render.py): placed and projecting as the real ones.

Rendering needs a GL backend chosen before mujoco is imported, so the checks that render run in a child process
with the setting `find_gl_env()` finds; they are skipped (with the reason) where none works.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
try:
    from stack.twin_mujoco.scene import assets_dir
    assets_dir()
except FileNotFoundError:
    pytest.skip("SO-101 MuJoCo assets not fetched (tools/fetch_so101_assets.py)", allow_module_level=True)

REPO = Path(__file__).resolve().parents[2]
TOL_PX = 0.75                      # a marker's rendered centroid vs the model's projection of its centre
TOL_MEAN_PX = 0.45                 # and the mean per camera: without the half-pixel raster term the ego's markers
                                   # miss by 0.6-0.7 px each, under TOL_PX (measured: worst 0.46, ego mean ~0.2)


def _render_checks() -> dict:
    """Runs in the child process (GL chosen by its environment). Returns measurements as JSON-able data."""
    from stack.rig.camera_model import CameraModel
    from stack.rig.kinematics import ArmKinematics
    from stack.rig.rig import load_rig
    from stack.robotd.config import load_config
    from stack.robotd.safety import ArmLimits, ArmModel, rest_pose
    from stack.tests.conftest import CALIBRATED
    from stack.twin_mujoco.render import SimRenderer, sim_cameras
    from stack.twin_mujoco.scene import build_scene
    from stack.twin_mujoco.sim import SimWorld

    rig, cfg = load_rig(), load_config()
    sizes = {n: (c.width, c.height) for n, c in cfg.cameras.items()}
    cams = sim_cameras(rig, sizes)
    ego = CameraModel.from_files(rig.camera_file("ego", "intrinsics_file"), rig.camera_file("ego", "extrinsics_file"))

    # the right arm raised, looking forward and down; the left at rest
    q_right = [-10.0, -10.0, 20.0, 50.0, 30.0, 40.0]
    kin = ArmKinematics(rig.arm_mount("right"))
    G = kin.frames(np.array(q_right[:5]))[kin._joint_pos[4]]                     # gripper_link (robotd's FK)
    c = rig.camera("wrist_right")
    x, y, z, w = c["rot_xyzw"]
    R_opt = G[:3, :3] @ np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    C = G[:3, 3] + G[:3, :3] @ np.asarray(c["offset_m"], float)
    wrist_pts = [(dx, dy, 0.18) for dx, dy in ((0.0, -0.03), (0.06, -0.05), (-0.07, -0.02), (0.03, 0.0))]
    ego_pts = [(0.20, 0.0, 0.03), (0.30, 0.15, 0.03), (0.36, 0.26, 0.03), (0.30, -0.30, 0.03), (0.15, 0.08, 0.03), (0.32, -0.05, 0.10)]
    spheres = ([("ego", p, p) for p in ego_pts]
               + [("wrist_right", tuple(C + R_opt @ np.array(o)), o) for o in wrist_pts])
    objects = [{"name": "block", "size": [0.04, 0.04, 0.04], "xy": [0.24, 0.10], "rgba": [0.1, 0.35, 0.85, 1]}]
    # markers: 8 mm cubes held in the air (the world is never stepped here, so nothing falls)
    markers = [{"name": f"marker_{i}", "size": [0.008] * 3, "xyz": list(p), "rgba": [1, 0, 0, 1]}
               for i, (_, p, _) in enumerate(spheres)]
    spec_scene = build_scene(rig, objects + markers, cameras=cams)
    m = spec_scene.model
    start = {"left": rest_pose(ArmModel("left", ArmKinematics(rig.arm_mount("left")), ArmLimits(dict(CALIBRATED["left"]))),
                               cfg.safety), "right": q_right}
    world = SimWorld(spec_scene, start=start)
    marker_geoms = [int(np.flatnonzero(m.geom_bodyid == spec_scene.objects[f"marker_{i}"]["body"])[0])
                    for i in range(len(spheres))]
    out = {"spheres": [], "block": None}
    r = SimRenderer(world, cams, fps=50)
    r.start()
    try:
        def grab(cam):
            time.sleep(0.03)                                   # past the render period: latest() renders afresh
            f, _ = r.latest(cam)
            assert f is not None, r.error or "no frame within FIRST_FRAME_WAIT_S"
            return f.astype(int)

        m.light_castshadow[:] = 0                             # a marker's shadow would pull its centroid
        rgba0 = m.geom_rgba.copy()
        for g in marker_geoms:
            m.geom_rgba[g, 3] = 0.0
        base = {cam: grab(cam) for cam in ("ego", "wrist_right")}
        for i, (cam, p, ref) in enumerate(spheres):
            m.geom_rgba[marker_geoms[i], 3] = 1.0
            diff = np.abs(grab(cam) - base[cam]).sum(axis=2) > 40
            m.geom_rgba[marker_geoms[i], 3] = 0.0
            ys, xs = np.nonzero(diff)
            if cam == "ego":
                exp = ego.project([p])[0][0]
            else:
                K = next(cc for cc in cams if cc.name == cam).K
                exp = [K[0, 0] * ref[0] / ref[2] + K[0, 2], K[1, 1] * ref[1] / ref[2] + K[1, 2]]
            out["spheres"].append({"cam": cam, "n": int(len(xs)), "got": [float(xs.mean()), float(ys.mean())] if len(xs)
                                   else None, "expected": [float(v) for v in exp]})
        m.geom_rgba[:] = rgba0
        for g in marker_geoms:
            m.geom_rgba[g, 3] = 0.0

        # the block: robot_use's table_to_pixel of its top centre lands on it; and it moves when it is moved
        from robot_use import geometry
        no_block = m.geom_rgba.copy()
        bgeom = [g for g in range(m.ngeom) if m.geom_bodyid[g] == spec_scene.objects["block"]["body"]]
        with_block = grab("ego")
        for g in bgeom:
            m.geom_rgba[g, 3] = 0.0
        without = grab("ego")
        m.geom_rgba[:] = no_block
        mask = np.abs(with_block - without).sum(axis=2) > 40
        px = geometry.table_to_pixel([0.24, 0.10, 0.04])
        ys, xs = np.nonzero(mask)
        out["block"] = {"pixel": px, "on_block": bool(mask[int(round(px[1])), int(round(px[0]))]),
                        "centroid": [float(xs.mean()), float(ys.mean())], "n": int(len(xs))}
        with world.lock:
            a = spec_scene.objects["block"]["qpos"]
            world.data.qpos[a:a + 2] = [0.30, 0.22]
        moved = grab("ego")
        px2 = geometry.table_to_pixel([0.30, 0.22, 0.04])
        out["moved"] = {"old_place_changed": bool(np.abs(moved - with_block)[int(round(px[1])), int(round(px[0]))].sum() > 40),
                        "new_place_changed": bool(np.abs(moved - with_block)[int(round(px2[1])), int(round(px2[0]))].sum() > 40)}
        out["error"] = r.error
        out["scene_cameras"] = m.ncam
    finally:
        r.stop()
    return out


@pytest.fixture(scope="module")
def rendered():
    from stack.twin_mujoco.render import find_gl_env

    env = find_gl_env()
    if env is None:
        pytest.skip("no OpenGL backend renders here (python -m stack.twin_mujoco.render; see its docstring)")
    code = ("import json, os, sys; sys.path.insert(0, 'lib'); from stack.tests.test_render import _render_checks; "
            "print('RESULT ' + json.dumps(_render_checks()), flush=True); os._exit(0)")
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**os.environ, **env}, capture_output=True,
                       text=True, timeout=300)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, p.stdout[-2000:] + p.stderr[-4000:]
    return json.loads(line[7:])


def test_rendered_cameras_project_as_the_camera_models(rendered):
    """Ego: the calibrated model with its distortion. Wrist: the URDF camera on robotd's FK, pinhole."""
    assert rendered["scene_cameras"] == 3
    errs = {}
    for s in rendered["spheres"]:
        assert s["n"] > 5, s
        err = np.hypot(s["got"][0] - s["expected"][0], s["got"][1] - s["expected"][1])
        assert err < TOL_PX, s
        errs.setdefault(s["cam"], []).append(err)
    assert set(errs) == {"ego", "wrist_right"}
    for cam, e in errs.items():
        assert np.mean(e) < TOL_MEAN_PX, (cam, e)


def test_table_to_pixel_lands_on_a_rendered_block_and_frames_follow_the_scene(rendered):
    b = rendered["block"]
    assert b["n"] > 200 and b["on_block"], b
    assert np.hypot(b["pixel"][0] - b["centroid"][0], b["pixel"][1] - b["centroid"][1]) < 15, b
    assert rendered["moved"] == {"old_place_changed": True, "new_place_changed": True}
    assert rendered["error"] is None


def test_without_gl_each_camera_serves_a_frame_saying_so(monkeypatch):
    import mujoco

    from stack.rig.rig import load_rig
    from stack.twin_mujoco.render import SimCameraIO, SimRenderer, sim_cameras
    from stack.twin_mujoco.scene import build_scene
    from stack.twin_mujoco.sim import SimWorld

    def no_gl(*a, **k):
        raise RuntimeError("no OpenGL context")

    monkeypatch.setattr(mujoco, "Renderer", no_gl)
    rig = load_rig()
    cams = sim_cameras(rig, {"ego": (1280, 720), "wrist_left": (1024, 576)})
    r = SimRenderer(SimWorld(build_scene(rig, cameras=cams)), cams)
    io = SimCameraIO(r, "wrist_left")
    io.start()
    try:
        f, ts = io.latest()
        assert f.shape == (576, 1024, 3) and time.time() - ts < 5
        assert "no OpenGL context" in io.error
    finally:
        io.stop()


def test_sim_cameras_follow_the_arms_configured():
    from stack.rig.rig import load_rig
    from stack.twin_mujoco.render import sim_cameras

    sizes = {"ego": (1280, 720), "wrist_left": (1024, 576), "wrist_right": (1024, 576)}
    names = [c.name for c in sim_cameras(load_rig(), sizes, arms=("right",))]
    assert names == ["ego", "wrist_right"]
    with pytest.raises(ValueError, match="calibrated size only"):    # the stack's geometry is the calibrated model's
        sim_cameras(load_rig(), {"ego": (640, 360)})


class _FakeRenderer:
    """Stands in for mujoco.Renderer where GL is not needed: a black frame of the canvas size."""
    def __init__(self, model, height, width):
        self.shape = (height, width, 3)

    def update_scene(self, data, camera):
        pass

    def render(self):
        return np.zeros(self.shape, np.uint8)

    def close(self):
        pass


def test_the_shared_renderer_runs_while_any_camera_is_started(monkeypatch):
    import mujoco

    from stack.rig.rig import load_rig
    from stack.twin_mujoco.render import SimCameraIO, SimRenderer, sim_cameras
    from stack.twin_mujoco.scene import build_scene
    from stack.twin_mujoco.sim import SimWorld

    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)
    rig = load_rig()
    cams = sim_cameras(rig, {"wrist_left": (320, 180), "wrist_right": (320, 180)})
    r = SimRenderer(SimWorld(build_scene(rig, cameras=cams)), cams, fps=20)
    a, b = SimCameraIO(r, "wrist_left"), SimCameraIO(r, "wrist_right")
    a.start(), b.start(), a.start()                                  # a second start of one camera counts once
    try:
        f, t0 = b.latest()
        assert f.shape == (180, 320, 3) and b.error is None
        a.stop()                                                     # one camera stopped: the other still renders
        time.sleep(0.1)
        f, t1 = b.latest()
        assert t1 > t0 and b.error is None and r._thread.is_alive()
    finally:
        b.stop()
    assert not r._thread.is_alive()
    assert "stopped" in b.error and "stopped" in a.error             # and it says so, rather than freezing
    f, _ = b.latest()
    assert f.shape == (180, 320, 3)
    b.start()                                                        # a stopped renderer stays stopped
    assert not r._thread.is_alive()
