# Calibrating a rig

Calibration tells the stack where everything is: each arm's zero and the range of its joints, where the arms
stand, the board and the camera stand, and where the ego camera is and how its lens bends the image. robotd's
safety checks, the overlay on the operator page and the agent's `pixel_to_table` all depend on it.

**The quick way:** one tool walks you through every step in the browser and writes the results:

```bash
.venv/bin/python tools/calibrate.py          # then open http://127.0.0.1:8098/
# from a laptop:  ssh -L 8098:localhost:8098 <user>@<robot-host>   then http://localhost:8098/
# or on the LAN (no login: a network you trust):  --host 0.0.0.0   then http://<robot-host>:8098/
```

Stop robotd first (`tools/run_rig.py` included), because it holds the arms and the cameras. Power the arms (12 V)
before the USB hub. The tool remembers what is done, so you can stop and come back later; each step can be redone.
`--camera-only` leaves the arm steps out and never opens an arm's serial port: the measurements and the camera
steps only, with the arms off if you like. `--trial` writes everything into a folder of its own instead (in `$ROBOT_USE_HOME/calibration/`), and gives the
servos their old calibration back.

**What you need besides the rig:**
- **The checkerboard**, for the lens and pose steps: print
  [`assets/checkerboard_9x6_25mm.pdf`](../assets/checkerboard_9x6_25mm.pdf) on A4 landscape at 100% (not "fit to
  page"). Measure a square: it must be 25 mm (otherwise pass `--squares-mm`). Stick it to something flat and stiff,
  such as foam board: a curled sheet gives a wrong lens.
- **A tape measure**, in metres from the rig's origin (on the table, midway between the arms' pan axes).
- **Optionally, the camera's datasheet**, for its horizontal field of view: a first estimate before the lens step.

The rest of this page says what each step does, what it produces and where that goes, and how to do it by hand.

## What you end up with

| Step | Produces | Where it goes | Used by |
|---|---|---|---|
| 1. Arms | each joint's zero (homing offset) and range | in the servos, and lerobot's file `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<calibration_id>.json` | robotd (joint limits, degrees), teleoperation |
| 2. Measurements | the distance between the arms' pan axes; the board; the camera stand | `robot.yaml`: `arms`, `board`, `obstacles` | robotd's collision checks, the overlay, the sim |
| 3. Camera estimate | the camera's position, tilt, yaw and field of view, from a tape and a datasheet | `robot.yaml`: `cameras.ego` | the overlay, `pixel_to_table`, the sim's camera |
| 4. Lens | focal length, centre and distortion | `calib/ego_intrinsics.json`, named in `robot.yaml` | the same, exactly |
| 5. Camera pose | where the camera is, measured from one photo | `calib/ego_extrinsics.json`, named in `robot.yaml` | the same, exactly |
| 6. Check | — | — | — |

The camera steps don't need the arms: they can be off. After step 3 the stack runs. Steps 4 and 5 make the camera exact, and a file named in `robot.yaml` always wins over
the estimate beside it. The operator page and `tools/doctor.py` say when the camera is still an estimate.

**How good is an estimate?** If the numbers come from a calibrated camera, an estimate built from them is within a
millimetre on the table. A tape measure and a datasheet's field of view are rougher. On this rig's wide lens, the
lens's distortion alone puts positions about 5 cm off at the edges, even with the pose exactly right. That is
enough to start with and to check the rig, but grasps need the lens step. The arms' safety does not depend on the
camera: robotd checks every motion against the arms' own readings and `robot.yaml`'s geometry.

## 1. The arms

lerobot's calibration, for each follower. Its id is `calibration_id` in `robot.yaml` (`bi_follower_left`,
`bi_follower_right`).

**Already calibrated with lerobot?** Keep it. The tool lists the SO-101 calibration files it finds in lerobot's
calibration folder, under any id or robot type. Choose yours with "Use": the tool sets the arm's `calibration_id` in
`robot.yaml`, copying the file into `robots/so_follower/` if it is elsewhere, and writes nothing to the servos. By hand:
set `calibration_id` to your file's name. robotd checks at its start that the servos hold the file's limits.

1. **Hold the arm, then turn torque off.** It falls under gravity.
2. **Move it to the middle of its range:** each joint about halfway between its two ends, the gripper half open.
   That pose becomes the zero (lerobot writes a homing offset into each servo).
3. **Move every joint except the wrist roll through its whole range**, end to end, and open and close the gripper
   fully. The extremes become the joint's limits. The wrist roll turns all the way round, so it keeps the full
   range.

The tool refuses to finish until each joint has moved at least about 26°. By hand, from the bimanual robot (the
files land where robotd looks):

```bash
lerobot-calibrate --robot.type=bi_so_follower --robot.id=bi_follower \
  --robot.left_arm_config.port=/dev/so101_follower_left \
  --robot.right_arm_config.port=/dev/so101_follower_right
```

**Recalibrating an arm moves its zero.** Then check the overlay (step 6). If the drawn arm sits off the real one,
`joint_offsets_deg` in `robot.yaml` need measuring again.

## 2. Measurements

Measure with a tape, in metres, in the rig frame. The origin is on the table, midway between the two arms'
shoulder-pan axes; +x is forward, +y is the robot's left, +z is up.

| `robot.yaml` | What | Typical |
|---|---|---|
| `arms.pan_axis_separation_m` | pan axis to pan axis | 0.37 |
| `board.thickness_m` | the board the arms stand on | 0.018 |
| `board.front_edge_x_m` | the board's front edge, ahead of the pan-axis line | 0.025 |
| `board.side_margin_m` | each side edge, outboard of its pan axis | 0.064 |
| `obstacles.camera_stand.front_face_x_m` | the stand's front face (negative: behind the pan-axis line) | -0.075 |
| `obstacles.camera_stand.depth_x_m`, `width_y_m` | the stand's size | 0.04, 0.025 |
| `obstacles.camera_stand.centre_y_m` | its centre (0 = midway between the arms) | 0 |
| `obstacles.camera_stand.top_z_m` | its top, above the table | 0.70 |

**A new rig:** `robot.yaml` comes with the joint offsets (`joint_offsets_deg`, `base_z_offset_m`) measured on the rig
it was written for. Those belong to those arms. Reset them to 0 (the tool's "a new rig" box does it), then measure
yours in step 6.

Also check `obstacles.camera_head` (a box around the camera itself), and each arm's `yaw_deg` (0 when it is bolted
square to the board). Every value in `robot.yaml` has a `sources` line saying how it was measured: change both
together.

## 3. The camera estimate

In `robot.yaml`, under `cameras.ego`:

| Key | What |
|---|---|
| `eye_m` | the centre of the lens, `[x, y, z]` in the rig frame |
| `tilt_down_deg` | from straight down: 0 looks at the table below it, 90 looks level |
| `yaw_deg` | + toward the robot's left (0 looks straight forward) |
| `roll_deg` | about the optical axis (usually 0) |
| `fov_h_deg` | the horizontal field of view, from the datasheet |
| `distortion` | the lens numbers (`focal_px`, `principal_point_px`, `coeffs`), when known: they win over `fov_h_deg` |

Without `intrinsics_file` and `extrinsics_file`, the stack builds the camera from these. In the tool, the board, the
stand and the pan axes are drawn on the live image from the numbers as you type them. Adjust the numbers until the
drawing lies on the real rig.

If your camera is the same model as another calibrated rig's, keep that rig's `distortion` numbers and measure only
the pose: the lens is the larger error.

## 4. The lens

With the checkerboard printed and mounted (see "What you need" above), set the camera's focus and do not touch it again: the lens calibration belongs to that focus and
that resolution.

The tool asks for the board in each ninth of the image at three tilts (27 frames). It takes each frame by itself
when the board is still and sharp, then solves. It refuses a fit worse than 1 px, or one where the board was never
tilted beyond 20°: flat-on views cannot tell focal length from distance, and they give a confidently wrong lens.

By hand: `tools/live_view.py --capture-to <folder>` (the same guided capture), then
`tools/calibrate_camera.py --frames <folder> --squares-mm 25`. The output is `intrinsics.json`.
[camera-calibration.md](camera-calibration.md) is the full record of how this rig's lenses were measured, and why.

## 5. The camera pose

Lay the board flat on the table in front of the arms: pattern up, square to the arms, all of it in view, weighed
down. Take the photo. The tool circles one corner and draws an arrow along the board. Measure the circled corner's
x and y from the rig origin, and say which way the arrow points on the table. The solve gives the camera's position
and angles, and it warns when the result is implausible for a camera between the arms (usually a wrong arrow).

By hand: `tools/solve_extrinsics.py` (two passes: look, then solve with `--corner-x --corner-y --board-yaw-deg`).
The pose changes whenever the camera or the arms move; the lens does not.

## 6. Check

```bash
.venv/bin/python tools/run_rig.py            # read-only: nothing moves
```

On the operator page's ego view, check three things:
- the drawn arms lie on the real ones;
- the yellow outline lies on the board;
- the red boxes lie on the camera stand.

Then hold each arm in three poses (rest, reaching forward, reaching to the centre) and compare the drawn tip with
the real one. They should agree to about 10 mm. A constant offset means an arm's mount (`yaw_deg`, the separation)
is wrong. An offset that grows as one joint moves means that joint's `joint_offsets_deg` is wrong. Measure the
joint's pivots with a tape in two poses and fit the offsets that put the model's pivots there: that is how this
rig's offsets in `robot.yaml` were found (their `sources` lines say how).

The wrist cameras need no calibration: their pose comes from the arm's URDF, and the stack measures nothing with
them.
