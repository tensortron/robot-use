# robot_use: the API for robot programs

Robot programs are Python that run in a sandbox. They act on the robot **only** through this library. Every
motion is checked by the safety layer before it moves, and is rejected with a reason rather than clipped.

```python
from robot_use import arm, estimate, geo, contact, checkpoint, log, done, give_up, StepFailed, WORK_ROLL
```

Allowed imports: `robot_use`, `math`, `numpy`, `time`, `json`, `random`, `itertools`, `functools`,
`collections`, `dataclasses`, `typing`, `statistics`, `enum`, `copy`, `re`. No files, network or devices.

## Frame and units

- **Rig frame**: origin on the table surface midway between the two shoulders. +x forward over the table,
  +y to the robot's left, +z up. Metres. The left shoulder is at y = +0.186, the right at y = -0.186.
- **Joints**: degrees (0 = middle of range). **Gripper**: 0 closed .. 100 open (about 8 cm at the tips);
  the safety layer accepts 0..98.
- **pitch**: the gripper's approach angle. 0 = forward and level, -90 = straight down. **roll**: the
  wrist_roll joint in degrees. **`WORK_ROLL` = -90** is the recommended roll for approaches and table grasps:
  the wrist camera sits on top of the gripper and the jaws open sideways. At roll +90 the camera hangs
  underneath the gripper, and low poses are refused because the camera would hit the table. `move_to(roll=None)` keeps the
  current roll, so pass `roll=WORK_ROLL` explicitly. `look_at` tries -90 first, then 0 and +90 (the camera beside or underneath is fine high above the table).
  The wrist never turns while the arm is folded near its rest: a `move_to` / `move_joints` that would (out of the
  rest, or through the fold) first goes out to a pre-rest pose, turns the wrist there, then continues - expect
  that detour; a straight-line move or a trajectory that would is refused.
- **speed**: a fraction (0, 1] of the joint speed caps.

## Arms: `arm("left")`, `arm("right")`

Each method that moves is **one step**. It blocks until the motion ends and returns a `StepResult`
(`.status`, `.condition_met`, `.grasped`, `.raw`), or raises `StepFailed` (`.result` holds the reason).

| call | effect |
|---|---|
| `move_to(xyz, pitch=None, roll=None, linear=False, speed=0.5, until=None)` | gripper tip to [x, y, z] |
| `move_by(dx=0, dy=0, dz=0, linear=True, speed=0.3, until=None)` | relative to where the tip is, keeping pitch and roll |
| `move_joints(speed=0.5, until=None, **joints)` | e.g. `move_joints(shoulder_pan=20)` |
| `open(width=95)` | open the gripper (0..98) |
| `close(effort=0.5)` | close (towards 0) until the jaws meet resistance at `effort` (0..1), then hold. `.grasped` is set if they stopped on something, None if they closed on air |
| `rest()` | slowly to the folded rest pose |
| `look_at(target, distance=0.15, speed=0.4)` | move so this arm's **wrist camera** looks straight at `target` (a point or an Estimate) from `distance` m; returns the viewpoint reached (`camera`, `elevation` and `distance` as achieved, `axis_miss_mm`, ..., `step`), camera on top (roll -90) where possible, else beside (0) or underneath (+90) - see `roll` in the result; if the move to one viewpoint is refused it tries the next; raises `NoViewpoint` if none is reachable and clear (not a failed step) |
| `reach(points)` | **no motion**: can the tip be at each point ([x, y, z] or {xyz, pitch?, roll?}), at which pitches, and is the pose clear |
| `state()`, `.tip`, `.joints`, `.gripper` | observation (not steps). `.tip` / `tip_xyz_m` is where the tip is (measured); `state()["command"]` is where robotd holds it (present while torque is on). Reaching out under load the arm sags, so the measured tip can sit 1-2 cm below the command |
| `target(xyz=..., pitch=..., gripper=...)` or `target(**joints)` | a target for `parallel` |

The wrist camera is tilted about 25 deg from the fingers, so pointing the jaws at something does not centre it
in the wrist image; `look_at` aims the camera itself.

`parallel(left_target, right_target, duration_s=2.0)` moves both arms as **one** motion, so their clearance
is checked along both paths.

## Conditions: `until=`

Evaluated by the robot every 20 ms during the step. The first one met ends the step with status
`condition_met` (`.condition_met` says which). A touch holds where it touched; the others decelerate. If
the motion ends without meeting it, the step is `done` and `.condition_met` is None.

`contact(level="light" | "firm", joints=None)` · `gripper_stalled(effort)` ·
`joint_crosses(joint, value, direction)` · `elapsed(seconds)`

An expected touch is **not** an obstruction. An unexpected one makes the step `blocked`: the arm stops,
backs off and holds, and its next motion needs human approval.

## Estimates

```python
block = estimate("red block", xy=geo.pixel_to_table(u, v, z=0.04)[:2], size=(0.04, 0.04, 0.04),
                 tol=0.03, source="ego pixel (u, v)")
block.top, block.centre, block.above(0.08), block.at(z)    # points for move_to
```

`tol` is how far a checkpoint may move the estimate without a new approval.

## Checkpoints

```python
decision = checkpoint("hover", block, views=["wrist_right", "ego"], need="block centred between the jaws")
```

The program pauses with the arms holding. The agent sees the named camera views (the ego view with the
robot's model drawn on it) and decides: **continue**, **refine** (the estimates are moved in place, each
within its `tol`; then repeat the alignment move), or **abort**. Put one where an estimate decides whether
the next step can work. For example, if the hover is not over the block, there is no point descending.

## Geometry: `geo`

| call | |
|---|---|
| `geo.pixel_to_table(u, v, z=0.0)` | where ego pixel (u, v) meets the horizontal plane at height z: the table (0) or an object's top. Full-resolution pixels (1280 x 720) |
| `geo.table_to_pixel(xyz)` | where a point appears in the ego image |
| `geo.frame_scale(width)` | factor from a downscaled frame's pixels to full resolution |
| `geo.ray(u, v)`, `geo.distance(a, b)` | |

## Waiting, logging, finishing

- `wait_until(predicate, timeout_s, poll_s=0.1)`: the arms hold while it polls. The timeout is required.
- `log(msg)` goes to the trace.
- `done(summary)` ends the program successfully; `give_up(reason)` ends it as failed, with the reason.
- `StepFailed` can be caught to try something else. Two failed steps in a row stop the program anyway.

## Limits the runner enforces

50 steps, 100 read-only `reach` / `look_at` queries, 180 s of motion, 900 s wall clock, 2 consecutive failed steps, 900 s at a checkpoint. A STOP
from the operator or the agent ends the program at once (`ProgramAborted`, which `except Exception`
cannot catch).
