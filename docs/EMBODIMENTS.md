# Beyond the SO-101: a parts-based design

**A proposal, not built.** Today the stack drives one robot: two SO-101 arms on a fixed table. This note records
how the agent-facing library and the robot description would extend to other bodies: other arms, a single arm, a
mobile base (AMR), a quadruped, wheeled and legged humanoids. It was written on 2026-10-14 with the public-repo
sprint's decisions D15 (the library renamed `robot_use`) and D16 (`rig.yaml` renamed `robot.yaml`), which prepare
for it. The work itself waits for a second embodiment.

## The idea

An agent drives a robot only by writing programs. A program asks for one **step** at a time; a safety daemon
(robotd) plans the step, checks it, runs it or refuses it with a reason, and the program decides what next. That
contract does not depend on the body. What depends on the body is the vocabulary of steps: an arm moves its tip, a
base drives to a pose, a legged robot walks. So the library splits into a **common layer** and **parts**.

## The common layer (any robot)

| Today | What it is | Why it carries over |
|---|---|---|
| `runtime`: `step()`, `state()`, `log`, `done`, `give_up`, `StepFailed`, aborts | the program asks for a step; the daemon plans, checks and runs it, or refuses with the reason | every body takes "do this next, tell me how it went". Refuse-never-clip, budgets, traces and episodes sit in this contract |
| `checkpoint()` | pause with the robot held safe; the agent sees the cameras and its estimates, then continues, refines or aborts | the agent's loop, for any body |
| step results | done, condition met, refused (why), not settled; what happened (`grasped`, `contact`) | every motion ends in one of these |
| `conditions`: `contact()`, `elapsed()`, `joint_crosses()` | "stop when ..." attached to a step | general; each part adds its own (a base reaches its goal or bumps; a legged robot loses footing) |
| `estimate()`, `refine()` | an object's believed position, size and uncertainty, revised as the robot looks again | perception under uncertainty is the same problem everywhere |
| `geometry`: pixel to world and back | what the agent sees, turned into where things are | general, once frames are added (below) |
| `wait_until()`, `parallel()` | wait for something; several parts on one timeline | `parallel(left, right)` becomes coordination of any parts: two arms, an arm and a base, a humanoid's arms and legs |

**Frames.** Today everything lives in one fixed frame on the table. A robot that moves needs:
- a world (map) frame and the robot's own frame;
- the robot's pose, from localization, in `state()`;
- each camera's pose at the moment its frame was taken, since the cameras move with the robot;
- estimates tagged with the frame they are in.

`pixel_to_table` becomes `pixel_to_surface`. The fixed bimanual rig is the case where the robot never moves.

## Parts (per body)

| Part | Its steps | Bodies |
|---|---|---|
| **Arm** (today's `Arm`) | `move_to(pose)`, `move_by`, `move_joints`, `reach()`, `look_at()` with a wrist camera | single arm, bimanual, mobile manipulator, a humanoid's arms. **The arm declares its pose space**: the SO-101 is 5-axis (xyz, pitch, roll); a 6- or 7-axis arm takes a full orientation |
| **End effector** (today inside `Arm`: the 0-100 jaw gripper) | `open`, `close(effort)`; grasped, stalled | a parallel gripper; a suction cup (on/off); a dexterous hand (grasp types). Its own part, attached to an arm |
| **Mobile base** | `go_to(x, y, yaw)` in the map, `drive(v, w, t)`, `stop`; the localized pose | AMR, a wheeled humanoid's base, a mobile manipulator |
| **Legs** | body-level commands to the robot's walking controller: `walk_to`, a body velocity, stance height, step over | quadruped, biped humanoid. Programs never command leg joints: balance belongs to a controller (a learned policy or MPC, a "System 1"). Safety adds stability and fall detection |
| **Head** (camera mount) | `look_at(target)` | humanoids, pan-tilt cameras |
| **Whole body** | parts on one timeline | humanoids, mobile manipulators |

## Who decides which robot it is

The program never reads a configuration file. It asks robotd:

```
robot.yaml     what the body is: its parts (arms, end effectors, base, legs, head), their kinematics, mounts
               and cameras (named rig.yaml until D16)
robotd.yaml    how this machine is wired (ports, devices) and robotd's safety settings
      |  robotd loads both at start, with a backend: hardware, sim or mock
      v
robotd         plans, checks and runs; reports in /state which parts the body has, of what kind, and what
               each accepts (an arm's pose space, a gripper's range)
      |  through the runner
      v
robot()        builds an Arm, an EndEffector, a Base, ... from that report
```

```python
from robot_use import robot, estimate, checkpoint, contact, done

r = robot()
left, right = r.arms["left"], r.arms["right"]      # the SO-101 bimanual: two arms, a fixed base
# an AMR:       r.base.go_to(...)
# a quadruped:  r.legs.walk_to(...); r.arms["front"] if it carries one
```

What this keeps:
- **One program, many backends:** the same program runs on the hardware, the sim and the twin preview, because
  robotd answers the same way on each.
- **Withheld facts stay withheld:** `robot()` reports names, kinds and what each part accepts, never mounts,
  offsets or camera poses. The operating agent stays denied the description itself.
- **Skills say what they need:** a skill written as `r.arms["left"].move_to(...)` can check `r.arms` before it
  starts, and fails clearly on a body without that part.

## What stays per embodiment

The planning, IK and safety checks behind the steps are the body's own:
- `stack/rig/kinematics.py`: FK and IK from the URDF.
- `stack/robotd/planner.py`: paths.
- `stack/robotd/safety.py`: limits, floor and board, obstacles, the other arm, the gripper hulls.
- `stack/robotd/compute.py`: the planning workers.
- `stack/robotd/motion.py`: execution and the contact monitor.

A new kind of part needs its counterpart there: a base needs navigation and collision avoidance; legs need the
walking controller's interface and stability checks. The agent's side does not change: programs, checkpoints,
skills, episodes and the MCP tools.

## How to get there from today

1. **Names first** (before the public release): the library `so2x_lib` becomes `robot_use` (D15, task 8c);
   `rig.yaml` becomes `robot.yaml` (D16, task 8d).
2. **Arms as configuration:** the arm list and names come from the description, not from the two places that
   hard-code `left` and `right` (`lib/robot_use/arm.py`, `stack/robotd/config.py`). This is the public-repo sprint's
   D13 and task 8b (deferred).
3. **`robot()` and the parts report:** robotd's `/state` describes the parts. `robot()` builds them, and `arm(name)`
   stays as a shortcut so every existing program and skill keeps working.
4. **The end effector as its own part**, and the arm's pose space declared rather than assumed (pitch and roll).
5. **Frames:** the robot's pose in `state()`, camera poses per frame, estimates with their frame. Needed only once
   something moves.
6. **A second embodiment** drives the rest: its part type, its planner and safety checks, its description.

## Open questions

- **Parts naming:** do parts get names from the description (`"left"`, `"front"`), roles (`"primary"`), or both?
- **Coordination:** how much should `parallel` across parts promise, given that a base and an arm may have very
  different timing?
- **Legged bodies:** what does the program see when the walking controller refuses or stumbles? Is that a refused
  step, or a fault?
- **Skills across bodies:** can a skill learned on one arm transfer to another with a different pose space, or is a
  skill always tied to its body?
