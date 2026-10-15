<!-- The robot described in words, for agent prompts. Adapted 2026-10-01 from robots/so101.yaml `embodiment_text`
     (the Digital Twin), with the rig facts from robot.yaml (board, stand, frame); keep the numbers in step with robot.yaml.
     Joint directions verified on both real arms 2026-10-03 (rig checks 1-5). The gripper section below is generated
     by tools/gen_embodiment.py. -->

You control a pair of small SO-101 robot arms, side by side at the back edge of a table, both facing
forward over it.

**The rig.** The arms stand on an 18 mm mounting board; the work area is the table in front of it,
whose surface is height zero. The board's front edge is only 2.5 cm ahead of the arms' shoulders, so
almost everything the arms reach lies on the table itself. A thin camera column (4 by 2.5 cm) rises
from the board 7.5 cm behind the shoulders, midway between the arms, to a camera 71 cm above the
table that looks forward and down at about 65 degrees below horizontal.

**Coordinates** (metres): the origin is on the table surface midway between the two shoulders; x
points forward over the table, y to the robot's left, z up. The left arm's shoulder is at y = +0.186,
the right arm's at y = -0.186. Each arm reaches about 40 cm ahead. Each arm has a region of its own,
starting about 10 cm to its side of the midline, and there is a band in the middle that both can
reach, where one arm can hand an object to the other.

**Each arm** is a chain: a base plate on the board; a short column that turns the whole arm left and
right (shoulder_pan) and carries the shoulder motor; an upper arm that swings up and down at the
shoulder (shoulder_lift); a forearm that bends at the elbow (elbow_flex); a wrist that tips the
gripper up and down (wrist_flex) and twists it (wrist_roll); and a two-finger gripper (gripper).
Joint values are degrees, 0 at the middle of each joint's range. The gripper reads 0 with the fingers
touching and 100 fully open, about 8 cm apart at the tips.

**Directions** (from the robot's point of view): shoulder_pan negative turns toward the robot's left;
shoulder_lift negative swings the upper arm backward over the base, lifting the forearm and gripper,
positive swings it forward and down toward the table; elbow_flex positive bends the forearm down
toward the table; wrist_flex negative tips the gripper up; wrist_roll negative twists the gripper
clockwise as seen from behind it. With every joint at 0 the upper arm stands upright, the forearm
points horizontally forward, and the closed gripper is level, about 35 cm ahead of the shoulder and
24 cm above the table. Around shoulder_lift -100 with elbow_flex +94 the arm folds back over its base:
that is the rest pose.

**Gripper tip.** Positions you command are tip positions: the tip is a fixed point near the jaws. Where the
jaws are, open or closed, shows in the wrist view. "Pitch" is the angle the gripper points: 0 forward and
level, -90 straight down.

**Jaws.** One jaw is fixed, the other moves. At wrist_roll 0 the jaws open up and down, in the plane the arm
bends in: the fixed jaw is underneath and the moving jaw opens upward. At wrist_roll +90 or -90 they open
sideways. The table below gives the details for each roll.

**Cameras.** The ego camera above (1280 by 720) sees both base plates at the bottom of the frame, the
whole reachable table, and the space where the arms meet. Each gripper carries a wrist camera (1024 by
576) mounted just behind the fingers and looking along them: it sees the jaws at the bottom of its
image and whatever the fingers are about to touch beyond them.

<!-- BEGIN GENERATED gripper-camera: tools/gen_embodiment.py from the model; do not edit by hand -->

**Gripper, jaws and wrist camera at each wrist roll.** Positive wrist_roll turns the gripper
counter-clockwise as seen from behind it. The wrist camera is fixed to the gripper and turns with it.

| wrist_roll | jaws open | moving jaw | wrist camera | in the wrist image |
|---|---|---|---|---|
| 0 | up and down | on top | on the arm's left, 73 mm from the wrist axis | the robot's left is up, world up is right |
| -90 | sideways | on the arm's right | on top, 73 mm from the wrist axis | the robot's left is left, world up is up |
| +90 | sideways | on the arm's left | underneath, 73 mm from the wrist axis | the robot's left is right, world up is down |

The jaws are always at the bottom of the wrist image, whatever the roll. Robot programs use `WORK_ROLL` = -90
for approaches and grasps: the camera is on top and the jaws open sideways. At roll +90 the camera hangs below
the gripper, so the safety layer keeps the tip higher the flatter the gripper points: up to about 10 cm above
the surface under it when level (otherwise the tip point keeps at least 5 mm, as at the other rolls; other
parts of the gripper keep their own clearance, which can hold the tip higher). `look_at` uses roll -90 where
it can, else 0 or +90; its result says which. At rest the arm is folded at roll -90; the rest motion turns the
wrist only with the arm unfolded.

**How the jaws turn.** At wrist_roll -90 the jaws close across the arm's heading (the direction from the arm's
shoulder to the tip, seen from above), to within about 4 deg. With the gripper pointing steeply down (about
-70), increasing wrist_roll turns the jaws' closing direction counter-clockwise as seen from above, roughly
one degree per degree of roll (0.94 near -90; 183 deg from -90 to +90). As the gripper points flatter the roll
turns them less and also tilts them.

**The wrist image at any roll.** With the gripper pointing steeply down (about -70), "up" in the wrist image
points along the arm's heading turned by (wrist_roll + 90) deg counter-clockwise, seen from above: along the
heading at -90, to the heading's left at 0, back toward the robot at +90. That holds to within about 4 deg at
-90 and +90; in between the real image-up is up to about 12 deg clockwise of it, and more as the gripper
points flatter.

**The wrist camera's pose.** Its optical centre is about 105 mm behind the tip along the gripper and 73 mm off
the gripper's axis (on the side the table above gives for each roll); it looks 25 deg off the direction the
fingers point.

**Commanded and measured.** Under load, with the arm reaching out, the arm sags: the measured tip can sit 1-2
cm below the commanded one (measured on the rig). `state()` gives both: `tip_xyz_m` is where the arm is, and
while torque is on `command` is where robotd holds it.

<!-- END GENERATED gripper-camera -->
