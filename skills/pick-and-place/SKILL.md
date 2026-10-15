---
name: pick-and-place
description: "[draft] Pick a box-like object with one arm, jaws turned square to a pair of flat parallel faces (any yaw), and place it in a target by kind: container (cup, bowl - released just above the rim) or surface (lowered to contact, checked, released). A starter skill: validated on a reference rig for an upright ~3.5-4 cm block into a cup with either arm, not yet on this robot."
status: draft
---

# Pick and place (a starter skill)

**Source:** shipped with robot-use as a starter. It was validated on a reference SO-101 rig (two clean runs in a row,
one per arm, an upright block into a cup at a turned wrist) and runs in the simulated rig's start layout (an
unturned wrist, roll about -90). It has not
run on **this** robot yet: treat it as a draft, use it, and validate it here under the normal lifecycle (clean runs
and the user's approval).

**What it covers:** upright ~3.5-4 cm blocks at any yaw, KIND "container". **Not covered yet:** lying objects, other
shapes and sizes, KIND "surface", round objects; an object and a target on opposite sides of the table (one arm
cannot reach both: hand over, or nudge the object first); objects ~27 cm or more out at the midline (the edge of both
arms' reach: nudge them in first).

**Expect to refine** (within tolerance, at the skill's own checkpoints): the `jaw offset` at "pre-grasp" (see below),
the grip height at "at grasp height", and the target's position along the arm's heading at "over target" (rim-centre
pixels can read 1.5-2.5 cm short).

## What it does
1. Grasp roll: turns the wrist so the jaws close **along the gripped dimension** (square to the grasp faces), using
   `closing direction = heading + 90 + 0.94*(roll + 90)`. Upright cube: smallest turn (period 90). Lying or
   rectangular: period 180. Gives up if the roll leaves `ROLL_LIMITS` (a guess); then grip the other dimension or
   nudge/rotate the object first.
2. Hover, then pre-grasp 2 cm above the object, each a refine loop (3x). The tip point is the inner face of the
   fixed jaw, commanded `GAP` (3 mm) off one grasp face, plus the **`jaw offset`** estimate (starts at 0). A real arm
   can land off its command across the grasp (servo lag under load, calibration): if the wrist view shows the fixed
   jaw further off the face than 2-5 mm, refine `jaw offset` down by the excess; if it is on or over the face, up.
   On the reference rig it settled at about +0.02 m; in the simulated rig it is 0. Also check that the object's near
   end is well below the fingertip line in the wrist image (not level with or ahead of the fingertips).
3. **Grip height from the object as it rests** (`h`): the fingertips go LOW, at max(1.3 cm, h - 2.5 cm), so the
   contact band covers most of the grasp faces. It measures the arm's **sag** (commanded - measured tip z) and
   commands the height so the actual fingertips reach it. The other arm's wrist camera looks from the side
   (`SIDE_VIEW`); a checkpoint **"at grasp height"** before closing confirms the object is between the jaws, at the
   right height. If the safety check refuses the descent with the other arm that close, it rests the other arm and
   goes down without the side view.
3b. **Turned wrist** (roll > 10 deg from -90): every move uses the steepest pitch from -88 up to the base pitch that
   `reach()` confirms, so both jaws stay level. Unturned, a move whose base pitch is just out of reach (a high carry
   over a cup) takes the nearest reachable pitch within 16 degrees.
4. Descends slowly with `until=contact("light")`: if a jaw lands on the object it lifts and gives up rather than
   closing on it.
5. Close (effort 0.5), lift 5 cm, checkpoint "lifted" (square in the grip, not tilted).
6. Place by `KIND`:
   - `container`: refine loop with the object's bottom ~3 cm above the rim, lower to `CLEAR` (1.5 cm) above the rim,
     release.
   - `surface`: refine loop 3 cm above the support, lower slowly to contact, checkpoint "touching, still held", release.
   The release lets go in two stages: it eases the squeeze first (a few units, slowly) so the object drops
   straight down, then opens wide (opening wide at once dragged a squeezed block over the rim; opening only a little
   left it wedged between the jaws, and the lift carried it off). Then it lifts well above the rim, checks
   "released", closes the jaws to 20 and goes to rest (an open jaw can catch a cup's rim on the way).

## Parameters
`ARM`; the object's `xy`, `size=(a, b, h)` as it rests, `OBJ_YAW` (deg of edge a), `SQUARE`, `GRIP_ACROSS`
("a"/"b"), `GRIP_Z`; `KIND`, the target's `xy`/`size`, `TOP_Z`, `CLEAR`; `PITCH` (-70; -66 at the reach edge),
`PLACE_PITCH` (-65), `GAP`, `ALONG`; the `jaw offset` estimate's starting value.

## Preconditions
- Run `check_reach` on the hover, pre-grasp, grasp and place points first.
- Grasp faces flat, parallel, <= ~5 cm apart (opening 80 is ~6.4 cm, minus the gaps). Never grip end-to-end through
  a round stud.

## What was learned (on the reference rig and in the sim)
- **Grip the body's two flat side faces**, jaws square to them, low: a grip spanning a stud, or along a top edge,
  squeezed the block out or let it pivot during the lift.
- **Jaw alignment**: at roll -90 the jaws close across the heading; a block turned ~28 deg to that missed. Turning
  the roll by the yaw is the fix (the yaw is an estimate, refinable at hover/pre-grasp).
- **Sag**: 1-2.3 cm at x 0.25-0.27 on the reference rig; hence the measured-sag compensation.
- **Lag**: one shoulder_pan lagged 2-4.6 deg under load (tip 1-2.6 cm off its command). Judge the real position from
  the images, not the command; small refines can stay inside the servo's deadband.
- **Landing push**: a descent with the fingertips short, or a jaw on the object's edge, pushed it. Hence the
  along-heading check and the contact-stop descent.
- **Cup drop**: a short drop (1.5 cm above the rim) is gentler than releasing high.
- **Side view**: at pitch -88 the wrist view's vertical axis is mostly height, not overlap; the other arm's side
  view decides whether the object is between the jaws. A refine of more than ~1 cm from one wrist view is unreliable
  (parallax).
