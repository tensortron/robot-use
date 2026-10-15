import math
from robot_use import arm, estimate, contact, checkpoint, log, done, give_up, StepFailed, WORK_ROLL, NoViewpoint

# pick-and-place (a starter draft: validated on a reference rig for upright blocks into a container, not yet on this
# one; other cases untested - see SKILL.md).
# Grasp any box-like object across a pair of parallel flat faces, with the jaws turned to those faces;
# place it by target kind: "container" (release just above the rim) or "surface" (lower to contact, then release).

# ======================= parameters (fill in from the ego view) =======================
ARM = "right"
SHOULDER = (0.0, 0.186) if ARM == "left" else (0.0, -0.186)
A = arm(ARM)
VIEW = f"wrist_{ARM}"
OTHER_ARM = "right" if ARM == "left" else "left"
SIDE_VIEW = True    # the other arm's wrist camera watches the grasp height from the side (then rests before the carry)

# Object as it rests NOW (upright or lying). size = (a, b, h): a along OBJ_YAW, b across it, h = height as it rests.
obj = estimate("object", xy=[0.22, -0.10], size=(0.04, 0.04, 0.04), tol=0.03, source="ego top-face centre @ z=h")
OBJ_YAW = 0.0       # deg, direction of the 'a' edge seen from above, CCW from +x (two top corners -> pixel_to_table)
SQUARE = True       # True if a == b (footprint repeats every 90 deg -> smaller wrist turns)
GRIP_ACROSS = "a"   # jaws close across this dimension: the two flat PARALLEL faces it spans (<= ~5 cm; never a round stud end)
GRIP_Z = None       # ACTUAL fingertip height of the grip above the table; None -> max(GRIP_FLOOR, h - GRIP_DEPTH) (tips low).
                    # h is the height AS IT RESTS: a Duplo 1x1 is ~0.045 upright (with stud), ~0.035 lying on its side.

# Target. KIND "container": cup / bowl / basket.  KIND "surface": box top, top of another object, or a table region (TOP_Z 0).
KIND = "container"  # the validated case; "surface" is written but untested
target = estimate("target", xy=[0.27, 0.0], size=(0.085, 0.085, 0.05), tol=0.03, source="ego top-face centre @ z=TOP_Z")
TOP_Z = 0.05        # rim height (container) or support-surface height (surface)
CLEAR = 0.015       # container: object bottom this far above the rim at release (gentle, short drop)

PITCH = -70         # grasp pitch; at the edge of reach use -66 (check_reach first)
PLACE_PITCH = -65
GAP = 0.003         # fixed jaw to the grasp face before closing: almost touching, so the moving jaw PINCHES the object
                    # instead of shoving it across the table (2026-10-14: with ~1.3 cm the lying block spun out while sliding)
ALONG = 0.0         # extra tip shift along the arm's heading (+ = further out); stretched arms land short
K_ROLL = 0.94       # jaw turn per degree of wrist roll at steep pitch (embodiment.md)
ROLL_LIMITS = (-150.0, -30.0)   # GUESS: keep the camera above/beside the gripper near the table
ROLLED_DEG = 10.0   # a roll further than this from -90 counts as 'turned'
ROLLED_PITCH = -88  # near-vertical gripper for turned grasps/places (check_reach: -90 is often out of reach high up)

OBJ_A, OBJ_B, OBJ_H = obj.size
W = OBJ_A if GRIP_ACROSS == "a" else OBJ_B
# The fingers come from above: they touch the object only between their tips and its top. Put the tips LOW so the
# contact band covers most of the grasp faces (>= 2.5 cm, or all of a shorter object), keeping ~8 mm off the table.
# Fits the successful upright runs (h 4.5 -> 0.020) and fixes the lying-block slip (h 3.5: was 0.019 -> band 1.1 cm).
GRIP_DEPTH = 0.025
GRIP_FLOOR = 0.013  # closing at pitch -88 sweeps the jaws ~8 mm below the tip: at 0.010 the safety layer refused (1.7 mm)
GZ = GRIP_Z if GRIP_Z is not None else max(GRIP_FLOOR, OBJ_H - GRIP_DEPTH)
# The tip point is the inner face of the fixed jaw. A real arm can land off its command across the grasp (servo lag
# under load, calibration): 'jaw offset' is that correction (metres, first number; + moves the fixed jaw AWAY from the
# object). Start at 0 (right in the simulated rig); refine it at the pre-grasp checkpoint from the wrist view. On the
# reference rig it settled at about +0.02.
jaw = estimate("jaw offset (m, first number)", xy=[0.0, 0.0], tol=0.03, source="0: the tip is the fixed jaw's inner face")


JAW_LIMITS = (-W / 2, 0.04)    # beyond these the commanded fixed jaw lands on the object, or far off it: re-estimate


def jaw_off():
    j = jaw.xy[0]                             # refines add up (each within tol, from the current value): bound it
    if not (JAW_LIMITS[0] <= j <= JAW_LIMITS[1]):
        give_up(f"jaw offset {j:.3f} is outside {JAW_LIMITS}: re-estimate the object's position and rerun")
    return j


def grasp_off():
    return W / 2 + GAP + jaw_off()            # object centre -> tip at the grasp: the fixed jaw GAP off the face


def hold_off():
    return W / 2 + jaw_off()                  # held object centre -> tip: the face against the fixed jaw


def unit(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))


def heading(p):
    return math.degrees(math.atan2(p[1] - SHOULDER[1], p[0] - SHOULDER[0]))


def grasp_roll(p):
    """Wrist roll that makes the jaws close along the gripped dimension (square to the grasp faces)."""
    want = OBJ_YAW + (0.0 if GRIP_ACROSS == "a" else 90.0)      # closing direction we want
    base = heading(p) + 90.0                                     # closing direction at roll -90
    period = 90.0 if SQUARE else 180.0
    d = (want - base + period / 2) % period - period / 2
    return -90.0 + d / K_ROLL


def fixed_dir(p, roll):
    """Unit vector from the object toward the fixed jaw (the side the commanded tip is on)."""
    return unit(heading(p) + 90.0 + K_ROLL * (roll + 90.0))


def sag():
    s = A.state()
    c = s.get("command") or s
    return max(0.0, min(0.03, c["tip_xyz_m"][2] - s["tip_xyz_m"][2]))


# The yaw is an estimate too: refine its FIRST number (degrees) at a checkpoint when the wrist view shows the
# grasp faces turned against the jaws (2026-10-14: ego yaw was ~20 deg off).
yaw = estimate("object yaw (deg, first number)", xy=[OBJ_YAW, 0.0], tol=30.0, source="ego: two top corners")


def set_roll():
    global OBJ_YAW, R
    OBJ_YAW = yaw.xy[0]
    R = grasp_roll(obj.at(0.0))
    log(f"yaw {OBJ_YAW:.0f} -> grasp roll {R:.1f} (closing turned {K_ROLL * (R + 90):.1f} deg from roll -90)")
    if not (ROLL_LIMITS[0] <= R <= ROLL_LIMITS[1]):
        give_up(f"needed roll {R:.0f} is outside {ROLL_LIMITS}: grip the other dimension or nudge/rotate the object first")


R = -90.0
set_roll()
# Roll away from -90 at a non-vertical pitch TILTS the jaw plane: the moving jaw rides higher than the fixed one
# (2026-10-14: at roll -40, pitch -70 it passed over a 3.5 cm lying block). When the wrist is turned, point the
# gripper as steeply down as the arm can at each point, so the roll is (nearly) a pure turn and both jaws stay level.
# -88 is often out of reach higher up (2026-10-14: hover 9.5 cm and the cup heights unreachable at -88, steepest
# -80..-84; a straight lift at -88 was refused), so each move asks reach() for the steepest pitch that works.
ROLLED = abs(R + 90.0) > ROLLED_DEG
if ROLLED:
    log(f"rolled grasp: steepest reachable pitch from {ROLLED_PITCH} at each point")


def pitch_at(points, base):
    """A pitch at which EVERY point is reachable and clear at roll R (give both ends of a straight-line move), from
    reach() queries of at most 20 entries. Turned wrist: the steepest in [ROLLED_PITCH, base] (2-deg steps). Otherwise `base` if it works,
    else the nearest pitch to it within +-16 deg (a high carry over a cup can be just out of reach at `base`)."""
    if ROLLED:
        cands = list(range(int(ROLLED_PITCH), int(base) + 1, 2))
    else:
        cands = [base] + [base + sgn * d for d in (4, 8, 12, 16) for sgn in (1, -1) if -88 <= base + sgn * d <= -40]
    q = [{"xyz": list(x), "pitch": p, "roll": R} for p in cands for x in points]
    per = max(len(points), 20 // len(points) * len(points))   # whole candidates per query, at most 20 entries
    res = []
    for k in range(0, len(q), per):
        rows = A.reach(q[k:k + per]).get("points", [])
        if len(rows) != len(q[k:k + per]):
            break                                 # a failed query: judge only what came back
        res += rows
    for i, p in enumerate(cands):
        rows = res[i * len(points):(i + 1) * len(points)]
        if len(rows) == len(points) and all(r.get("reachable") and r.get("pose_ok", True) for r in rows):
            return p
    log(f"no pitch in {cands[0]}..{cands[-1]} reaches {points}; using {base}")
    return base


def grasp_tip(z):
    p = obj.at(z)
    f, h = fixed_dir(p, R), unit(heading(p))
    return [p[0] + grasp_off() * f[0] + ALONG * h[0], p[1] + grasp_off() * f[1] + ALONG * h[1], z]


def place_tip(z):
    p = target.at(z)
    f = fixed_dir(p, R)
    return [p[0] + hold_off() * f[0], p[1] + hold_off() * f[1], z]


# ---------------------------------------------------------------- grasp
A.open(80)
for i in range(3):
    set_roll()
    A.move_to(grasp_tip(OBJ_H + 0.05), pitch=pitch_at([grasp_tip(OBJ_H + 0.05)], PITCH), roll=R, speed=0.3)
    if checkpoint(f"hover {i+1}", obj, yaw, views=[VIEW, "ego"],
                  need="jaws square to the grasp faces (else refine 'object yaw'); object on the moving-jaw side") != "refine":
        break
for i in range(3):
    set_roll()
    # the pre-grasp pitch must also hold at grasp height: the descent is a straight line at one pitch
    GP = pitch_at([grasp_tip(OBJ_H + 0.02), grasp_tip(GZ)], PITCH)
    A.move_to(grasp_tip(OBJ_H + 0.02), pitch=GP, roll=R, speed=0.3)
    if checkpoint(f"pre-grasp {i+1}", obj, yaw, jaw, views=[VIEW, "ego"],
                  need=("fixed jaw 2-5 mm off one grasp face (almost touching, NOT touching) and parallel to it - "
                        "further off: refine 'jaw offset' DOWN by the excess; on or over the face: UP; "
                        + ("the grasp face's near end >= 1.5 cm BELOW the fingertip line in the wrist image"
                           if GP > -80 else
                           "position ALONG the fingers is judged later from the side view (at pitch -88 the "
                           "wrist view's vertical axis is mostly height, not overlap)"))) != "refine":
        break
set_roll()
GP = pitch_at([grasp_tip(OBJ_H + 0.02), grasp_tip(GZ)], PITCH)

# Grip height is an estimate too (first number, metres): checked from the side at grasp height before closing.
grip = estimate("grip height (m, first number)", xy=[GZ, 0.0], tol=0.02, source=f"h={OBJ_H:.3f} as it rests - {GRIP_DEPTH}")
views = [VIEW, "ego"]
if SIDE_VIEW:
    try:
        arm(OTHER_ARM).look_at(obj.at(OBJ_H / 2), distance=0.22)
        views = [VIEW, f"wrist_{OTHER_ARM}", "ego"]
    except Exception as e:                    # NoViewpoint, or a robotd timeout (2026-10-14) - not worth ending the run
        log(f"no side view from the other arm ({e}); height check from the working wrist view only")

s = sag()
log(f"sag {s:.3f} -> grasp command z {grip.xy[0] + s:.3f}")
if len(views) == 3:
    # the side-view arm must not stand in the descent's way: ask reach() (pose checks include the other arm) at
    # points along the straight line down, rather than have the descent refused (a refused step spoils a clean run)
    z0, z1 = OBJ_H + 0.02, grip.xy[0] + s
    line = [grasp_tip(z0 + (z1 - z0) * k / 4) for k in range(5)]
    rows = A.reach([{"xyz": p, "pitch": GP, "roll": R} for p in line]).get("points", [])
    near = [q for r in rows for q in r.get("pose_problems", []) if f"{OTHER_ARM} arm" in q]   # "... from the left arm's jaws"
    if len(rows) != len(line):
        log("reach() did not answer for the descent; going down as planned (the fallback below still applies)")
    elif near:
        log(f"the side-view arm is too near the descent ({near[0]}); resting it, height check from the working "
            "wrist view only")
        arm(OTHER_ARM).rest()
        views = [VIEW, "ego"]
    elif not all(r.get("reachable") and r.get("pose_ok", True) for r in rows):
        log(f"the descent itself does not reach: {[r.get('pose_problems') or r.get('reachable') for r in rows]}")
try:
    r = A.move_to(grasp_tip(grip.xy[0] + s), pitch=GP, roll=R, linear=True, speed=0.12, until=contact("light"))
except StepFailed as e:
    if len(views) != 3 or e.result.get("status") != "rejected":   # only a refusal; a block, STOP or fault ends it
        raise
    # the other arm, watching from the side, can stand too near the descent for the safety check (2026-10-15, sim)
    log(f"descent refused with the side view out ({e}); resting the other arm and going down without it")
    arm(OTHER_ARM).rest()
    views = [VIEW, "ego"]
    r = A.move_to(grasp_tip(grip.xy[0] + s), pitch=GP, roll=R, linear=True, speed=0.12, until=contact("light"))
if r.condition_met:                           # a jaw landed on the object: closing now would push or tip it
    A.move_by(dz=0.03, speed=0.1)
    give_up("a jaw touched the object on the way down; re-estimate (pre-grasp view) and rerun")
# The SIDE view decides here: the object must be BETWEEN the jaws (one jaw on each side of it), not beside or past
# them. 2026-10-14: at pitch -88 the wrist view's 'below the fingertip line' cue was read as horizontal overlap; a
# 2.6 cm 'deeper' refine put both jaws ~3 cm past the block and they closed on air - visible in the side view.
for i in range(3):
    if checkpoint(f"at grasp height {i+1}", obj, grip, views=views,
                  need=f"SIDE VIEW: the object BETWEEN the two jaws (one each side), fixed jaw 2-5 mm off its face; "
                       f"fingertips low (~1 cm above the table, h {OBJ_H*100:.1f} cm as it rests), not on the table. "
                       "Refine 'object' (x, y) and/or 'grip height' (m) if not") != "refine":
        break
    s = sag()
    A.move_by(dz=0.02, speed=0.1)                       # never slide sideways at grasp height: up, over, down
    A.move_to(grasp_tip(grip.xy[0] + s + 0.02), pitch=GP, roll=R, speed=0.15)
    r = A.move_to(grasp_tip(grip.xy[0] + s), pitch=GP, roll=R, linear=True, speed=0.08, until=contact("light"))
    if r.condition_met:                       # landed on the object again: never close on it
        A.move_by(dz=0.03, speed=0.1)
        give_up("a jaw touched the object on the refined descent; re-estimate (pre-grasp view) and rerun")
g = A.close(effort=0.5)
log(f"grasped={g.grasped} gripper={A.gripper:.1f}")
t = A.tip
up = [t[0], t[1], t[2] + 0.05]
A.move_to(up, pitch=pitch_at([up], PITCH if ROLLED else GP), roll=R, speed=0.15)   # unturned: keep the grasp's pitch
if SIDE_VIEW and len(views) == 3:
    arm(OTHER_ARM).rest()                     # out of the way before the carry
checkpoint("lifted", obj, views=[VIEW, "ego"], need="held square between both jaws, not tilted in the grip, off the table (abort if not)")

# ---------------------------------------------------------------- place
GH = grip.xy[0]                               # where the jaws hold it above its base (checked at grasp height)
p_high = place_tip(TOP_Z + GH + 0.08)
A.move_to(p_high, pitch=pitch_at([p_high], PLACE_PITCH), roll=R, speed=0.2)
s2 = sag()
low_z = TOP_Z + CLEAR + GH + s2 if KIND == "container" else max(TOP_Z + GH - 0.005 + s2, GRIP_FLOOR)
for i in range(3):
    PP = pitch_at([place_tip(TOP_Z + GH + 0.03 + s2), place_tip(low_z)], PLACE_PITCH)   # same pitch for the lowering
    A.move_to(place_tip(TOP_Z + GH + 0.03 + s2), pitch=PP, roll=R, speed=0.2)
    need = ("held object over the opening, rim visible all round it" if KIND == "container"
            else "held object over the target area, its footprint inside the support")
    if checkpoint(f"over target {i+1}", target, views=[VIEW, "ego"], need=need) != "refine":
        break
PP = pitch_at([place_tip(TOP_Z + GH + 0.03 + s2), place_tip(low_z)], PLACE_PITCH)
if KIND == "container":
    A.move_to(place_tip(low_z), pitch=PP, roll=R, linear=True, speed=0.1, until=contact("light"))
else:
    # aim 5 mm below resting height so contact ends the step - but never command the jaws below the grip floor
    r = A.move_to(place_tip(low_z), pitch=PP, roll=R, linear=True, speed=0.08, until=contact("light"))
    log(f"set-down: {r.status} {r.condition_met}")
    checkpoint("touching, still held", target, views=[VIEW, "ego"], need="object resting on the target, upright, before release")
# Let go in two stages: ease the squeeze first, so the object drops straight down, then open wide before the lift.
# 2026-10-15 (sim): opening wide at once dragged a squeezed 4 cm block over the rim on the moving jaw; opening only
# ~30 units past the grip left it wedged between the jaws, and the lift carried it off.
A.open(min(80.0, A.gripper + 8.0), speed=0.1)
A.open(80.0, speed=0.3)
t = A.tip
up = [t[0], t[1], max(t[2] + 0.05, TOP_Z + GH + 0.08)]   # well above the rim before the jaws close or the arm turns
try:
    A.move_to(up, pitch=pitch_at([up], PLACE_PITCH if ROLLED else PP), roll=R, speed=0.15)   # unturned: keep the pitch
except StepFailed as e:
    if e.result.get("status") == "rejected":  # refused before moving: never leave open jaws over the target (an idle
        A.close(effort=0.3)                   # rest would sweep them); gently, in case the object is still between them
    raise
checkpoint("released", target, views=[VIEW, "ego"], need="object on/in the target, clear of the jaws")
A.open(20)                                    # jaws in before going to rest (2026-10-15, sim: a cup wedged on the jaws)
A.rest()
done("placed")
