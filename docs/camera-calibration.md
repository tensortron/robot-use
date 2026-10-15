# Camera calibration — what the lenses actually do

> To calibrate a rig, start with [CALIBRATION.md](CALIBRATION.md) and `tools/calibrate.py`. This page is the full
> record of how this rig's lenses were measured, and why each precaution is there.

**Measured 2026-09-22 (ego camera, 1280×720).**

| | value |
|---|---|
| horizontal FOV | **75.2°** |
| vertical FOV | 46.9° |
| diagonal FOV | 83.0° |
| fx, fy | 830.5, 829.4 px |
| principal point | 586.9, 365.6 px (centre is 640, 360) |
| distortion k1..k3, p1, p2 | −0.4157, +0.1799, −0.0366, −0.00086, −0.00206 |
| reprojection error | 0.282 px (worst frame 0.508) |
| frames | 27, all 9 cells × 3 tilt bands; tilt 14–41°, spread 27° |

Intrinsics: `~/calib/ego2/intrinsics.json`, copied into the repo as `calib/ego_intrinsics.json`
(2026-10-13), which is the file `robot.yaml` points at.

So the lens's advertised **103° is not what this mode sees** — it is the diagonal
over the full sensor, and 1280×720 crops to 75% of it (measured; see the mode table
below). The simulator's `fov_h_deg: 103.0` in
`scripts/newton/robots/so101.yaml` framed every render differently from the real
camera; **replace it with 75.2**.

### How the three modes relate

Each mode treats the sensor differently, which is why intrinsics cannot be assumed to
carry across a resolution change. Measured by comparing the size of one board square
on a single static scene:

| mode | square | vs 720p | relationship | fx | HFOV | VFOV | diagonal |
|---|---|---|---|---|---|---|---|
| 1920×1080 | 38.10 px | ×1.118 | **full sensor** | ~928 | **~92°** | ~60° | ~100° |
| 1280×720 | 34.09 px | — | **crop** to 75% of it | 830.5 | **75.2°** | 46.9° | 83.0° |
| 1024×576 | 20.26 px¹ | ×0.798 | **scale** of the 720p crop | 662.7 | **75.4°** | 47.0° | 83.1° |

¹ measured against a different scene, so compare its ratio, not the raw pixels.

The test is one number: a square scales by the pixel ratio (1.5 for 1080p, 0.8 for
576p) if the mode merely resamples the same region, and stays put if it crops. 1080p
gave 1.118 — neither — so it sees **more sensor**, not just more pixels.

**This explains the data sheet.** Its 103° diagonal is within measurement error of the
~100° at 1920×1080: the spec describes the full-sensor mode, and 1280×720 crops to 75%
of it in each axis. The figure was never wrong, just not a description of the mode the
rig runs.

Practical consequence: **720p → 576p costs nothing in field of view, but 1080p → 720p
costs a quarter of the scene in each axis.**

The 1080p row is scaled from the 720p calibration, not solved — good to a couple of
degrees, no better. Two reasons it cannot be tightened by arithmetic: the principal
point is 53 px off centre, so the crop is not symmetric, and distortion coefficients do
not survive a crop the way they survive a scale. 1080p would need its own capture, and
it is moot for the rig anyway — three cameras at 1080p do not fit the USB budget, which
is why these modes were chosen ([usb-bandwidth.md](usb-bandwidth.md)).

### Wrist cameras — inferred, not measured

Same camera model, run at 1024×576. That mode **scales** the same sensor region
rather than cropping it further, so the field of view is unchanged. Measured on one
static scene: a board square is 25.39 px at 1280×720 and 20.26 px at 1024×576, a
ratio of 0.798 against 0.800 predicted by pure scaling (a crop would give 1.0).

| | 1280×720 (measured) | 1024×576 (inferred) |
|---|---|---|
| fx | 830.5 | **662.7** (×0.8) |
| HFOV | 75.2° | **75.4°** |
| VFOV | 46.9° | **47.0°** |

Scale every pixel quantity by 0.8 (fx, fy, cx, cy); the distortion coefficients are
in normalised coordinates and carry over unchanged.

**Left inferred on purpose (decision 2026-09-22).** The wrist cameras were never captured, and the
calibration work was closed with them at ~75.4 deg. The reasoning below is why that is tolerable, and
the caveats are why it is not free. If a wrist view ever needs to be trusted metrically, capture them.

**Two caveats.** This is the *ego* lens. The mode behaviour (scaling) will hold for
any unit of this model, but these are cheap manual-focus lenses: focal length and
distortion vary unit to unit, and a different focus setting shifts the focal length
outright — expect a wrist camera within a few degrees of 75°, not identical. And the
**principal point does not transfer at all**: the ego camera's sits 53 px left of
centre, which is that unit's sensor mounting. Assume centre for a wrist camera until
it is measured.

Good enough to unblock sim work. Not a substitute for calibrating them, which is
about ten minutes each with the guided capture below.

## What you need

A printed checkerboard and a ruler. No arm movement and no server running: stop any
policy client first, since it holds the cameras. The intrinsics step needs only the
board; the extrinsics step at the end also needs the ruler, to locate the board on
the table.

1. Print [`assets/checkerboard_9x6_25mm.pdf`](../assets/checkerboard_9x6_25mm.pdf) on
   A4, **landscape, at 100% ("actual size")** — turn "fit to page" off, since every
   printer's default shrinks slightly for its unprintable margin and that quietly
   rescales your measuring instrument. It is 10 × 7 squares, which is **9 × 6 inner
   corners**; inner corners are what the solvers count, and the outer border does not.
   An SVG of the same board sits beside it, and `tools/make_checkerboard.py` regenerates
   both at any square size or page.

   The PDF is written directly in physical units rather than converted from the SVG by a
   renderer that might resample it. Checked by rasterising it back at 300 dpi and
   measuring: a 25 mm square comes out at 24.998 mm, which is corner-detection noise
   rather than scale error.
2. **Glue or tape it to something stiff and flat** — foam board, clipboard, the back
   of a hardback book. A curled sheet quietly ruins the fit, because the solver
   assumes the board is a perfect plane.
3. **Measure one square with a ruler** and write the number down in millimetres.
   The file is exact; your printer is not, so the print is the target and your ruler is
   the ground truth. Measure across several squares and
   divide, which averages out the ruler. This number does *not* affect the field of
   view — that comes out of the corner geometry alone, confirmed by running the solver
   at 25 and 50 mm and getting identical intrinsics. It sets the metric scale, which is
   what the extrinsics step below depends on entirely.

## Capture — guided (preferred)

The Orin is headless, so `tools/live_view.py` streams the camera to a browser with
the calibration overlay on it, and can bank the frames itself.

**Set the focus first and do not touch the ring afterwards.** These lenses are
manual-focus and intrinsics are per-focus-setting; if the ring moves, the
calibration is void.

```bash
cd <repo>
./.venv/bin/python tools/live_view.py --camera /dev/cam_ego \
    --width 1280 --height 720 --capture-to ~/calib/ego --want 27
# laptop:  ssh -L 8099:localhost:8099 <user>@<robot-host>   then http://localhost:8099
```

It tracks **27 slots** — 9 grid cells × 3 tilt bands (flat 12–26°, mid 26–38°, steep
38–60°) — and tells you which it still wants:

```
NEXT: cell (2,2) tilt mid (26-38d)
```

Hold the board there and it saves the frame itself, with a green flash and a line in
the terminal. A pose already banked is refused (*"this pose is already banked"*), so
a set cannot collapse onto one orientation. Three gates before a frame is kept: it
must **fill an empty slot**, the board must be **still** (centroid steady over five
detections) and **sharp** enough — hand-held motion blur otherwise inflates the fit
silently. Saved frames are the raw capture, taken before the overlay is drawn.

`--want 27` is one frame per slot, a complete set; a larger value tops up the
thinnest slots afterwards. Drop `--capture-to` to use it purely as a viewfinder.

Cell `(c,r)` is zero-indexed from the top-left. Tilt is the angle between the board's
**face** and the camera's line of sight, in any direction — the tool measures
magnitude only, so vary the direction yourself (tip each edge away in turn, and roll
the board in its own plane). Note this camera looks down at ~65°, so a board resting
on the table already reads ~25°; to reach *mid* and *steep* tilt it **further** the
way it already leans, rather than lifting it upright to face the lens, which reduces
tilt toward the degenerate 0°.

Why the tilt fuss: the first ego attempt put all 30 frames in cell (1,1) at 22–27°
(3° of spread). For a planar target that is degenerate — the solver cannot separate
focal length from distance — and it returns a confident wrong number with a small
reprojection error. `capture_checkerboard.py` reports cell coverage only afterwards,
and never checks tilt spread at all.

## Capture — manual

One camera at a time. Nothing else may hold the camera — stop the live view and any
policy client first.

```bash
cd <repo>

# ego, at its deployment resolution
./.venv/bin/python tools/capture_checkerboard.py \
    --camera /dev/cam_ego --width 1280 --height 720 \
    --out ~/calib/ego --cols 9 --rows 6 --want 30 --manual

# each wrist, at theirs
./.venv/bin/python tools/capture_checkerboard.py \
    --camera /dev/cam_wrist_left --width 1024 --height 576 \
    --out ~/calib/wrist_left --cols 9 --rows 6 --want 30 --manual
./.venv/bin/python tools/capture_checkerboard.py \
    --camera /dev/cam_wrist_right --width 1024 --height 576 \
    --out ~/calib/wrist_right --cols 9 --rows 6 --want 30 --manual
```

`--manual` waits for Enter before each grab, which is the easier mode when you are
holding the board yourself. Drop it for a timed grab every `--interval` seconds.

The Jetson has no display, so the feedback is in the terminal. Each grab prints
whether the board was found and which cell of a 3 × 3 grid its centre landed in, and
the run ends with a coverage map.

**How to hold the board.** Fill a good part of the frame, keep the whole board
visible, and hold still enough not to blur. Then vary two things:

- **Position** — visit all nine cells of the frame, and spend real effort on the four
  corners. The distortion the calibration is trying to measure lives at the edges; a
  board that only ever sits in the middle leaves it unconstrained.
- **Tilt** — 20–45° away from square-on, in both axes, and rolled. This is what
  separates focal length from distance. A set of perfectly flat-on views is
  degenerate, and the solver will report a confident, wrong focal length.

Aim for 20–30 accepted frames. If the coverage map ends with empty cells, capture
more there before solving.

## Solve

Runs anywhere with OpenCV, on the frames from the step above.

```bash
./.venv/bin/python tools/calibrate_camera.py \
    --frames ~/calib/ego --squares-mm <your measured number> --cols 9 --rows 6
```

It fits both the pinhole model with radial and tangential distortion, and the fisheye
model, and prints focal length, principal point, distortion, reprojection error, and
the implied horizontal, vertical, and diagonal field of view. It writes
`intrinsics.json` beside the frames, plus `undistorted_check.jpg`, a raw-versus-
corrected pair.

**Reading the result.** A low reprojection error is necessary but not sufficient, and
this is the trap worth knowing about: a set of flat-on views fits itself beautifully
and still gets the focal length badly wrong, because from square-on a small near board
and a large far one produce the same picture. Measured on a synthetic test, boards held
square-on gave a 0.34 px error with the focal length 24% wrong — 757 px against a true
1000, an 80° field of view against a true 65°. Tilt is what breaks that tie, so the
solver now measures how far the board was actually tilted and refuses to vouch for the
number when it was not.

So read three things: the reprojection error (below 0.5 px is good, above 1 px means
look at the `vis_*.jpg` frames for a board the detector got wrong), the tilt line, and
the coverage map from the capture step. In `undistorted_check.jpg` the straight edges
of the board should be straight on the right-hand side, and the image corners stretched
outward.

**Use the pinhole numbers for the sim**, which renders an ideal pinhole camera. The
fisheye fit is a cross-check: on the ego camera it converged at 0.229 px against the
pinhole's 0.282 and reported 75.3° against 75.2°, and that agreement is the strongest
evidence the answer is right (see the local-minimum trap above). Prefer fisheye as the
*model* only past roughly 90°, which neither lens here reaches.

If the fisheye fit reports that it did not converge, that is the tool refusing a bad
answer rather than a failure: without tilt it otherwise returns a negative focal
length, which it will not write to the file.

Note the ego lens has real barrel distortion (k1 = −0.42) and a principal point 53 px
left of centre. A sim rendering an ideal pinhole will still differ from the real camera
at the frame edges even with the FOV matched; that is the next-order mismatch after
this one.

### The solver can land in a bad minimum — how to tell

`cv2.calibrateCamera` with no initial focal length converges to a local minimum on
wide lenses. On this same 27-frame set it reported **67.9° HFOV with fx 950 / fy 975
and rms 0.423**, where seeding the focal length gives **75.2° with fx = fy = 830 and
rms 0.282**. `calibrate_camera.py` now seeds it (and still runs the free solve,
keeping it only if it genuinely fits better), but the tells are worth knowing, since
this is the "confident wrong number" the whole procedure exists to avoid:

- **fx and fy more than ~1% apart.** Sensor pixels are square. 2.6% apart is the
  symptom, not a rounding artefact.
- **A worse fit claiming a different answer.** Constraining parameters should never
  *improve* rms; if it does, the unconstrained solve converged badly.
- **Disagreement between models.** Here fisheye (75.3°) and the rational model
  (74.9°) agreed with the seeded pinhole; the default was the lone dissenter.

Seeds anywhere from 833 to 1100 converge to the same answer, so nothing is tuned.

## Extrinsics — where the ego camera actually is

**Measured 2026-09-22, on the rebuilt rig.**

```yaml
eye_m: [-0.062, -0.020, 0.708]   # rig frame (since 2026-10-01; was 0.690 in the base-plane convention)
tilt_down_deg: 25.2
```

| quantity | planned | measured | difference |
|---|---|---|---|
| height above work surface | 70.0 cm | 70.8 cm | +0.8 cm |
| setback behind the arm-base line | 5.0 cm | 6.2 cm | +1.2 cm |
| below horizontal | 65.0° | 64.8° | −0.2° |
| lateral, *assumed zero* | 0 | **−2.0 cm** (robot's right) | −2.0 cm |
| heading, *assumed zero* | 0 | **−1.4°** (toward the right) | −1.4° |

Reprojection error 0.159 px. Inputs: circled corner at x +0.292, y −0.0667 m, board yaw 180°,
squares 25 mm, intrinsics from `~/calib/ego2`. Full output in `~/calib/ego_pose/extrinsics.json`.

The stand was built close to plan. The two quantities that had **never been checked** are the
finding: the camera sits 2 cm right of centre and is yawed 1.4° the same way. Using the error table
of this rig's calibration brief (not published), that is about 23 px + 20 px of apparent
displacement for a 4 cm object — roughly a cube-width of systematic bias, in one consistent
direction, that the sim did not model.

### Frames: write the measured numbers as they are (changed 2026-10-01)

`solve_extrinsics` reports in the **rig frame**: x from the shoulder-pan line, z above the **work surface**
the board lay on. **so101.yaml stores `eye_m` in exactly that frame** (0.708, not 0.690). The sim converts it
itself (the Digital Twin's `scripts/newton/rig_frame.py`): its own z = 0 is the arm-base plane, 18 mm higher (the
arms stand on an 18 mm board), and its x = 0 is the arm roots, 38.8 mm behind the pan axis.

```
0.708 m  measured above the work surface   (what solve_extrinsics reports; goes into so101.yaml as is)
0.690 m  = 0.708 − 0.018                    (the same point in sim-world z; computed by the sim, never typed)
```

Before 2026-10-01 the yaml held 0.690 and the sim applied the camera position unconverted, which put the sim
camera 3.9 cm too far behind the arms and the sim table 18 mm too high. Both are fixed and verified
(`<dataset-store>/calib/rig_frame_check/`). Do not subtract the 18 mm again.

### Caveat: one frame

0.159 px is an excellent fit, but it cannot detect an error in the *tape measurements* — a
centimetre off in `--corner-x` gives an equally clean solve that is a centimetre wrong. To tighten
it, move the board to a different spot and re-run: two independent placements agreeing to a few
millimetres is far stronger than one good-looking fit.

### How it was done

The same board, the same afternoon, also measures the camera's *position*, which was previously a
tape-measure estimate, and before that a silhouette fit that turned out 13 cm and 30° wrong.
Lay the board flat on the table where a ruler can reach it, take one photo, and `solvePnP` recovers
the rest.

Do the intrinsics first. Camera pose and focal length trade off against each other — a camera twice
as far away with twice the focal length produces almost the same picture — so a guessed field of
view yields a confident, wrong height. That degeneracy is exactly what made the earlier fit wrong.

Place the board flat, pattern up, whole board visible, in front of the arms, and square to the rig:
edges parallel and perpendicular to the arm-base line. Weigh it down so it lies genuinely flat. The
rig frame has its origin on the work surface, on the line through both shoulder-pan axes, midway between
them; +x forward away from the arms, +y to the robot's left, +z up.

**Two planes, since the rebuild (2026-09-22).** The rig is a mounting board carrying both arms and
the camera stand on one edge, so it can be set down on any table. The work surface is that table,
18 mm below the board's top face, and the arm base plates sit on the board. The checkerboard lies on
the work surface, so the height this step reports is above the WORK SURFACE. Since 2026-10-01 the sim config's `eye_m`
takes the reported numbers AS THEY ARE (rig frame); the simulator converts them to its own frame (z = 0 on the
arm-base plane, x = 0 at the arm roots). Do not subtract the board. See "Frames" above.

The two shoulder-pan axes are 371.5 mm apart. Measuring between the marks you make for them is a free
check that you found the right centres before anything downstream inherits the error.

Take the photo with the same tool, at the same resolution, asking for one frame — it
only saves the frame if it can actually find the board, which is the check you want:

```bash
./.venv/bin/python tools/capture_checkerboard.py \
    --camera /dev/cam_ego --width 1280 --height 720 \
    --out ~/calib/ego_pose --cols 9 --rows 6 --want 1 --manual
mv ~/calib/ego_pose/frame_000.png ~/calib/ego_pose/board_on_table.png
```

Put it in its own directory, not in with the intrinsics frames, so a later re-solve of
the intrinsics does not pick up a photo taken from a completely different viewpoint.
Then:

```bash
# pass 1 — no measurements yet; writes extrinsics_check.jpg
./.venv/bin/python tools/solve_extrinsics.py \
    --image ~/calib/ego_pose/board_on_table.png --intrinsics ~/calib/ego/intrinsics.json --squares-mm 25

# pass 2 — measure the circled corner, read the arrow, solve
./.venv/bin/python tools/solve_extrinsics.py \
    --image ~/calib/ego_pose/board_on_table.png --intrinsics ~/calib/ego/intrinsics.json --squares-mm 25 \
    --corner-x 0.18 --corner-y -0.10 --board-yaw-deg 90
```

Two passes, because only the detector knows how it numbered the board. Pass 1 circles one inner
corner and draws an arrow along the numbering direction. Measure that corner's x and y with a ruler,
and read which way the arrow points on the table: toward the robot's left is 90, to the robot's
right 270, straight forward 0, back toward the arms 180. Those two facts are the entire input. The
remaining ambiguity — which face of the board the numbering treats as up — is resolved by requiring
the camera to come out above the table, not by you.

The output is the camera's position in the rig frame, its angle below horizontal, the `tilt_down_deg`
the config wants (counted from straight down, not from horizontal), and where the centre of view
meets the table.

A quarter-turn error in the declared arrow is the one mistake the fit cannot catch: all four
declarations reproject to exactly the same error, because the board's pose in the camera is the same
either way and only its placement on the table differs. So the tool checks the answer against the rig
instead. If the pose it implies is impossible for a camera mounted between the arms — pointing more
than 45° off forward, sitting more than 35 cm from the base line or 15 cm off centre, or looking
along the table — it says so and prints what each of the four declarations would have meant, one per
row, so you can pick the row that matches the rig and re-run. Validated on a synthetic scene: a
declaration 90° out reports a camera 28 cm down the table pointing 90° sideways, and the correct row
is visible in the same table.

Verified against a synthetic scene with a known camera pose: 0.53 m up and 65° below horizontal came
back as 0.531 m and 65.2°.

**Extrinsics die when the camera moves.** Repositioning the arms or remounting the ego camera voids
them, and they have to be re-measured. The intrinsics survive, because they belong to the lens.

## Keeping the captures

The frames start on the Orin under `~/calib/`, which is fine while capturing and wrong as a
resting place. Copy each camera's directory to the workstation mount afterwards, where data and
results live, and keep a short README beside it saying which camera, which resolution, which
board and when:

```bash
# from the workstation
rsync -av <robot-host>:~/calib/ <dataset-store>/training/so101_camera_calibration/2026-09-21/
```

The `intrinsics.json` files are small and are the actual deliverable, so they also belong in the
repo that consumes them, under the Digital Twin's `assets/calibration/<camera>.json`.

**In this repo:** after a new calibration, copy the results to `calib/ego_intrinsics.json` and
`calib/ego_extrinsics.json`, the files `robot.yaml` points at (paths there are relative to the repo).
Replace the absolute paths inside them (`frames_dir`, `image`, `intrinsics`) with `~/...`; they are only a
record (a test fails on `/home/` or `/Users/` in them). Update the summary under `cameras.ego` in
`robot.yaml` (`eye_m`, `tilt_down_deg`, `yaw_deg`, focal length, principal point, coefficients) to match. Then
restart robotd. The operating agent must not read these files: `agent_ws/.agents/settings.json` denies them, and
a test checks it.

## Where the numbers go

| Output | Consumer |
|---|---|
| `fov_h_deg` | `scripts/newton/robots/so101.yaml` in the Digital Twin repository, replacing the 103.0 lens spec |
| camera matrix + distortion | the OpenCV distortion model Isaac Lab's camera spawners accept, so sim frames bend at the edges the way this lens does |
| principal point | a check on the mount: far off centre means the sensor is cropped off-axis |
| `eye_m` and `tilt_down_deg` | the same config's ego camera block, replacing the tape-measure estimate — **measured, see above**; written in the rig frame as reported (the sim converts it) |

Keep `intrinsics.json` per camera and per resolution. If a camera is replaced or a
lens is refocused, the calibration is void and has to be redone.

**Recalibrate after the arm repositioning.** Moving the arms to a wider separation
and remounting the ego camera changes the extrinsics — where the camera is — but not
the intrinsics, which belong to the lens and the mode. So the numbers here survive
the move; the camera's position and angle do not, and those have to be re-measured
separately.
