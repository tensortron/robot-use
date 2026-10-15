"""Where is the ego camera, really? One photo of the checkerboard lying on the table answers it.

The sim's ego camera pose is currently a tape-measure estimate, and the estimate before that (a
silhouette fit against a photo) was 13 cm and 30 deg wrong. This measures it: the board lies flat on
the table where you can reach it with a ruler, `solvePnP` recovers the board's pose in the camera,
and the two combine into the camera's height, lateral offset and downward tilt in the rig's own
frame -- the numbers `eye_m` and `tilt_down_deg` in the Digital Twin's
`scripts/newton/robots/so101.yaml` are written in.

Run `calibrate_camera.py` first. Pose and focal length trade off against each other, so a guessed
field of view gives a confident, wrong height; that degeneracy is what made the photo fit wrong.

TWO PASSES, because only the detector knows how it numbered the board.

    # 1. look: no measurements needed, writes extrinsics_check.jpg
    ./.venv/bin/python tools/solve_extrinsics.py --image ~/calib/ego_pose/board_on_table.png \
        --intrinsics ~/calib/ego/intrinsics.json --squares-mm 25

    # 2. solve: measure to the corner the image circled, and read the arrow's direction off it
    ./.venv/bin/python tools/solve_extrinsics.py --image ~/calib/ego_pose/board_on_table.png \
        --intrinsics ~/calib/ego/intrinsics.json --squares-mm 25 \
        --corner-x 0.18 --corner-y -0.10 --board-yaw-deg 90

HOW TO PLACE THE BOARD. Flat on the table, pattern up, whole board visible, in front of the arms,
square to the rig: its edges parallel and perpendicular to the arm-base line, which is the line
through both arms' shoulder-pan axes. Weigh it down so it lies genuinely flat.

    rig frame: origin on the table at the midpoint between the two arm bases,
               +x forward (away from the arms, toward the workspace),
               +y to the robot's LEFT,  +z up.

Pass 1 circles one inner corner and draws an arrow along the board's numbering direction. Measure
that corner's x and y with a ruler, and read the arrow: it points to the robot's left -> 90, to the
robot's right -> 270, straight forward -> 0, back toward the arms -> 180. Those two facts are the
whole input. The remaining ambiguity -- which side of the board the numbering calls "up" -- is not
yours to resolve: the tool fixes it by requiring the camera to end up above the table.

Extrinsics die the moment the camera moves or the arms are repositioned. Intrinsics do not: they
belong to the lens and the resolution, and survive a remount.
"""
from __future__ import annotations
import argparse, json, math, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so the tools run from any cwd
from board_detect import find_board


def object_points(cols: int, rows: int, sq: float) -> np.ndarray:
    """Board-frame corners in the order findChessboardCorners returns them: row-major, X along the
    `cols` direction, Y along `rows`. Index i is (c, r) = (i % cols, i // cols)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    return objp


def annotate(img, corners, cols, rows, path):
    vis = img.copy()
    cv2.drawChessboardCorners(vis, (cols, rows), corners, True)
    c0 = np.round(corners[0, 0]).astype(int); cx = np.round(corners[cols - 1, 0]).astype(int)
    cv2.arrowedLine(vis, tuple(c0), tuple(cx), (0, 140, 255), 4, cv2.LINE_AA, tipLength=0.08)
    cv2.circle(vis, tuple(c0), 24, (0, 0, 255), 3)
    cv2.putText(vis, "measure to THIS corner", (max(5, c0[0] - 170), max(28, c0[1] - 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    mid = ((c0 + cx) // 2)
    cv2.putText(vis, "board-yaw arrow", (max(5, mid[0] - 110), min(img.shape[0] - 8, mid[1] + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return c0, cx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="one photo of the board lying flat on the table")
    p.add_argument("--intrinsics", required=True, help="intrinsics.json from calibrate_camera.py, SAME camera and resolution")
    p.add_argument("--squares-mm", type=float, required=True, help="measured side of one square, mm")
    p.add_argument("--cols", type=int, default=9, help="INNER corners across")
    p.add_argument("--rows", type=int, default=6, help="INNER corners down")
    p.add_argument("--corner-x", type=float, default=None, help="rig-frame x of the circled corner, metres (+ = forward)")
    p.add_argument("--corner-y", type=float, default=None, help="rig-frame y of the circled corner, metres (+ = robot's left)")
    p.add_argument("--board-yaw-deg", type=float, default=None,
                   help="rig direction the drawn arrow points: 0 forward, 90 robot's left, 180 back, 270 robot's right")
    p.add_argument("--config-plane-offset-m", type=float, default=0.0,
                   help="height of the consuming config's z=0 plane ABOVE the plane the board lay on. "
                        "Leave 0 for the SO-101: since 2026-10-01 its config (the Digital Twin's scripts/newton/robots/"
                        "so101.yaml) stores eye_m in THIS tool's rig frame and the sim converts it (rig_frame.py).")
    p.add_argument("--out", default=None, help="where to write the outputs (default: beside --image)")
    run(p.parse_args())


def run(a) -> dict:
    """Pass 1 (no corner or arrow given): find the board and write extrinsics_check.jpg; returns {"pass": 1,
    "corner_px", "arrow_to_px", "check"}. Pass 2: solve and write extrinsics.json; returns what it wrote, plus
    "alternatives" (the pose under each of the four arrow directions, for a caller to offer). `a` has the command
    line's fields; tools/calibrate.py calls this directly. A fatal problem raises SystemExit with the reason."""
    out = a.out or os.path.dirname(os.path.abspath(a.image))
    os.makedirs(out, exist_ok=True)

    if not os.path.exists(a.intrinsics):
        raise SystemExit(f"[FATAL] no intrinsics at {a.intrinsics} -- run calibrate_camera.py on this camera first: "
                         "the pose cannot be separated from the focal length without it")
    intr = json.load(open(a.intrinsics))
    K = np.array(intr["pinhole"]["K"]); dist = np.array(intr["pinhole"]["dist"])
    img = cv2.imread(a.image)
    if img is None: raise SystemExit(f"[FATAL] cannot read {a.image}")
    H, W = img.shape[:2]
    if [W, H] != list(intr["image_size"]):
        raise SystemExit(f"[FATAL] the photo is {W}x{H} but these intrinsics are for "
                         f"{intr['image_size'][0]}x{intr['image_size'][1]}. Intrinsics are per-mode: "
                         "retake the photo at the calibrated resolution, or calibrate this one.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners, how = find_board(gray, a.cols, a.rows)
    if not found: raise SystemExit(f"[FATAL] no {a.cols}x{a.rows} board found: is all of it visible, lit and in focus?")
    check = os.path.join(out, "extrinsics_check.jpg")
    c0, cx = annotate(img, corners, a.cols, a.rows, check)

    if a.corner_x is None or a.corner_y is None or a.board_yaw_deg is None:
        print(f"[pass 1] board found, {len(corners)} corners -> {check}\n")
        print(f"  The circled corner is at pixel ({c0[0]}, {c0[1]}); the arrow runs from it to ({cx[0]}, {cx[1]}).")
        print("  Now measure that corner in the rig frame (origin midway between the arm bases, +x forward,")
        print("  +y to the robot's left) and read which way the arrow points on the table:\n")
        print("      arrow toward the robot's left   --board-yaw-deg 90")
        print("      arrow toward the robot's right  --board-yaw-deg 270")
        print("      arrow straight forward          --board-yaw-deg 0")
        print("      arrow back toward the arms      --board-yaw-deg 180\n")
        print("  Then re-run with --corner-x --corner-y --board-yaw-deg added.")
        return {"pass": 1, "corner_px": [int(c0[0]), int(c0[1])], "arrow_to_px": [int(cx[0]), int(cx[1])],
                "check": check}

    objp = object_points(a.cols, a.rows, a.squares_mm / 1000.0)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: raise SystemExit("[FATAL] solvePnP did not converge")
    R_cb, _ = cv2.Rodrigues(rvec)                                   # board -> camera
    C_b = -R_cb.T @ tvec.ravel()                                    # camera centre, board frame
    d_b = R_cb.T @ np.array([0.0, 0.0, 1.0])                        # optical axis, board frame
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    rms = float(cv2.norm(corners, proj, cv2.NORM_L2) / math.sqrt(len(proj)))

    z_sign = 1.0 if C_b[2] > 0 else -1.0    # does the board's own Z point toward the camera, or away?

    def place(yaw_deg):
        """Put the solved board-frame pose into the rig frame, for a declared arrow direction.

        The user declares where the board X axis points. Y then follows from Z, and the sign of Z is
        not something the numbering fixes, so take the one that puts the camera above the table. The
        reprojection error above is computed BEFORE any of this and comes out identical for all four
        declarations: a quarter-turn-wrong arrow cannot be caught by the fit, only by plausibility."""
        psi = math.radians(yaw_deg)
        Xw = np.array([math.cos(psi), math.sin(psi), 0.0])
        Zw = np.array([0.0, 0.0, z_sign])
        R_wb = np.column_stack([Xw, np.cross(Zw, Xw), Zw])          # right-handed by construction
        C = R_wb @ C_b + np.array([a.corner_x, a.corner_y, 0.0])
        d = R_wb @ d_b
        bel = math.degrees(math.asin(max(-1.0, min(1.0, -d[2]))))
        hea = math.degrees(math.atan2(d[1], d[0]))                  # 0 = straight forward, + = robot left
        return C, d, bel, hea

    C_w, d_w, below, heading = place(a.board_yaw_deg)
    tilt_down = 90.0 - below                                        # the config convention: 0 = straight down
    hit = C_w + d_w * (-C_w[2] / d_w[2]) if d_w[2] < -1e-6 else None

    print(f"[extrinsics] {os.path.basename(a.image)} at {W}x{H} | reprojection error {rms:.3f} px | "
          f"arrow declared at {a.board_yaw_deg:g} deg, corner at ({a.corner_x:+.3f}, {a.corner_y:+.3f}) m")
    print(f"\n  camera position, rig frame   x {C_w[0]:+.3f}   y {C_w[1]:+.3f}   z {C_w[2]:+.3f} m")
    print(f"    {abs(C_w[0])*100:.1f} cm {'ahead of' if C_w[0] >= 0 else 'behind'} the arm-base line, "
          f"{abs(C_w[1])*100:.1f} cm to the robot's {'left' if C_w[1] >= 0 else 'right'}, "
          f"{C_w[2]*100:.1f} cm above the table")
    print(f"  looking                      {below:.1f} deg below horizontal  ->  tilt_down_deg {tilt_down:.1f}"
          "  (config counts from straight down)")
    side = ("straight forward" if abs(heading) < 0.05 else
            "toward the robot's left" if heading > 0 else "toward the robot's right")
    print(f"  heading                      {heading:+.1f} deg ({side}; 0 = straight forward)")
    if hit is not None:
        print(f"  centre of view meets the table at x {hit[0]:+.3f}  y {hit[1]:+.3f} m")

    if rms > 1.0:
        print(f"\n  WARNING: {rms:.2f} px is a poor fit -- wrong --squares-mm, a board that is not flat, "
              "or intrinsics from a different camera or resolution.")

    # A wrong arrow direction is the one error the fit CANNOT see: all four declarations reproject
    # identically, so only the rig geometry says which is possible. A camera mounted between the arms
    # looks roughly forward from roughly over the base line; a large heading, or a camera half a metre
    # down the table, means the declaration is wrong rather than the measurement.
    implausible = []
    if abs(heading) > 45.0: implausible.append(f"it would point {abs(heading):.0f} deg away from forward")
    if abs(C_w[0]) > 0.35: implausible.append(f"it would sit {abs(C_w[0])*100:.0f} cm from the base line")
    if abs(C_w[1]) > 0.15: implausible.append(f"it would sit {abs(C_w[1])*100:.0f} cm off centre")
    if below < 5.0: implausible.append(f"it would look along the table, {below:.1f} deg below horizontal")
    if implausible:
        print("\n  WARNING: this pose is implausible for a camera mounted between the arms -- "
              + "; ".join(implausible) + ".")
        print("  The likeliest cause is --board-yaw-deg. The reprojection error is IDENTICAL for all four")
        print("  declarations, so the fit cannot tell them apart. Here is what each one would mean:\n")
        print("      arrow           cam x    y       z     heading   below horizontal")
        for yaw, label in ((0.0, "forward"), (90.0, "left"), (180.0, "back"), (270.0, "right")):
            Ci, _, bi, hi = place(yaw)
            mark = "   <- declared" if abs(yaw - a.board_yaw_deg % 360.0) < 1e-6 else ""
            print(f"      {yaw:3.0f} ({label:<7s})  {Ci[0]:+6.3f} {Ci[1]:+6.3f} {Ci[2]:+6.3f}   {hi:+7.1f}   {bi:6.1f}{mark}")
        if not any(abs(y - a.board_yaw_deg % 360.0) < 1e-6 for y in (0.0, 90.0, 180.0, 270.0)):
            print(f"      (you declared {a.board_yaw_deg:g}, which is not one of these four)")
        print("\n  Pick the row that matches the rig, re-read the arrow in extrinsics_check.jpg, and re-run.")

    res = dict(image=os.path.abspath(a.image), intrinsics=os.path.abspath(a.intrinsics), image_size=[W, H],
               board=dict(cols=a.cols, rows=a.rows, square_m=a.squares_mm / 1000.0, yaw_deg=a.board_yaw_deg,
                          corner_xy=[a.corner_x, a.corner_y], corner_px=[int(c0[0]), int(c0[1])],
                          board_z_sign=float(z_sign)),
               reprojection_rms_px=rms, eye_m=C_w.tolist(), optical_axis=d_w.tolist(),
               deg_below_horizontal=below, tilt_down_deg=tilt_down, heading_deg=heading,
               table_hit_xy=(hit[:2].tolist() if hit is not None else None),
               implausible=implausible,                                 # empty when the pose passes
               config_plane_offset_m=a.config_plane_offset_m,
               eye_m_config=[C_w[0], C_w[1], C_w[2]-a.config_plane_offset_m],
               rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist())
    json.dump(res, open(os.path.join(out, "extrinsics.json"), "w"), indent=1)
    res["alternatives"] = {}
    for yaw in (0.0, 90.0, 180.0, 270.0):
        Ci, _, bi, hi = place(yaw)
        res["alternatives"][int(yaw)] = {"eye_m": Ci.tolist(), "heading_deg": hi, "below_deg": bi,
                                         "plausible": abs(hi) <= 45 and abs(Ci[0]) <= 0.35 and abs(Ci[1]) <= 0.15
                                         and bi >= 5}
    print(f"\n[extrinsics] wrote {out}/extrinsics.json and extrinsics_check.jpg")
    if implausible:
        # Deliberately NOT printing the copy-paste line: the warning above scrolls past, and the one
        # thing that must not happen is this pose reaching so101.yaml. The json records why.
        print("  No config line, because the pose above is implausible. Fix --board-yaw-deg and re-run.")
    else:
        # The height above is measured from the plane the BOARD lay on. A config whose z=0 sits on a
        # different plane needs that difference removed, which is easy to forget and has been got
        # backwards before -- so print the raw number only when the two planes are the same.
        off = a.config_plane_offset_m
        print(f"  for the config:  eye_m: [{C_w[0]:.3f}, {C_w[1]:.3f}, {C_w[2]-off:.3f}]"
              f"   tilt_down_deg: {tilt_down:.1f}   yaw_deg: {heading:.1f}")
        if off:
            print(f"    (z = {C_w[2]:.3f} measured above the board's plane, minus the {off*100:.1f} cm "
                  "offset you declared for the config's own z = 0)")
        else:
            print("    rig frame: x from the shoulder-pan line, z above the plane THE BOARD LAY ON. The SO-101 config")
            print("    takes exactly these numbers (the sim converts them, rig_frame.py); do NOT subtract the 18 mm board.")
    return res


if __name__ == "__main__":
    main()
