"""Solve camera intrinsics from checkerboard frames, and say what they mean for the sim.

Reads the frames `capture_checkerboard.py` saved, finds the board in each, and fits both models:
the standard pinhole + radial/tangential distortion, and the fisheye model (better past ~90 deg of
field). Reports focal length, principal point, distortion, reprojection error in pixels, and the
horizontal / vertical / diagonal field of view the fit implies -- the number the sim's camera config
needs. Writes `intrinsics.json` next to the frames and an undistorted copy of one frame to look at.

    ./.venv/bin/python tools/calibrate_camera.py --frames ~/calib/ego --squares-mm 25 --cols 9 --rows 6

`--squares-mm` is the side of one printed square. Verified by running it both ways: it does NOT
change the focal length, field of view or distortion -- those come out of the corner geometry alone,
so a wrong square size costs the intrinsics nothing. It sets the metric scale, which is what
`solve_extrinsics.py` needs, so measure it properly anyway (across several squares, divided; printers
scale) and the same number then serves both steps.

`--cols`/`--rows` are inner corners. A low reprojection error is necessary but NOT sufficient: a set
of flat-on views fits itself beautifully and still gets the focal length wrong, measured here at
0.287 px error with a 5.5% focal error. What bounds the focal length is TILT -- views of the board
angled 20-45 deg away from square-on. Read the error together with the capture step's coverage map,
and if the board was never tilted, recapture rather than trust the number. Above ~1 px, look at the
`vis_*.jpg` frames for a misdetected board.

The output feeds two places: `fov_h_deg` in the Digital Twin's `scripts/newton/robots/so101.yaml`
(so sim renders frame the scene as this camera does), and the OpenCV distortion model that Isaac
Lab's camera spawners accept (so sim frames bend at the edges as this lens does).
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so the tools run from any cwd
from board_detect import find_board


def fov(f_px: float, extent_px: int) -> float:
    return 2.0 * math.degrees(math.atan(extent_px / (2.0 * f_px)))


def fov_diag(fx: float, fy: float, W: int, H: int) -> float:
    """Exact, unlike averaging the focal lengths against the pixel diagonal: that is off by half a
    degree at fx/fy = 0.97 and by nearly three at 1.33."""
    return 2.0 * math.degrees(math.atan(math.hypot(W / 2.0 / fx, H / 2.0 / fy)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="directory of frames from capture_checkerboard.py")
    p.add_argument("--squares-mm", type=float, required=True, help="measured side of one square, mm")
    p.add_argument("--cols", type=int, default=9, help="INNER corners across")
    p.add_argument("--rows", type=int, default=6, help="INNER corners down")
    p.add_argument("--out", default=None, help="where to write intrinsics.json (default: --frames)")
    run(p.parse_args())


def run(a) -> dict:
    """Solve the lens from a.frames (frame_*.png) and write intrinsics.json into a.out (default a.frames). `a` has
    the command line's fields (frames, squares_mm, cols, rows, out); tools/calibrate.py calls this directly. Returns
    what it wrote; a fatal problem raises SystemExit with the reason."""
    out = a.out or a.frames
    os.makedirs(out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.frames, "frame_*.png")))
    if not paths: raise SystemExit(f"[FATAL] no frame_*.png in {a.frames}")
    objp = np.zeros((a.rows * a.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * (a.squares_mm / 1000.0)   # metres

    grays = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:                                  # truncated or zero-byte: capture was interrupted
            print(f"  skip {os.path.basename(path)}: unreadable"); continue
        grays.append((path, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    if not grays: raise SystemExit(f"[FATAL] none of the {len(paths)} files in {a.frames} could be read as an image")
    # The MODE, not the first file: a few stray frames at another resolution must not discard the rest.
    sizes = Counter(g.shape[::-1] for _, g in grays)
    size = sizes.most_common(1)[0][0]
    if len(sizes) > 1:
        print(f"  mixed resolutions {dict(sizes)}: keeping {size[0]}x{size[1]}, the most common. "
              "Intrinsics are per-mode, so calibrate each separately.")

    obj_pts, img_pts, used, fallbacks = [], [], [], 0
    for path, gray in grays:
        if gray.shape[::-1] != size: print(f"  skip {os.path.basename(path)}: {gray.shape[::-1]} != {size}"); continue
        found, corners, how = find_board(gray, a.cols, a.rows)
        if not found: print(f"  skip {os.path.basename(path)}: board not found"); continue
        if how != "sb+accuracy": fallbacks += 1
        obj_pts.append(objp); img_pts.append(corners); used.append(path)
    print(f"[calib] {len(used)} / {len(paths)} frames usable at {size[0]}x{size[1]}, board {a.cols}x{a.rows} @ {a.squares_mm} mm"
          + (f" ({fallbacks} needed a fallback detector -- usually uneven lighting, and their corners can be "
              f"several px out, so watch the per-frame errors)" if fallbacks else ""))
    if len(used) < 8: raise SystemExit("[FATAL] need at least ~8 good frames (20-30 is better)")

    W, H = size

    def fit(objs, imgs):
        # Seed the focal length instead of letting OpenCV guess it.
        #
        # With no guess, OpenCV's internal initialisation lands in a local
        # minimum on wide lenses. Measured on the ego camera's 27-frame guided
        # set: the default converged to fx 950 / fy 975 (2.6% apart -- pixels
        # are square, so that alone is a tell) at 67.9 deg HFOV with rms 0.423,
        # while ANY sane seed converges to fx = fy = 830 at 75.3 deg with rms
        # 0.282. The seeded fit is better, agrees with the fisheye (75.3) and
        # rational (74.9) models, and the default was the only one disagreeing.
        # A worse fit reporting a different focal length is the "confident wrong
        # number" this pipeline exists to avoid.
        guess = W * 0.65                      # ~75 deg HFOV; any value converges
        K0 = np.array([[guess, 0, W / 2], [0, guess, H / 2], [0, 0, 1]], np.float64)
        # np.zeros((1, 5)), not np.zeros(5): the shape passed in is the shape
        # returned, and the rest of this file indexes dist as dist[0][i].
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objs, imgs, size, K0, np.zeros((1, 5)), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        free = cv2.calibrateCamera(objs, imgs, size, None, None)
        if free[0] < rms - 1e-9:              # keep whichever genuinely fits better
            rms, K, dist, rvecs, tvecs = free
        dist = np.asarray(dist).reshape(1, -1)
        errs = []
        for i in range(len(objs)):
            proj, _ = cv2.projectPoints(objs[i], rvecs[i], tvecs[i], K, dist)
            errs.append(float(cv2.norm(imgs[i], proj, cv2.NORM_L2) / math.sqrt(len(proj))))
        return rms, K, dist, rvecs, tvecs, errs

    rms, K, dist, rvecs, tvecs, per_frame = fit(obj_pts, img_pts)

    # One frame with corners in the wrong place -- a hard shadow edge across the board is the usual
    # cause -- drags the whole fit. Its own reprojection error gives it away, so refit without the
    # frames that stand far outside the rest. Capped so a genuinely poor capture cannot be whittled
    # down to a flattering subset: it drops at most a third, and never below the 8-frame minimum.
    med = float(np.median(per_frame))
    keep = [i for i, e in enumerate(per_frame) if e <= max(3.0 * med, med + 0.25)]
    dropped = []
    if len(keep) < len(per_frame) and len(keep) >= max(8, int(0.67 * len(per_frame))):
        dropped = [(os.path.basename(used[i]), per_frame[i]) for i in range(len(used)) if i not in keep]
        print(f"  dropped {len(dropped)} frame(s) whose corners disagree with the rest "
              f"(error {min(e for _, e in dropped):.2f}-{max(e for _, e in dropped):.2f} px against a "
              f"{med:.2f} px median): " + ", ".join(n for n, _ in dropped))
        obj_pts = [obj_pts[i] for i in keep]; img_pts = [img_pts[i] for i in keep]
        used = [used[i] for i in keep]
        rms, K, dist, rvecs, tvecs, per_frame = fit(obj_pts, img_pts)
    elif len(keep) < len(per_frame):
        print(f"  NOTE: {len(per_frame) - len(keep)} frame(s) have errors well above the {med:.2f} px median, "
              "but dropping them would leave too few to trust. Recapture rather than lean on this fit.")
    # How far from square-on was the board actually held? This is what pins the focal length down,
    # and the reprojection error cannot see it: a set of flat-on views fits itself to a third of a
    # pixel with the focal length a quarter wrong. The quantity is the board's rotation away from the
    # image plane, i.e. the angle between its normal and the optical axis -- NOT the angle to the line
    # of sight, which is non-zero merely because the board sits off-centre and reads ~36 deg on views
    # that are in fact perfectly square-on.
    tilts = np.array([math.degrees(math.acos(min(1.0, abs(float(cv2.Rodrigues(rv)[0][2, 2])))))
                      for rv in rvecs])

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"\n[pinhole]  reprojection error {rms:.3f} px (worst frame {max(per_frame):.3f})")
    print(f"  focal  fx {fx:8.1f}  fy {fy:8.1f} px      principal point  cx {cx:7.1f}  cy {cy:7.1f} px (centre is {W/2:.0f}, {H/2:.0f})")
    print(f"  FOV    horizontal {fov(fx, W):5.1f} deg   vertical {fov(fy, H):5.1f} deg   diagonal {fov_diag(fx, fy, W, H):5.1f} deg")
    print(f"  distortion k1 {dist[0][0]:+.4f}  k2 {dist[0][1]:+.4f}  p1 {dist[0][2]:+.5f}  p2 {dist[0][3]:+.5f}  k3 {dist[0][4]:+.4f}")
    print(f"  board tilt  median {np.median(tilts):.0f} deg, max {tilts.max():.0f} deg over {len(tilts)} frames")
    if tilts.max() < 20.0:
        print(f"\n  WARNING: the board was never tilted more than {tilts.max():.0f} deg from square-on, so this")
        print("  focal length is NOT trustworthy however small the reprojection error looks -- flat-on views")
        print("  cannot separate focal length from distance. On a synthetic test this exact failure returned")
        print("  0.34 px error with the focal length 24% wrong. Recapture with the board angled 20-45 deg,")
        print("  in both axes, and rolled.")
    elif np.median(tilts) < 15.0:
        print(f"\n  NOTE: median tilt is only {np.median(tilts):.0f} deg. A few tilted views carry the focal")
        print("  length here; more of them would tighten it.")

    fish = None
    try:
        fK = np.zeros((3, 3)); fD = np.zeros((4, 1))
        obj_f = [o.reshape(1, -1, 3) for o in obj_pts]; img_f = [c.reshape(1, -1, 2) for c in img_pts]
        frms, fK, fD, _, _ = cv2.fisheye.calibrate(
            obj_f, img_f, size, fK, fD,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_CHECK_COND,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        # Reject nonsense that converged instead of failing. A negative focal length is the obvious
        # case; the DANGEROUS one is positive and absurd -- untilted input has produced fx = 11 038 px
        # (17x the truth) at a 0.07 px error, which passes every check that looks only at the fisheye
        # fit itself. The pinhole fit on the same corners is the reference: the two models disagree
        # about distortion, not about roughly how long the lens is.
        if fK[0, 0] <= 0 or frms > 5.0 or not (0.5 * fx < fK[0, 0] < 2.0 * fx):
            raise cv2.error(f"fisheye fit is not usable: focal {fK[0,0]:.1f} px against the pinhole fit's "
                            f"{fx:.1f} px, error {frms:.1f} px")
        print(f"\n[fisheye]  reprojection error {frms:.3f} px")
        print(f"  focal  fx {fK[0,0]:8.1f}  fy {fK[1,1]:8.1f} px      FOV horizontal {fov(fK[0,0], W):5.1f} deg")
        print(f"  distortion k1..k4 {' '.join(f'{v[0]:+.4f}' for v in fD)}")
        fish = dict(rms=float(frms), K=fK.tolist(), D=fD.ravel().tolist(), fov_h_deg=fov(fK[0, 0], W))
    except cv2.error as e:
        print(f"\n[fisheye]  did not converge ({str(e).splitlines()[-1][:90]}); the pinhole model is the one to use")

    res = dict(frames_dir=os.path.abspath(a.frames), n_frames=len(used), image_size=[W, H],
               board=dict(cols=a.cols, rows=a.rows, square_m=a.squares_mm / 1000.0),
               dropped_frames=[dict(frame=n, rms=e) for n, e in dropped],
               pinhole=dict(rms=float(rms), K=K.tolist(), dist=dist.ravel().tolist(),
                            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                            fov_h_deg=fov(fx, W), fov_v_deg=fov(fy, H), fov_d_deg=fov_diag(fx, fy, W, H),
                            per_frame_rms=per_frame, tilt_deg=tilts.tolist(),
                            tilt_median_deg=float(np.median(tilts)), tilt_max_deg=float(tilts.max())),
               fisheye=fish)
    with open(os.path.join(out, "intrinsics.json"), "w") as fh: json.dump(res, fh, indent=1)

    img = cv2.imread(used[0]); newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 1, size)
    side = np.hstack([img, cv2.undistort(img, K, dist, None, newK)])
    cv2.imwrite(os.path.join(out, "undistorted_check.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n[calib] wrote {out}/intrinsics.json and undistorted_check.jpg (left: raw, right: undistorted --")
    print("        straight edges should be straight on the right, and the corners stretched outward)")
    print(f"\nFor the sim camera config: fov_h_deg {fov(fx, W):.1f} at {W}x{H}. If this is the ego camera, it")
    print("replaces the 103.0 in so101.yaml, which is the lens spec rather than a measurement of this mode.")
    return res


if __name__ == "__main__":
    main()
