"""Find a checkerboard, robustly. Shared by the three calibration tools so they cannot drift apart.

OpenCV offers two detectors and they fail on different inputs, so trying only one loses frames that
the other would have found:

  * `findChessboardCornersSB` with CALIB_CB_ACCURACY is the best of the three and gives corners to a
    fraction of a pixel -- but its refinement REJECTS some images outright. Measured here: a 1654 x
    2339 render of a valid board fails with the flag and succeeds without it, and the same board
    resized to 1280 wide passes either way. A camera frame at 1280x720 is in the safe region, which
    is why this only showed up on synthetic renders, but a false negative is silent and costs the
    user a frame they thought they captured.
  * plain `findChessboardCornersSB` is more permissive and still sub-pixel.
  * the classic `findChessboardCorners` finds boards the other two refuse, at integer precision, so
    its result is refined with `cornerSubPix` before being used.

Each call returns (found, corners, which_detector) so a caller can report what it took.
"""
from __future__ import annotations

import cv2

import numpy as np

_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
_HUGE_PX = 5000           # above this, CALIB_CB_ACCURACY starts returning wrong grids (see below)
_MAX_SPACING_RATIO = 5.0  # widest corner spacing over narrowest, within one detected board


def plausible(corners, cols: int, rows: int) -> bool:
    """Reject a grid that is geometrically impossible for a real board.

    A detector can return found=True with a collapsed or scrambled grid, which is far worse than
    returning nothing: the frame then poisons the calibration silently. Adjacent corner spacings on
    a genuine board vary smoothly with perspective and lens distortion, so the widest over the
    narrowest is bounded. Measured over 71 synthetic boards spanning a 65 deg lens, a 72 deg wrist
    lens and a true 106 deg fisheye, the worst valid ratio was 2.62, so the 5.0 threshold here keeps
    roughly a factor of two in hand.
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    spacing = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
                              np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    lo = spacing.min()
    return bool(lo > 1e-6 and spacing.max() / lo <= _MAX_SPACING_RATIO)


def find_board(gray, cols: int, rows: int):
    """gray: single-channel image. cols/rows: INNER corners. Returns (found, corners, detector)."""
    # On very large images the ACCURACY refinement is both unreliable and nondeterministic: on a
    # 7016x4961 render it returned a wrong grid in 8 of 10 trials, with different answers between
    # identical runs even single-threaded, while plain SB was correct every time. Camera frames are
    # far below this; it is the printed-board renders used to check the generator that get there.
    levels = [("sb", lambda im: cv2.findChessboardCornersSB(im, (cols, rows)))]
    accuracy = ("sb+accuracy", lambda im: cv2.findChessboardCornersSB(
        im, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY))
    levels.insert(0 if max(gray.shape) <= _HUGE_PX else 1, accuracy)
    levels.append(("classic+subpix", lambda im: cv2.findChessboardCorners(
        im, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)))

    for name, detect in levels:
        found, corners = detect(gray)
        if not found or corners is None or not plausible(corners, cols, rows):
            continue
        if name == "classic+subpix":
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX)
        return True, corners, name

    return False, None, "none"
