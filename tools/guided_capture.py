"""Guided checkerboard capture: which board poses a lens calibration still needs, and banking the frames that fill them.

Used by `live_view.py` (a viewfinder over HTTP) and `calibrate.py` (the guided calibration). It tracks 27 slots - the
3x3 grid of image cells times three tilt bands - and saves a raw frame when the board fills an empty slot while still
and sharp. Why tilt: a board slid around flat reads the same angle in every frame, and the solver then cannot separate
focal length from distance; it returns a confident wrong number with a small reprojection error. So every cell is
wanted at several DIFFERENT tilts, and the overlay shows the range of tilts seen so far.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_detect import find_board, plausible  # noqa: E402

GREEN, AMBER, RED, GREY = (80, 220, 120), (60, 180, 240), (80, 80, 240), (160, 160, 160)

# Bands rather than exact angles because the operator is holding a board by hand; what the solver needs is spread,
# not precision.
BANDS = [(12, 26, "flat"), (26, 38, "mid"), (38, 60, "steep")]


def band_of(t):
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= t < hi:
            return i
    return None


def tilt_deg(corners, cols, rows, sq_mm, W, H):
    """Angle between the board's face and the camera's line of sight.

    Uses a guessed focal length (~width): tilt is a relative indicator here, and a rough K is plenty. The real
    number comes from calibrate_camera.py.
    """
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_mm
    K = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(objp, corners, K, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


class GuidedCapture:
    """Feed it every camera frame (`step`); it draws the guidance on the frame and banks frames into `capture_to`.

    `capture_to=None` makes it a viewfinder only (grid, detection, tilt). `kept`, `slots`, `guide` and `status` say
    where it stands; `done` once `want` frames are banked."""

    def __init__(self, cols=9, rows=6, squares_mm=25.0, capture_to=None, want=27, cooldown=0.8,
                 min_sharpness=60.0, log=print):
        self.cols, self.rows, self.squares_mm = cols, rows, squares_mm
        self.capture_to, self.want, self.cooldown, self.min_sharpness = capture_to, want, cooldown, min_sharpness
        self.log = log
        self.slots = {(c, r, b): 0 for c in range(3) for r in range(3) for b in range(len(BANDS))}
        self.kept = 0
        self.seen = set()
        self.tilts = []          # every tilt observed, to show the RANGE, not just now
        self.hold = []           # recent corner centroids, for a stillness test
        self.cooldown_until = 0.0
        self.n = 0
        self.last = (False, None)
        self.guide, self.status = "", "no board"
        if capture_to:
            os.makedirs(capture_to, exist_ok=True)
            self.kept = len([f for f in os.listdir(capture_to) if f.startswith("frame_")])

    @property
    def done(self) -> bool:
        return bool(self.capture_to) and self.kept >= self.want

    def _next_slot(self, col, row):
        """The slot to ask for next: prefer one in the cell already occupied."""
        here = [k for k, v in self.slots.items() if v == 0 and k[0] == col and k[1] == row]
        if here:
            return here[0]
        rest = [k for k, v in self.slots.items() if v == 0]
        return rest[0] if rest else None

    def _draw_slotmap(self, frame, W):
        """Nine 3-pip groups: one pip per tilt band, lit when that slot is filled."""
        x0, y0, s = W - 190, 44, 14
        for r in range(3):
            for c in range(3):
                for b in range(len(BANDS)):
                    x = x0 + c * (s * 3 + 6) + b * s
                    filled = self.slots[(c, r, b)] > 0
                    cv2.circle(frame, (x, y0 + r * (s + 4)), 4, GREEN if filled else (70, 70, 70), -1 if filled else 1)

    def step(self, frame):
        """One camera frame in, the same frame with the guidance drawn on it out (a frame it banks is saved raw)."""
        H, W = frame.shape[:2]
        raw = frame.copy()          # saved frames must carry no overlay
        self.n += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.n % 3 == 0:         # detection is the expensive part; every 3rd frame keeps the view smooth
            found, corners, _ = find_board(gray, self.cols, self.rows)
            self.last = (found, corners) if found else (False, None)
        found, corners = self.last

        for i in (1, 2):                                    # the 3x3 grid
            cv2.line(frame, (W * i // 3, 0), (W * i // 3, H), GREY, 1)
            cv2.line(frame, (0, H * i // 3), (W, H * i // 3), GREY, 1)

        status, colour = "no board", RED
        guide = ""
        if found and plausible(corners, self.cols, self.rows):
            cv2.drawChessboardCorners(frame, (self.cols, self.rows), corners, True)
            pts = corners.reshape(-1, 2)
            cx, cy = pts.mean(0)
            col, row = min(2, int(cx // (W / 3))), min(2, int(cy // (H / 3)))
            sq = float(np.linalg.norm(pts[1] - pts[0]))
            t = tilt_deg(corners, self.cols, self.rows, self.squares_mm, W, H)
            self.seen.add((col, row))
            if t:
                self.tilts.append(t)
            x0, y0 = col * W // 3, row * H // 3             # the cell the board is in, so position is unambiguous
            cv2.rectangle(frame, (x0, y0), (x0 + W // 3, y0 + H // 3), (90, 200, 90), 2)
            # A single in-band tilt is NOT enough: every frame of the first failed capture read 25 deg (board flat
            # on the table). What bounds the focal length is the SPREAD, so show the range seen so far.
            spread = (max(self.tilts) - min(self.tilts)) if len(self.tilts) > 5 else 0.0
            tcol = GREEN if spread >= 15 else (AMBER if t and 20 <= t <= 45 else RED)
            tmsg = f"tilt {t:3.0f}d" if t else "tilt ?"
            if self.tilts:
                tmsg += f"  range {min(self.tilts):.0f}-{max(self.tilts):.0f}d"
                tmsg += "  VARY THE TILT" if spread < 15 else "  spread ok"
            status, colour = f"cell ({col},{row})  {sq:4.1f} px/sq  {tmsg}", tcol
            if sq < 30:
                status += "  (closer: want 36-100)"

            if self.capture_to:
                b = band_of(t) if t else None
                # Stillness: a hand-held board blurs while moving, and blurred corners quietly inflate the fit.
                self.hold.append((cx, cy))
                self.hold[:] = self.hold[-5:]
                still = len(self.hold) == 5 and max(abs(x - cx) + abs(y - cy) for x, y in self.hold) < 6.0
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var() > self.min_sharpness
                want = self._next_slot(col, row)
                if self.done:
                    guide = f"DONE -- {self.kept} frames, all {len(self.slots)} slots."
                elif want:
                    guide = (f"NEXT: cell ({want[0]},{want[1]}) tilt {BANDS[want[2]][2]} "
                             f"({BANDS[want[2]][0]}-{BANDS[want[2]][1]}d)")
                else:
                    # Every slot has one frame but `want` asks for more: top up the thinnest slots rather than stall.
                    guide = f"TOP-UP: any pose ({self.kept}/{self.want}) -- coverage is already complete"
                fills = b is not None and (
                    self.slots.get((col, row, b), 1) == 0
                    or (want is None and self.slots.get((col, row, b), 99) == min(self.slots.values())))
                now = time.time()
                if fills and still and sharp and now > self.cooldown_until and not self.done:
                    name = f"frame_{self.kept:03d}.png"
                    cv2.imwrite(os.path.join(self.capture_to, name), raw)
                    self.slots[(col, row, b)] += 1
                    self.kept += 1
                    self.cooldown_until = now + self.cooldown
                    self.log(f"  [{self.kept}/{self.want}] {name} | cell ({col},{row}) tilt {t:.0f}d {BANDS[b][2]} | "
                             f"{sq:.0f} px/sq | slots filled {sum(1 for v in self.slots.values() if v)}/"
                             f"{len(self.slots)}")
                    cv2.rectangle(frame, (0, 0), (W, H), GREEN, 12)   # flash
                elif not fills and b is not None:
                    guide += "   (this pose is already banked)"
                elif not still:
                    guide += "   -- hold still"
                elif not sharp:
                    guide += "   -- too blurry"
        elif self.capture_to:
            want = self._next_slot(1, 1)
            guide = ("DONE" if self.done else
                     f"NEXT: cell ({want[0]},{want[1]}) tilt {BANDS[want[2]][2]} ({BANDS[want[2]][0]}-{BANDS[want[2]][1]}d)"
                     if want else "TOP-UP: any pose")

        self.status, self.guide = status, guide
        cv2.rectangle(frame, (0, 0), (W, 34), (20, 24, 30), -1)
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        right = f"{self.kept}/{self.want} kept" if self.capture_to else f"cells seen {len(self.seen)}/9"
        cv2.putText(frame, right, (W - 190, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    GREEN if (self.done if self.capture_to else len(self.seen) == 9) else AMBER, 2)
        if guide:
            cv2.rectangle(frame, (0, H - 38), (W, H), (20, 24, 30), -1)
            cv2.putText(frame, guide, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2)
        if self.capture_to:
            self._draw_slotmap(frame, W)
        return frame
