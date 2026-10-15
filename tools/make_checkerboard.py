"""Generate the calibration checkerboard, at an exact physical size, as PDF and SVG.

A calibration target is a measuring instrument, so the only thing that matters is that the squares
come out the size they claim. Both files are written with explicit physical units and no scaling
anywhere in the pipeline: the PDF is emitted directly, rectangle by rectangle, in PostScript points
(72 per inch), rather than converted from the SVG by a renderer that might resample it.

    ./.venv/bin/python tools/make_checkerboard.py --out assets/checkerboard_9x6_25mm

writes `<out>.pdf` and `<out>.svg`. Print the PDF at 100% scale, "actual size", NOT "fit to page" --
every printer's default is to shrink slightly for its unprintable margin, which silently rescales
the board. Then MEASURE a square anyway and pass what you measure as `--squares-mm` to the solvers;
the print is the target, your ruler is the ground truth.

Defaults are 10 x 7 squares of 25 mm, which is 9 x 6 INNER corners (the count the detectors want)
on A4 landscape. `--squares-mm 30 --cols 8 --rows 6` and similar still fit; the tool refuses a
combination that would run off the page.

The output was checked by rasterising the PDF back at 300 dpi, re-detecting the board and measuring
the corner spacing: a requested 25 mm square came back as 24.998 mm, i.e. the residual is corner
detection noise, not scale error. The tool does not re-run that check itself; it needs poppler.
"""
from __future__ import annotations
import argparse, os, textwrap

MM = 72.0 / 25.4          # PostScript points per millimetre


def draw_squares(cols, rows, sq, ox, oy, emit, phase=0):
    """Every other square. `emit(x, y, w, h)` works in the caller's own units and axis direction.

    `phase` exists because PDF counts y UP from the page bottom and SVG counts it DOWN from the top,
    so row 0 is the bottom row in one file and the top row in the other. With an ODD row count the
    pattern is self-mirroring and the two agree by luck; with an EVEN row count they come out colour
    inverted -- the same board in negative. Measured before this argument existed: 99.5% of pixels
    disagreed at 8x6. The SVG therefore passes phase=(rows-1)%2, which is 0 exactly when the flip is
    harmless."""
    for r in range(rows):
        for c in range(cols):
            if (r + c + phase) % 2 == 0:
                emit(ox + c * sq, oy + r * sq, sq, sq)


CAP_MARGIN_MM = 10.0        # the caption starts here, at the PAGE edge, not at the board's edge


def layout_caption(text, page_w_mm, cap=9.0):
    """Pick a font size and wrap the caption so it cannot run off the page. Returns (lines, size_pt).

    Helvetica averages 0.474 em per character (measured with ghostscript's stringwidth), so 0.5 is a
    safe estimate. The width available is the PAGE minus its margins: an earlier version budgeted the
    page width but drew the text from the board's left edge, which on a small board sits far inside
    the page, and the line was then truncated at the trim edge."""
    avail_pt = (page_w_mm - 2 * CAP_MARGIN_MM) * MM
    fs = max(4.0, min(cap, avail_pt / (0.5 * 62)))            # aim for ~62 characters per line
    per_line = max(12, int(avail_pt / (0.5 * fs)))
    return textwrap.wrap(text, per_line) or [""], fs


def write_pdf(path, cols, rows, sq_mm, page_mm, caption):
    pw, ph = page_mm[0] * MM, page_mm[1] * MM
    sq = sq_mm * MM
    bw, bh = cols * sq, rows * sq
    ox, oy = (pw - bw) / 2.0, (ph - bh) / 2.0 + 4 * MM       # PDF y runs UP, so the caption sits below

    ops = ["0 g"]                                            # fill colour: black
    draw_squares(cols, rows, sq, ox, oy, lambda x, y, w, h: ops.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re f"))
    lines, fs = layout_caption(caption, page_mm[0])
    ops += ["0.2 w", f"{ox:.4f} {oy:.4f} {bw:.4f} {bh:.4f} re S", f"BT /F1 {fs:.2f} Tf"]
    for i, line in enumerate(lines):                         # PDF y runs up, so later lines sit lower
        ops += [f"1 0 0 1 {CAP_MARGIN_MM * MM:.4f} {oy - (7 + 4.5 * i) * MM:.4f} Tm",
                f"({line.replace(chr(92), chr(92)*2).replace('(', chr(92) + '(').replace(')', chr(92) + ')')}) Tj"]
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.4f} {ph:.4f}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode("latin-1"),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode()
    open(path, "wb").write(bytes(out))


def write_svg(path, cols, rows, sq, page, caption):
    W, H = page
    _lines, _fs = layout_caption(caption, W)
    bw, bh = cols * sq, rows * sq
    ox, oy = (W - bw) / 2.0, (H - bh) / 2.0 - 4              # SVG y runs DOWN
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    draw_squares(cols, rows, sq, ox, oy,
                 lambda x, y, w, h: parts.append(f'<rect x="{x:.4f}" y="{y:.4f}" width="{w}" height="{h}" fill="#000000"/>'),
                 phase=(rows - 1) % 2)                        # undo the y-axis flip; see draw_squares
    parts += [f'<rect x="{ox:.4f}" y="{oy:.4f}" width="{bw}" height="{bh}" fill="none" stroke="#000000" stroke-width="0.2"/>',
              *[f'<text x="{CAP_MARGIN_MM:.1f}" y="{oy + bh + 7 + 4.5 * i:.1f}" font-family="Helvetica,Arial,sans-serif" '
                f'font-size="{_fs / MM:.2f}" fill="#000000">{line.replace("&", "&amp;").replace("<", "&lt;")}</text>'
                for i, line in enumerate(_lines)], '</svg>']
    open(path, "w").write("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/checkerboard_9x6_25mm", help="path without extension")
    p.add_argument("--cols", type=int, default=10, help="SQUARES across (inner corners = this minus 1)")
    p.add_argument("--rows", type=int, default=7, help="SQUARES down (inner corners = this minus 1)")
    p.add_argument("--squares-mm", type=float, default=25.0)
    p.add_argument("--page-mm", default="297x210", help="page size in mm, default A4 landscape")
    a = p.parse_args()

    try:
        W, H = (float(v) for v in a.page_mm.lower().split("x"))
        if W <= 0 or H <= 0: raise ValueError
    except ValueError:
        raise SystemExit(f"[FATAL] --page-mm must be WIDTHxHEIGHT in mm, e.g. 297x210; got {a.page_mm!r}")
    if a.cols < 3 or a.rows < 3:
        raise SystemExit(f"[FATAL] {a.cols}x{a.rows} squares is not a usable board: it would have "
                         f"{a.cols-1}x{a.rows-1} inner corners. Use at least 3x3 squares (4x4 is the practical floor).")
    if a.squares_mm <= 0:
        raise SystemExit(f"[FATAL] --squares-mm must be positive; got {a.squares_mm:g}")
    bw, bh = a.cols * a.squares_mm, a.rows * a.squares_mm
    if bw > W - 10 or bh > H - 30:
        raise SystemExit(f"[FATAL] a {bw:g}x{bh:g} mm board does not fit on {W:g}x{H:g} mm with room for the "
                         "two caption lines and a white border. Use fewer squares, a smaller --squares-mm, or a bigger page.")
    caption = (f"{a.cols} x {a.rows} squares = {a.cols-1} x {a.rows-1} inner corners, {a.squares_mm:g} mm nominal. "
               "PRINT AT 100% (actual size, NOT fit-to-page), then MEASURE a square and pass it as --squares-mm.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_pdf(a.out + ".pdf", a.cols, a.rows, a.squares_mm, (W, H), caption)
    write_svg(a.out + ".svg", a.cols, a.rows, a.squares_mm, (W, H), caption)
    print(f"[board] {a.out}.pdf and .svg | {a.cols}x{a.rows} squares = {a.cols-1}x{a.rows-1} inner corners "
          f"@ {a.squares_mm:g} mm | board {bw:g}x{bh:g} mm on a {W:g}x{H:g} mm page")
    print(f"        solve with:  --cols {a.cols-1} --rows {a.rows-1} --squares-mm <measured>")


if __name__ == "__main__":
    main()
