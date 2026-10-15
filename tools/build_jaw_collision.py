#!/usr/bin/env python3
"""Convex decomposition of the SO-101 jaws for robotd's hull checks (sprint 2026-10-05, task 3, decision D3).

The jaws are concave (their single convex hulls are 2.3-2.6x their volume), so one hull per jaw is too bulky for
the obstacle and other-arm checks (for the table it is exact). This splits each jaw into convex pieces:

  coacd (default) - CoACD cuts the real mesh with planes; each piece is the convex hull of its part of the mesh,
                    so the union is never smaller than the jaw - the guarantee a safety check needs.
  vhacd           - V-HACD 4 with the PhysX parameters the RTX/Isaac asset uses (maxConvexHulls 32, hullVertexLimit
                    64, voxelResolution 500000, errorPercentage 10, shrinkWrap false; PhysX's minThickness has no
                    equivalent) - closest to Isaac's decomposition, for comparison. Voxel-based: can under-cover.

Coverage is measured, not assumed: for every mesh vertex, how far it lies outside the union of the pieces. If any
does, each piece is inflated (its faces pushed out by that distance plus 0.1 mm) and coverage re-measured.

Meshes: the upstream SO-ARM100 jaws (`moving_jaw_so101_v1.stl`, `wrist_roll_follower_so101_v1.stl` from the fetched
assets), in their own mesh frames (metres); the scene places the pieces exactly where the MJCF places the mesh.

    .venv/bin/python tools/build_jaw_collision.py                     # CoACD -> stack/assets/gripper_collision/
    .venv/bin/python tools/build_jaw_collision.py --method vhacd --out /tmp/x.json
    .venv/bin/python tools/build_jaw_collision.py --compare RTX.json  # report vs another decomposition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "stack" / "assets" / "gripper_collision" / "so101_jaws.json"
JAWS = ("moving_jaw_so101_v1", "wrist_roll_follower_so101_v1")
RTX_PHYSX = {"maxConvexHulls": 32, "hullVertexLimit": 64, "minThickness": 0.001, "voxelResolution": 500000,
             "errorPercentage": 10.0, "shrinkWrap": False}


def _hull_planes(V):
    from scipy.spatial import ConvexHull
    return ConvexHull(V).equations                      # (F, 4): n.x + d <= 0 inside, n unit


def outside_distance(points, hulls) -> np.ndarray:
    """For each point, how far outside the union of the convex hulls it lies (0 = inside some hull)."""
    best = np.full(len(points), np.inf)
    P = np.c_[points, np.ones(len(points))]
    for H in hulls:
        eq = _hull_planes(np.asarray(H, float))
        d = np.max(P @ eq.T, axis=1)                       # > 0: outside this hull (by at least that much)
        best = np.minimum(best, np.maximum(d, 0.0))
    return best


def inflate(H, by):
    """Move every face plane of a convex piece outward by `by` (the hull of the offset planes' intersection)."""
    from scipy.spatial import HalfspaceIntersection
    H = np.asarray(H, float)
    eq = _hull_planes(H).copy()
    eq[:, 3] -= by                                        # n.x + d - by <= 0: each face moved out by `by`
    c = H.mean(axis=0)
    return HalfspaceIntersection(eq, c).intersections


def decompose(mesh, method):
    if method == "coacd":
        import coacd
        # no decimation: decimating a piece to a vertex limit cuts it below the surface (measured: up to 3.2 mm);
        # undecimated, each piece is the exact hull of its part of the mesh
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), max_convex_hull=RTX_PHYSX["maxConvexHulls"],
                                decimate=False, seed=0)
        params = {"threshold": 0.05, "max_convex_hull": 32, "decimate": False, "seed": 0}
        return [np.asarray(v, float) for v, _ in parts], params
    import vhacdx
    parts = vhacdx.compute_vhacd(mesh.vertices, mesh.faces.astype(np.uint32), maxConvexHulls=32, resolution=500000,
                                 minimumVolumePercentErrorAllowed=10.0, shrinkWrap=False, maxNumVerticesPerCH=64)
    return [np.asarray(v, float) for v, _ in parts], {"vhacdx": "V-HACD 4", **RTX_PHYSX}


def build(method: str, raw: bool = False) -> dict:
    import trimesh

    from stack.twin_mujoco.scene import assets_dir

    out = {"method": method, "source": "SO-ARM100 @ 5f6d2b8 Simulation/SO101/assets (tools/fetch_so101_assets.py)",
           "frame": "each jaw's own STL frame, metres; placed like the MJCF places that mesh",
           "built": time.strftime("%Y-%m-%d %H:%M"), "parts": {}}
    for name in JAWS:
        mesh = trimesh.load(assets_dir() / "assets" / f"{name}.stl")
        t0 = time.time()
        hulls, params = decompose(mesh, method)
        miss = outside_distance(mesh.vertices, hulls)
        inflated = 0.0
        if miss.max() > 1e-7 and not raw:                 # inflate only the pieces nearest the uncovered vertices
            inflated = float(miss.max()) + 1e-4
            P = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
            dist = np.stack([np.max(P @ _hull_planes(np.asarray(h)).T, axis=1) for h in hulls])   # (pieces, verts)
            owner = set(np.argmin(dist[:, miss > 1e-7], axis=0).tolist())
            hulls = [inflate(h, inflated) if i in owner else h for i, h in enumerate(hulls)]
            miss = outside_distance(mesh.vertices, hulls)
        vol = sum(trimesh.Trimesh(h).convex_hull.volume for h in hulls)
        out["params"] = params
        out["parts"][name] = {
            "hulls": [np.round(h, 6).tolist() for h in hulls],
            "pieces": len(hulls), "vertices": int(sum(len(h) for h in hulls)),
            "max_vertex_outside_mm": round(float(miss.max()) * 1000, 4), "inflated_mm": round(inflated * 1000, 3),
            "volume_ratio": {"pieces_over_mesh": round(vol / mesh.volume, 3),
                             "single_hull_over_mesh": round(mesh.convex_hull.volume / mesh.volume, 3)},
            "seconds": round(time.time() - t0, 1)}
    return out


def _from_rtx_export(b: dict, urdf: Path) -> dict:
    """The RTX hand-off format (parts: a list, vertices in robot_renamed.urdf link frames) -> ours (keyed by mesh,
    vertices in each jaw's STL frame), using that URDF's placement of the mesh in its link."""
    import xml.etree.ElementTree as ET

    from stack.rig.rig import _rpy
    root = ET.parse(urdf).getroot()
    place = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            o = v.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, :3] = _rpy(*[float(x) for x in o.get("rpy", "0 0 0").split()])
                T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
            place[(link.get("name"), Path(m.get("filename")).stem)] = T
    out = {"parts": {}}
    for p in b["parts"]:
        mesh = p["part"] if p["part"].endswith("_v1") else p["part"]
        T = place.get((p["link"], mesh))
        if T is None:
            raise SystemExit(f"no placement of {mesh} in {p['link']} in {urdf}")
        Ti = np.linalg.inv(T)
        out["parts"][mesh] = {"hulls": [(np.c_[np.asarray(h, float), np.ones(len(h))] @ Ti.T)[:, :3] for h in p["hulls"]]}
    return out


def compare(a: dict, b: dict, urdf: str | None = None):
    import trimesh

    from stack.twin_mujoco.scene import assets_dir
    if isinstance(b.get("parts"), list):
        if not urdf:
            raise SystemExit("this decomposition is in the Digital Twin's export format: give --urdf, the Digital "
                             "Twin's robot_renamed.urdf it was made against")
        b = _from_rtx_export(b, Path(urdf).expanduser())
    for name in JAWS:
        pa, pb = a["parts"].get(name), (b.get("parts") or {}).get(name)
        if not pa or not pb:
            print(f"{name}: missing in one of the two"); continue
        mesh = trimesh.load(assets_dir() / "assets" / f"{name}.stl")
        for label, p in (("ours", pa), ("other", pb)):
            hulls = [np.asarray(h, float) for h in p["hulls"]]
            miss = outside_distance(mesh.vertices, hulls)
            vol = sum(trimesh.Trimesh(h).convex_hull.volume for h in hulls)
            print(f"{name:30s} {label:5s} pieces {len(hulls):3d}  vertices outside max {miss.max()*1000:6.2f} mm "
                  f"({(miss > 1e-4).mean()*100:4.1f}% > 0.1 mm)  volume / mesh {vol / mesh.volume:5.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("coacd", "vhacd"), default="coacd")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--raw", action="store_true", help="no inflation: report the method's own coverage")
    ap.add_argument("--compare", help="another decomposition (same JSON shape, e.g. the RTX export) vs --out")
    ap.add_argument("--urdf", help="with --compare of a Digital Twin export: the URDF its vertices are in (the "
                                   "Digital Twin's robot_renamed.urdf; a separate repository)")
    a = ap.parse_args()
    if a.compare:
        compare(json.loads(Path(a.out).read_text()), json.loads(Path(a.compare).expanduser().read_text()), a.urdf)
        return 0
    res = build(a.method, raw=a.raw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res) + "\n")
    for name, p in res["parts"].items():
        print(f"{name:30s} pieces {p['pieces']:3d} vertices {p['vertices']:5d} outside {p['max_vertex_outside_mm']} mm "
              f"inflated {p['inflated_mm']} mm  volume/mesh {p['volume_ratio']['pieces_over_mesh']} "
              f"(single hull {p['volume_ratio']['single_hull_over_mesh']})  {p['seconds']} s")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
