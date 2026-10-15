#!/usr/bin/env python3
"""Fetch the upstream SO-101 MuJoCo model with its meshes (19 MB, not kept in git) into ~/.cache/robot-use.

    .venv/bin/python tools/fetch_so101_assets.py                 # git clone, pinned commit
    .venv/bin/python tools/fetch_so101_assets.py --from DIR      # copy an existing SO-ARM100/Simulation/SO101

Pinned to the same TheRobotStudio/SO-ARM100 commit as stack/assets (Apache-2.0). The MuJoCo twin
(stack/twin_mujoco) reads it from there, or from $ROBOT_USE_SO101_ASSETS.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/TheRobotStudio/SO-ARM100.git"
COMMIT = "5f6d2b876a53a4872e405b991dd925556c9e38a4"
DEST = Path("~/.cache/robot-use/so101_mujoco").expanduser()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="an existing SO-ARM100/Simulation/SO101 directory")
    a = ap.parse_args()
    if (DEST / "so101_new_calib_camera.xml").exists() and (DEST / "assets").is_dir():
        print(f"already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    old = Path("~/.cache/so2x/so101_mujoco").expanduser()       # fetched before the rename (D15): reuse it
    if (old / "so101_new_calib_camera.xml").exists() and (old / "assets").is_dir():
        shutil.copytree(old, DEST)
        print(f"copied from {old}: {DEST}")
        return 0
    if a.src:
        src = Path(a.src).expanduser()
    else:
        tmp = DEST.parent / "SO-ARM100.clone"
        if not tmp.exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", REPO_URL, str(tmp)], check=True)
            subprocess.run(["git", "-C", str(tmp), "sparse-checkout", "set", "Simulation/SO101"], check=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", "-q", COMMIT], check=True)
        src = tmp / "Simulation" / "SO101"
    shutil.copytree(src, DEST)
    (DEST / "SOURCE").write_text(f"{REPO_URL} @ {COMMIT} (Simulation/SO101), Apache-2.0\n")
    print(f"installed {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
