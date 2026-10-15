#!/usr/bin/env python3
"""Contact thresholds from free-motion samples (rig check 13).

robotd in `contact.mode: record_only` logs every tick's lag (|command - measured|, deg) and current (mA) per arm
joint as `contact_samples` events (tools/contact_exercise.py produces them). This reads them, prints per-joint
percentiles overall and per speed, proposes thresholds above normal motion, and replays the contact monitor's
rule (lag AND current over threshold for `ticks` consecutive ticks) to count the blocks each setting would
have raised in this free motion - which should be none.

    .venv/bin/python tools/contact_thresholds.py --since 14:40            # today's log, from 14:40
    .venv/bin/python tools/contact_thresholds.py --log $ROBOT_USE_HOME/agent_logs/<date>.jsonl --arm right
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LAG_MARGIN, LAG_FLOOR = 1.5, 2.0               # proposed lag = p99.9 x 1.5 + 2 deg
CUR_MARGIN, CUR_FLOOR = 1.3, 100.0             # proposed current = p99.9 x 1.3 + 100 mA


def load(paths, since: float, arm: str | None):
    out = []
    for p in paths:
        for line in Path(p).expanduser().read_text().splitlines():
            if '"contact_samples"' not in line:
                continue
            r = json.loads(line)
            if r.get("event") != "contact_samples" or r["ts"] < since or (arm and r["arm"] != arm):
                continue
            out.append(r)
    return out


def blocks(rows: np.ndarray, lag_thr, cur_thr, ticks: int) -> int:
    """How many joints-runs would have tripped the monitor (lag AND current over, `ticks` in a row)."""
    n = rows.shape[1] // 2
    over = (rows[:, :n] > lag_thr) & (rows[:, n:] > cur_thr)
    hits = 0
    for j in range(n):
        run = 0
        for v in over[:, j]:
            run = run + 1 if v else 0
            if run == ticks:
                hits += 1
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from stack import paths
    ap.add_argument("--log", nargs="*", default=[str(paths.agent_logs() / f"{datetime.now():%Y-%m-%d}.jsonl")],
                    help="robotd's event logs (default: today's, in $ROBOT_USE_HOME/agent_logs)")
    ap.add_argument("--since", default="00:00", help="HH:MM today, or a unix time")
    ap.add_argument("--arm", choices=("left", "right"))
    ap.add_argument("--config", default=str(ROOT / "stack" / "robotd.yaml"),
                    help="the robotd config to compare against (the one robotd will run with afterwards)")
    a = ap.parse_args()
    since = (float(a.since) if a.since.replace(".", "").isdigit() and len(a.since) > 5
             else time.mktime(datetime.combine(datetime.now().date(),
                                               datetime.strptime(a.since, "%H:%M").time()).timetuple()))
    recs = load(a.log, since, a.arm)
    if not recs:
        print("no contact_samples events (was robotd in contact.mode record_only?)")
        return 1
    joints = recs[0]["joints"]
    n = len(joints)
    allrows = np.vstack([np.asarray(r["rows"], float) for r in recs if r["rows"]])
    secs = len(allrows) / 50.0
    print(f"{len(recs)} motions, {len(allrows)} ticks ({secs / 60:.1f} min of motion), arms "
          f"{sorted({r['arm'] for r in recs})}\n")
    print(f"{'joint':14s} {'lag p50':>8s} {'p99':>6s} {'p99.9':>6s} {'max':>6s} |"
          f" {'mA p50':>7s} {'p99':>6s} {'p99.9':>6s} {'max':>6s}")
    p999 = np.zeros((2, n))
    for j in range(n):
        lag, cur = allrows[:, j], allrows[:, n + j]
        pl, pc = np.percentile(lag, [50, 99, 99.9]), np.percentile(cur, [50, 99, 99.9])
        p999[:, j] = pl[2], pc[2]
        print(f"{joints[j]:14s} {pl[0]:8.2f} {pl[1]:6.2f} {pl[2]:6.2f} {lag.max():6.2f} |"
              f" {pc[0]:7.0f} {pc[1]:6.0f} {pc[2]:6.0f} {cur.max():6.0f}")
    print("\nby speed (p99.9 lag deg / mA, worst joint):")
    for sp in sorted({r.get("speed") for r in recs}, key=lambda v: (v is None, v)):
        rows = np.vstack([np.asarray(r["rows"], float) for r in recs if r.get("speed") == sp and r["rows"]])
        print(f"  speed {sp}: {np.percentile(rows[:, :n], 99.9, axis=0).max():.2f} deg, "
              f"{np.percentile(rows[:, n:], 99.9, axis=0).max():.0f} mA  ({len(rows)} ticks)")
    cfg = yaml.safe_load(Path(a.config).expanduser().read_text())["contact"]
    lag_j = p999[0] * LAG_MARGIN + LAG_FLOOR
    cur_j = p999[1] * CUR_MARGIN + CUR_FLOOR
    prop = (float(np.ceil(lag_j.max())), float(np.ceil(cur_j.max() / 50) * 50))
    print(f"\nper joint, p99.9 x margin + floor: lag {np.round(lag_j, 1).tolist()} deg, current "
          f"{np.round(cur_j).tolist()} mA")
    print(f"robotd's contact thresholds are one value for all joints. Proposed: lag_deg {prop[0]:g}, "
          f"current_ma {prop[1]:g} (current config: lag_deg {cfg['lag_deg']}, current_ma {cfg['current_ma']}, "
          f"ticks {cfg['ticks']})")
    for label, (lt, ct) in (("current config", (cfg["lag_deg"], cfg["current_ma"])), ("proposed", prop)):
        hits = sum(blocks(np.asarray(r["rows"], float), lt, ct, int(cfg["ticks"])) for r in recs if r["rows"])
        print(f"  replay, {label}: {hits} false block(s) in this free motion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
