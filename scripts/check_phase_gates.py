#!/usr/bin/env python3
"""Count consecutive iterations passing Dominion Phase 3 graduation gates.

Reads the live losses.jsonl from the RunPod pod over SSH (always fresher than
the repo mirror). Gates per docs/plans/dominion-training-plan.md:
    avg_provinces > 3.0
    draw_rate < 0.05
    avg_turns < 40
All three must hold for 20 consecutive iters to graduate to Phase 4.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

DEFAULT_SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
               "-i", "~/.ssh/id_ed25519", "-p", "26242", "root@38.147.83.30"]
POD_LOSSES = "/workspace/dominion_data/losses.jsonl"
TARGET_STREAK = 20

GATES = {
    "avg_provinces > 3.0":  lambda d: d.get("avg_provinces", 0.0) > 3.0,
    "draw_rate < 0.05":     lambda d: d.get("draw_rate", 1.0) < 0.05,
    "avg_turns < 40":       lambda d: d.get("avg_turns", 99) < 40,
}


def fetch_losses(tail: int, ssh_cmd: list[str]) -> list[dict]:
    cmd = ssh_cmd + [f"tail -n {tail} {POD_LOSSES}"]
    raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail", type=int, default=100, help="Iterations to fetch from pod (default: 100)")
    ap.add_argument("--local", metavar="PATH", help="Read from a local losses.jsonl instead of SSH")
    args = ap.parse_args()

    if args.local:
        with open(args.local) as f:
            records = [json.loads(l) for l in f if l.strip()]
        records = records[-args.tail:]
    else:
        records = fetch_losses(args.tail, DEFAULT_SSH)

    if not records:
        print("no records", file=sys.stderr)
        return 2

    streak = 0
    last_fail = None
    for r in records:
        failures = [label for label, ok in GATES.items() if not ok(r)]
        if failures:
            streak = 0
            last_fail = (r.get("iteration"), failures, r)
        else:
            streak += 1

    latest = records[-1]
    print(f"iter {latest.get('iteration')} | "
          f"prov={latest.get('avg_provinces'):.2f} "
          f"turns={latest.get('avg_turns'):.1f} "
          f"draw={latest.get('draw_rate'):.3f}")
    print(f"consecutive passing iters: {streak}/{TARGET_STREAK}")
    if last_fail and streak < TARGET_STREAK:
        it, fails, rec = last_fail
        print(f"last fail @ iter {it}: {', '.join(fails)}")
        print(f"  values: prov={rec.get('avg_provinces')} "
              f"turns={rec.get('avg_turns')} draw={rec.get('draw_rate')}")
    if streak >= TARGET_STREAK:
        print("STATUS: Phase 3 gates HELD for 20+ iters — ready for Phase 4 review")
        return 0
    print("STATUS: not yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
