#!/usr/bin/env python3
"""Quick check of Dominion training metrics. Run on RunPod."""
import json
import sys
import time
from collections import Counter

with open("/workspace/dominion_data/losses.jsonl") as f:
    all_lines = f.readlines()

all_rows = [json.loads(l) for l in all_lines if l.strip()]

# Detect duplicates
iters = [r["iteration"] for r in all_rows]
dupes = {k: v for k, v in Counter(iters).items() if v > 1}
if dupes:
    dup_range = sorted(dupes.keys())
    print(f"DUPLICATE ITERS: {dup_range[0]}-{dup_range[-1]} ({len(dupes)} iters)")
    # Show last occurrence of each duplicate (the newer data)
    print("Showing LATEST entries only\n")

# Deduplicate: keep last occurrence of each iter
seen = {}
for i, r in enumerate(all_rows):
    seen[r["iteration"]] = r
rows = sorted(seen.values(), key=lambda r: r["iteration"])

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rows = rows[-n:]

hdr = f"{'Iter':>6} {'Len':>5} {'Draw':>5} {'Prov':>5} {'Duch':>5} {'MProv%':>6} {'Waste':>5} {'Val':>6} {'Pol':>6}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['iteration']:>6} "
        f"{r.get('avg_len',0):>5.0f} "
        f"{r['draw_rate']:>5.3f} "
        f"{r.get('avg_provinces',0):>5.2f} "
        f"{r.get('avg_duchies',0):>5.2f} "
        f"{r.get('mcts_province_pct',0):>6.1f} "
        f"{r.get('avg_coins_wasted',0):>5.2f} "
        f"{r['value']:>6.4f} "
        f"{r['policy']:>6.4f}"
    )

# Heartbeat
try:
    with open("/workspace/dominion_data/heartbeat.json") as f:
        hb = json.loads(f.read())
    age = time.time() - hb["timestamp"]
    print(f"\nHeartbeat: iter {hb['iteration']} phase={hb['phase']} {hb['detail']} "
          f"total_games={hb['total_games']} age={age/60:.1f}min")
except Exception:
    print("\nNo heartbeat")
