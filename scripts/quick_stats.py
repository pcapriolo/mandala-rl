#!/usr/bin/env python3
"""
Quick training statistics - simple table view.
"""
import torch
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_time_ago(timestamp):
    """Format timestamp as time ago."""
    now = datetime.now()
    diff = now - timestamp

    if diff.days > 0:
        return f"{diff.days}d ago"
    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours}h ago"
    minutes = (diff.seconds % 3600) // 60
    return f"{minutes}m ago"


def main():
    checkpoint_dir = Path("data/checkpoints")
    elo_file = Path("data/elo_ratings.json")

    print("=" * 80)
    print(" " * 20 + "🎯 MANDALA RL TRAINING STATUS")
    print("=" * 80)

    # Current checkpoint
    latest = checkpoint_dir / "model_latest.pt"
    if latest.exists():
        try:
            ckpt = torch.load(latest, map_location='cpu', weights_only=False)
            iteration = ckpt.get('iteration', 0)
            games = ckpt.get('total_games', 0)
            buffer_size = len(ckpt.get('replay_buffer', []))
            timestamp = datetime.fromtimestamp(latest.stat().st_mtime)

            print(f"\n📊 Current Progress:")
            print(f"   Iteration:      {iteration}")
            print(f"   Total Games:    {games:,}")
            print(f"   Buffer Size:    {buffer_size:,} examples")
            print(f"   Avg per Game:   {buffer_size/games:.1f} examples" if games > 0 else "")
            print(f"   Last Updated:   {timestamp.strftime('%Y-%m-%d %H:%M')} ({format_time_ago(timestamp)})")

        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
    else:
        print("⚠️  No checkpoint found")
        iteration = 0

    # Checkpoint history
    checkpoints = sorted(checkpoint_dir.glob("model_iter_*.pt"))
    if checkpoints and len(checkpoints) > 1:
        print(f"\n📈 Checkpoint History: ({len(checkpoints)} saved)")
        print(f"   {'Iter':<6} {'Size':<12} {'Age':<12}")
        print(f"   {'-'*6} {'-'*12} {'-'*12}")

        for ckpt_path in checkpoints[-5:]:  # Last 5
            try:
                iter_num = int(ckpt_path.stem.split('_')[-1])
                size_mb = ckpt_path.stat().st_size / (1024 * 1024)
                timestamp = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
                age = format_time_ago(timestamp)

                print(f"   {iter_num:<6} {size_mb:>6.1f} MB   {age:<12}")
            except Exception:
                continue

    # Elo ratings
    if elo_file.exists():
        try:
            with open(elo_file, 'r') as f:
                elo_data = json.load(f)

            ratings = []
            for model_id, rating in elo_data.items():
                if model_id.startswith('iter_'):
                    iter_num = int(model_id.split('_')[1])
                    ratings.append((iter_num, rating))

            if ratings:
                ratings.sort()

                print(f"\n🏆 Elo Ratings:")
                print(f"   {'Iter':<6} {'Elo':<8} {'Change':<10}")
                print(f"   {'-'*6} {'-'*8} {'-'*10}")

                prev_rating = None
                for i, (iter_num, rating) in enumerate(ratings[-10:]):  # Last 10
                    if prev_rating is not None:
                        change = rating - prev_rating
                        change_str = f"{change:+.1f}"
                    else:
                        change_str = "—"

                    print(f"   {iter_num:<6} {rating:>7.1f}  {change_str:<10}")
                    prev_rating = rating

                if len(ratings) >= 2:
                    total_change = ratings[-1][1] - ratings[0][1]
                    avg_change = total_change / (len(ratings) - 1)
                    print(f"\n   Total gain: {total_change:+.1f} points")
                    print(f"   Avg per eval: {avg_change:+.1f} points")

        except Exception as e:
            print(f"⚠️  Error loading Elo data: {e}")
    else:
        print(f"\n🏆 Elo Ratings:")
        print(f"   No evaluations yet (start at iteration {max(10, iteration+1)})")

    # Training efficiency
    print(f"\n⚡ Training Efficiency:")
    if iteration > 0 and buffer_size > 0:
        examples_per_iter = buffer_size / iteration
        games_per_iter = games / iteration if games > 0 else 0

        print(f"   {examples_per_iter:,.0f} examples per iteration")
        print(f"   {games_per_iter:.0f} games per iteration")

        if games_per_iter > 0:
            hours_per_iter = 7.0  # Rough estimate
            print(f"   ~{hours_per_iter:.1f} hours per iteration (estimated)")
    else:
        print(f"   Starting training...")

    # Health check
    print(f"\n🏥 Quick Health Check:")
    checks = []

    if iteration > 0:
        checks.append(("✅", f"Training active (iteration {iteration})"))
    else:
        checks.append(("⚠️ ", "Training not started"))

    if buffer_size > 20000:
        checks.append(("✅", f"Good data diversity ({buffer_size:,} examples)"))
    elif buffer_size > 10000:
        checks.append(("🟡", f"Building data ({buffer_size:,} examples)"))
    else:
        checks.append(("🔴", f"Low data ({buffer_size:,} examples)"))

    if elo_file.exists():
        checks.append(("✅", "Elo tracking enabled"))
    else:
        checks.append(("⏳", "Awaiting first evaluation"))

    for emoji, message in checks:
        print(f"   {emoji} {message}")

    print("\n" + "=" * 80)
    print("Run training_dashboard.py for detailed metrics and explanations")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
