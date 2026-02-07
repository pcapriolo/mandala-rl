#!/usr/bin/env python3
"""
Continuous distributed augmentation loop.

Fires webhooks one at a time, waits for completion, merges results, and repeats.
"""
import argparse
import requests
import time
import pickle
import torch
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.training.replay_buffer import ReplayBuffer

# Configuration
WEBHOOK_URL = "https://f75ecf45-b024-4693-bc22-dd0b2f1056be-00-20vc6rnxkl2w7.kirk.replit.dev/api/webhooks/launch/cb3dc298-cb9e-4eb2-994e-c402b935484a"
CALLBACK_URL = "https://four-cobras-appear.loca.lt/callback/results"
CHECKPOINT_URL = "https://mandala-checkpoints.loca.lt/model_latest_lightweight.pt"

def trigger_workflow(total_games: int, num_workers: int, mcts_sims: int = 100):
    """Trigger a distributed self-play workflow."""

    payload = {
        "checkpoint_url": CHECKPOINT_URL,
        "callback_url": CALLBACK_URL,
        "total_games": total_games,
        "num_workers": num_workers,
        "mcts_simulations": mcts_sims,
        "temperature": 1.0,
        "temperature_threshold": 30,
        "c_puct": 1.0,
        "num_res_blocks": 8,
        "channels": 96,
        "input_channels": 50,
        "num_actions": 30
    }

    print(f"🚀 Triggering workflow: {total_games} games, {num_workers} workers, {mcts_sims} MCTS sims")

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if result.get('success'):
            execution_id = result.get('executionId')
            print(f"✅ Workflow triggered: {execution_id}")
            return execution_id
        else:
            print(f"❌ Workflow trigger failed: {result}")
            return None

    except Exception as e:
        print(f"❌ Error triggering workflow: {e}")
        return None

def wait_for_results(execution_id: str, timeout_minutes: int = 25):
    """Wait for workflow results to arrive."""

    results_dir = Path("data/distributed_results")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    print(f"⏳ Waiting for results (timeout: {timeout_minutes} min)...")

    while True:
        elapsed = time.time() - start_time

        # Check for result file
        result_files = list(results_dir.glob(f"*{execution_id}*.pkl"))
        result_files = [f for f in result_files if f.stat().st_size > 1000]

        if result_files:
            print(f"\n✅ Results received after {elapsed/60:.1f} minutes")
            return result_files[0], True

        # Check for error
        error_files = list(results_dir.glob(f"error_{execution_id}*.json"))
        if error_files:
            print(f"\n❌ Workflow failed after {elapsed/60:.1f} minutes")
            return error_files[0], False

        if elapsed > timeout_seconds:
            print(f"\n⏰ Timeout after {timeout_minutes} minutes")
            return None, False

        # Check every 15 seconds
        time.sleep(15)
        mins = int(elapsed / 60)
        print(f"   Waiting... {mins}/{timeout_minutes} min", end='\r')

def merge_into_checkpoint(result_file: Path):
    """Merge new examples into the latest checkpoint."""

    checkpoint_path = Path("data/checkpoints/model_latest.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\n📥 Loading examples from {result_file.name}...")
    with open(result_file, 'rb') as f:
        examples = pickle.load(f)
    num_new = len(examples)
    print(f"   Found {num_new} new examples")

    # Load checkpoint
    print(f"📂 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get current buffer
    if 'replay_buffer' in checkpoint:
        checkpoint_buffer_data = checkpoint['replay_buffer']
        old_size = len(checkpoint_buffer_data)
    else:
        checkpoint_buffer_data = []
        old_size = 0

    print(f"   Current buffer size: {old_size} examples")

    # Merge
    merged_buffer = ReplayBuffer(max_size=500000)
    merged_buffer.add_examples(checkpoint_buffer_data)
    merged_buffer.add_examples(examples)

    new_size = len(merged_buffer)
    added = new_size - old_size

    # Update checkpoint
    checkpoint['replay_buffer'] = merged_buffer.get_all_data()

    # Save
    print(f"💾 Saving updated checkpoint...")
    torch.save(checkpoint, checkpoint_path)

    print(f"✅ Merged: +{added} examples (total: {new_size})")

    # Mark as merged
    merged_marker = result_file.parent / ".merged_files.txt"
    with open(merged_marker, 'a') as f:
        f.write(f"{result_file.name}\n")

    return True

def main():
    parser = argparse.ArgumentParser(description="Continuous distributed augmentation")
    parser.add_argument('--games', type=int, default=10, help='Games per batch')
    parser.add_argument('--workers', type=int, default=5, help='Parallel workers')
    parser.add_argument('--mcts-sims', type=int, default=100, help='MCTS simulations')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches (None = infinite)')
    parser.add_argument('--timeout', type=int, default=25, help='Timeout per batch (minutes)')

    args = parser.parse_args()

    print("="*70)
    print("🔄 CONTINUOUS DISTRIBUTED AUGMENTATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Games per batch: {args.games}")
    print(f"  Workers: {args.workers}")
    print(f"  MCTS sims: {args.mcts_sims}")
    print(f"  Max batches: {args.max_batches or 'unlimited'}")
    print(f"  Timeout: {args.timeout} min")
    print("="*70)

    batch_count = 0
    total_examples_added = 0

    while args.max_batches is None or batch_count < args.max_batches:
        batch_count += 1
        print(f"\n{'='*70}")
        print(f"📦 BATCH {batch_count}")
        print(f"{'='*70}")

        # Trigger workflow
        execution_id = trigger_workflow(args.games, args.workers, args.mcts_sims)
        if not execution_id:
            print("❌ Failed to trigger workflow, waiting 60s before retry...")
            time.sleep(60)
            continue

        # Wait for results
        result_file, success = wait_for_results(execution_id, args.timeout)
        if not success or not result_file:
            print("❌ No results received, continuing to next batch...")
            time.sleep(30)
            continue

        # Merge into checkpoint
        if merge_into_checkpoint(result_file):
            with open(result_file, 'rb') as f:
                examples = pickle.load(f)
            total_examples_added += len(examples)
            print(f"\n📊 Total examples added so far: {total_examples_added}")

        print(f"\n✅ Batch {batch_count} complete")
        print(f"   Waiting 10s before next batch...")
        time.sleep(10)

    print(f"\n{'='*70}")
    print(f"✅ COMPLETED {batch_count} BATCHES")
    print(f"   Total examples added: {total_examples_added}")
    print(f"{'='*70}")

    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("✅ Checkpoints are safe - you can resume anytime")
        exit(0)
