"""
Run distributed self-play with async webhook trigger.

Fires the webhook without waiting for execution ID, then polls for new results.
"""
import argparse
import requests
import time
import pickle
from pathlib import Path
from datetime import datetime
from mandala_rl.training.replay_buffer import ReplayBuffer

# Configuration
WEBHOOK_URL = "https://f75ecf45-b024-4693-bc22-dd0b2f1056be-00-20vc6rnxkl2w7.kirk.replit.dev/api/webhooks/launch/cb3dc298-cb9e-4eb2-994e-c402b935484a"
CALLBACK_URL = "https://four-cobras-appear.loca.lt/callback/results"
CHECKPOINT_URL = "https://mandala-checkpoints.loca.lt/model_latest_lightweight.pt"

def trigger_workflow_async(total_games: int, num_workers: int, mcts_sims: int = 100):
    """Trigger workflow without waiting for response."""

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

    print(f"🚀 Triggering distributed run: {total_games} games, {num_workers} workers")
    print(f"   MCTS simulations: {mcts_sims}")
    print(f"   (Not waiting for execution ID - webhook may timeout)")

    try:
        # Try with short timeout, but don't fail if it times out
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result.get('success'):
            execution_id = result.get('executionId')
            print(f"✅ Got execution ID: {execution_id}")
            return execution_id
    except requests.exceptions.Timeout:
        print(f"⚠️  Webhook timed out (expected) - workflow likely started")
        return None
    except Exception as e:
        print(f"⚠️  Webhook error: {e}")
        print(f"   Assuming workflow started anyway...")
        return None

def wait_for_new_results(timeout_minutes: int = 25, check_interval: int = 15):
    """Wait for any new result files to appear."""

    results_dir = Path("data/distributed_results")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    # Get current files to detect new ones
    existing_files = set(results_dir.glob("examples_*.pkl"))
    existing_files = {f for f in existing_files if f.stat().st_size > 1000}

    print(f"⏳ Waiting for new results (timeout: {timeout_minutes} min)...")
    print(f"   Current examples files: {len(existing_files)}")

    while True:
        elapsed = time.time() - start_time

        # Check for new result files
        current_files = set(results_dir.glob("examples_*.pkl"))
        current_files = {f for f in current_files if f.stat().st_size > 1000}

        new_files = current_files - existing_files

        if new_files:
            new_file = list(new_files)[0]
            print(f"\n✅ New results received after {elapsed/60:.1f} minutes")
            print(f"   File: {new_file.name}")
            return new_file

        if elapsed > timeout_seconds:
            print(f"\n⏰ Timeout after {timeout_minutes} minutes")
            return None

        # Check every interval
        time.sleep(check_interval)
        mins_elapsed = int(elapsed / 60)
        mins_remaining = timeout_minutes - mins_elapsed
        print(f"   Waiting... ({mins_elapsed}/{timeout_minutes} min, ~{mins_remaining} min remaining)", end='\r')

def merge_results(result_file: Path):
    """Merge distributed results into replay buffer."""

    buffer_path = Path("data/replay_buffer.pkl")

    # Load examples
    print(f"\n📥 Loading examples from {result_file.name}...")
    with open(result_file, 'rb') as f:
        examples = pickle.load(f)
    print(f"   Found {len(examples)} examples")

    # Load or create buffer
    if buffer_path.exists():
        print(f"📂 Loading existing replay buffer...")
        with open(buffer_path, 'rb') as f:
            buffer = pickle.load(f)
        old_size = len(buffer.buffer)
        print(f"   Current size: {old_size}")
    else:
        print(f"📝 Creating new replay buffer...")
        buffer = ReplayBuffer(max_size=500000)
        old_size = 0

    # Add examples
    print(f"➕ Adding examples to buffer...")
    for state, policy, value in examples:
        buffer.add(state, policy, value)

    new_size = len(buffer.buffer)
    added = new_size - old_size

    # Save buffer
    print(f"💾 Saving replay buffer...")
    with open(buffer_path, 'wb') as f:
        pickle.dump(buffer, f)

    print(f"✅ Merged successfully:")
    print(f"   Added: {added} examples")
    print(f"   Total buffer size: {new_size}")

    # Mark as merged
    merged_marker = result_file.parent / ".merged_files.txt"
    with open(merged_marker, 'a') as f:
        f.write(f"{result_file.name}\n")

def main():
    parser = argparse.ArgumentParser(description="Run distributed self-play batch (async)")
    parser.add_argument('--games', type=int, default=10, help='Number of games to generate')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--mcts-sims', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--no-merge', action='store_true', help='Skip automatic merging')
    parser.add_argument('--timeout', type=int, default=25, help='Minutes to wait for results')

    args = parser.parse_args()

    print("="*70)
    print("🎮 DISTRIBUTED SELF-PLAY BATCH (ASYNC)")
    print("="*70)

    # Trigger workflow (may timeout, but that's ok)
    execution_id = trigger_workflow_async(args.games, args.workers, args.mcts_sims)

    # Wait for new results
    result_file = wait_for_new_results(timeout_minutes=args.timeout)
    if not result_file:
        print("❌ No new results received")
        return 1

    # Merge results
    if not args.no_merge:
        merge_results(result_file)
    else:
        print(f"⏭️  Skipping merge (--no-merge specified)")

    print("\n" + "="*70)
    print("✅ BATCH COMPLETE")
    print("="*70)
    return 0

if __name__ == "__main__":
    exit(main())
