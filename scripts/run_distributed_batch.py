"""
Run distributed self-play in small batches to augment local training.

Usage:
    python scripts/run_distributed_batch.py --games 10 --workers 5
"""
import argparse
import requests
import time
import json
from pathlib import Path
from datetime import datetime

# Configuration
WEBHOOK_URL = "https://f75ecf45-b024-4693-bc22-dd0b2f1056be-00-20vc6rnxkl2w7.kirk.replit.dev/api/webhooks/launch/cb3dc298-cb9e-4eb2-994e-c402b935484a"
CALLBACK_URL = "https://four-cobras-appear.loca.lt/callback/results"
CHECKPOINT_URL = "https://mandala-checkpoints.loca.lt/model_latest_lightweight.pt"

def trigger_distributed_run(total_games: int, num_workers: int, mcts_sims: int = 100):
    """Trigger a distributed self-play run."""
    
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
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if result.get('success'):
            execution_id = result.get('executionId')
            print(f"✅ Workflow triggered successfully")
            print(f"   Execution ID: {execution_id}")
            return execution_id
        else:
            print(f"❌ Workflow trigger failed: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Error triggering workflow: {e}")
        return None

def wait_for_results(execution_id: str, timeout_minutes: int = 10):
    """Wait for results to arrive from the callback."""
    
    results_dir = Path("data/distributed_results")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    print(f"⏳ Waiting for results (timeout: {timeout_minutes} min)...")
    
    while True:
        elapsed = time.time() - start_time
        
        # Check for result files with this execution ID
        result_files = list(results_dir.glob(f"*{execution_id}*.pkl"))
        result_files = [f for f in result_files if f.stat().st_size > 1000]
        
        if result_files:
            print(f"✅ Results received after {elapsed/60:.1f} minutes")
            return result_files[0]
        
        if elapsed > timeout_seconds:
            print(f"⏰ Timeout after {timeout_minutes} minutes")
            return None
        
        # Check every 15 seconds
        time.sleep(15)
        dots = int((elapsed % 60) / 15)
        print(f"   Waiting{'.' * (dots + 1)}", end='\r')

def merge_results(result_file: Path):
    """Merge distributed results into replay buffer."""
    import pickle
    from mandala_rl.training.replay_buffer import ReplayBuffer
    
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
    parser = argparse.ArgumentParser(description="Run distributed self-play batch")
    parser.add_argument('--games', type=int, default=10, help='Number of games to generate')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--mcts-sims', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--no-merge', action='store_true', help='Skip automatic merging')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎮 DISTRIBUTED SELF-PLAY BATCH")
    print("="*70)
    
    # Trigger workflow
    execution_id = trigger_distributed_run(args.games, args.workers, args.mcts_sims)
    if not execution_id:
        return 1
    
    # Wait for results
    result_file = wait_for_results(execution_id, timeout_minutes=10)
    if not result_file:
        print("❌ No results received")
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
