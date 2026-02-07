"""
Merge distributed self-play results into the local replay buffer.

Usage:
    python scripts/merge_distributed_examples.py
"""
import pickle
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.training.replay_buffer import ReplayBuffer

def merge_distributed_examples():
    """Merge distributed examples into the replay buffer."""
    
    # Paths
    distributed_dir = Path("data/distributed_results")
    buffer_path = Path("data/replay_buffer.pkl")
    
    # Find all distributed example files (not already merged)
    merged_marker = distributed_dir / ".merged_files.txt"
    already_merged = set()
    if merged_marker.exists():
        already_merged = set(merged_marker.read_text().strip().split('\n'))
    
    example_files = [
        f for f in distributed_dir.glob("examples_*.pkl")
        if f.name not in already_merged and f.stat().st_size > 1000  # Skip empty files
    ]
    
    if not example_files:
        print("No new distributed examples to merge.")
        return
    
    print(f"Found {len(example_files)} new distributed result files")
    
    # Load or create replay buffer
    if buffer_path.exists():
        print(f"Loading existing replay buffer from {buffer_path}")
        with open(buffer_path, 'rb') as f:
            buffer = pickle.load(f)
        print(f"  Current buffer size: {len(buffer.buffer)}")
    else:
        print("Creating new replay buffer")
        buffer = ReplayBuffer(max_size=500000)
    
    # Merge distributed examples
    total_added = 0
    for example_file in example_files:
        print(f"\nMerging {example_file.name}...")
        
        with open(example_file, 'rb') as f:
            examples = pickle.load(f)
        
        print(f"  Found {len(examples)} examples")

        # Add to buffer
        buffer.add_examples(examples)
        total_added += len(examples)
        
        # Mark as merged
        already_merged.add(example_file.name)
        print(f"  Added {len(examples)} examples to buffer")
    
    # Save updated buffer
    print(f"\nSaving replay buffer...")
    print(f"  New buffer size: {len(buffer.buffer)}")
    with open(buffer_path, 'wb') as f:
        pickle.dump(buffer, f)
    
    # Update merged marker
    with open(merged_marker, 'w') as f:
        f.write('\n'.join(sorted(already_merged)))
    
    print(f"\n✅ Successfully merged {total_added} examples from {len(example_files)} files")
    print(f"📊 Total replay buffer size: {len(buffer.buffer)}")

if __name__ == "__main__":
    merge_distributed_examples()
