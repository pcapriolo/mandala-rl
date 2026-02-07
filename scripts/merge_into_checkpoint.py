#!/usr/bin/env python3
"""
Merge distributed examples from replay_buffer.pkl into the latest checkpoint.

This allows the running trainer to pick up distributed examples in the next iteration.
"""
import torch
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.training.replay_buffer import ReplayBuffer

def merge_into_checkpoint():
    checkpoint_path = Path("data/checkpoints/model_latest.pt")
    replay_buffer_path = Path("data/replay_buffer.pkl")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    if not replay_buffer_path.exists():
        print(f"❌ Replay buffer not found: {replay_buffer_path}")
        return 1

    print("="*70)
    print("🔀 MERGING DISTRIBUTED EXAMPLES INTO CHECKPOINT")
    print("="*70)

    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get current replay buffer from checkpoint
    if 'replay_buffer' in checkpoint:
        checkpoint_buffer_data = checkpoint['replay_buffer']
        print(f"   Current buffer size: {len(checkpoint_buffer_data)} examples")
    else:
        checkpoint_buffer_data = []
        print(f"   No existing buffer in checkpoint")

    # Load distributed examples
    print(f"\n📥 Loading distributed examples: {replay_buffer_path}")
    with open(replay_buffer_path, 'rb') as f:
        distributed_examples = pickle.load(f)
    print(f"   Distributed examples: {len(distributed_examples.buffer)} examples")

    # Merge buffers
    print(f"\n🔀 Merging buffers...")
    merged_buffer = ReplayBuffer(max_size=500000)

    # Add checkpoint examples first (preserve existing training data)
    merged_buffer.add_examples(checkpoint_buffer_data)
    print(f"   Added {len(checkpoint_buffer_data)} checkpoint examples")

    # Add distributed examples
    distributed_data = list(distributed_examples.buffer)
    merged_buffer.add_examples(distributed_data)
    print(f"   Added {len(distributed_data)} distributed examples")

    final_size = len(merged_buffer)
    print(f"   Final buffer size: {final_size} examples")

    # Update checkpoint with merged buffer
    checkpoint['replay_buffer'] = merged_buffer.get_all_data()

    # Backup original checkpoint
    backup_path = checkpoint_path.parent / f"{checkpoint_path.stem}_backup.pt"
    print(f"\n💾 Creating backup: {backup_path}")
    torch.save(torch.load(checkpoint_path, weights_only=False), backup_path)

    # Save updated checkpoint
    print(f"💾 Saving updated checkpoint: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)

    print(f"\n✅ Successfully merged!")
    print(f"   Total examples now in checkpoint: {final_size}")
    print(f"   Backup saved to: {backup_path}")

    # Clean up the distributed buffer file (optional - keep it for records)
    # replay_buffer_path.unlink()
    # print(f"   Cleaned up: {replay_buffer_path}")

    print("\n" + "="*70)
    print("✅ MERGE COMPLETE - Next iteration will use merged buffer")
    print("="*70)

    return 0

if __name__ == "__main__":
    exit(merge_into_checkpoint())
