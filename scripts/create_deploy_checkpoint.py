#!/usr/bin/env python3
"""
Create lightweight deploy checkpoints for Railway/production deployment.

Strips optimizer state, scheduler state, and replay buffer from full
training checkpoints. Output is ~3MB per model instead of ~50MB+.

Usage:
    # Auto-detect latest checkpoints for both games
    python scripts/create_deploy_checkpoint.py

    # Specific checkpoints
    python scripts/create_deploy_checkpoint.py \
        --mandala data/checkpoints/model_iter_20.pt \
        --lc data/lost_cities/checkpoints/model_iter_11.pt
"""

import argparse
import sys
import torch
from pathlib import Path


def find_latest(checkpoint_dir):
    """Find most recent .pt file in directory."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    # Prefer model_iter_*.pt (full iteration checkpoints) over game-level ones
    iter_cps = sorted(cp_dir.glob("model_iter_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if iter_cps:
        return iter_cps[0]
    all_cps = sorted(cp_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return all_cps[0] if all_cps else None


def create_lightweight(source_path, output_path):
    """Strip a checkpoint to model weights + metadata only."""
    print(f"  Loading: {source_path}")
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)

    lightweight = {
        'model_state_dict': checkpoint['model_state_dict'],
        'iteration': checkpoint.get('iteration', 'unknown'),
        'total_games': checkpoint.get('total_games', 'unknown'),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lightweight, output_path)

    src_mb = source_path.stat().st_size / (1024 * 1024)
    dst_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({src_mb:.1f}MB -> {dst_mb:.1f}MB)")
    print(f"  Iteration: {lightweight['iteration']}, Games: {lightweight['total_games']}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create lightweight deploy checkpoints")
    parser.add_argument('--mandala', type=str, default=None, help='Mandala checkpoint path')
    parser.add_argument('--lc', type=str, default=None, help='Lost Cities checkpoint path')
    parser.add_argument('--output-dir', type=str, default='data/deploy', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    created = 0

    # Mandala
    mandala_src = Path(args.mandala) if args.mandala else find_latest("data/checkpoints")
    if mandala_src and mandala_src.exists():
        print("\nMandala:")
        if create_lightweight(mandala_src, output_dir / "mandala" / "model.pt"):
            created += 1
    else:
        print("\nMandala: no checkpoint found, skipping")

    # Lost Cities
    lc_src = Path(args.lc) if args.lc else find_latest("data/lost_cities/checkpoints")
    if lc_src and lc_src.exists():
        print("\nLost Cities:")
        if create_lightweight(lc_src, output_dir / "lost_cities" / "model.pt"):
            created += 1
    else:
        print("\nLost Cities: no checkpoint found, skipping")

    if created > 0:
        print(f"\nCreated {created} deploy checkpoint(s) in {output_dir}/")
        print("Add to git: git add data/deploy/ && git commit -m 'Update deploy checkpoints'")
    else:
        print("\nNo checkpoints created. Make sure training checkpoints exist.")
        sys.exit(1)


if __name__ == '__main__':
    main()
