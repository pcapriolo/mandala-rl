#!/usr/bin/env python3
"""
Run a single Elo evaluation between two checkpoints.

Usage:
    python scripts/single_elo_eval.py --model1 7 --model2 6
    python scripts/single_elo_eval.py --model1 8 --model2 0  # Big gap
"""
import argparse
import torch
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.evaluation.arena import Arena
from mandala_rl.evaluation.elo import EloRating


def load_model_from_checkpoint(checkpoint_path, device='mps'):
    """Load a model from checkpoint."""
    print(f"Loading {checkpoint_path.name}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get architecture params (should match training config)
    model = MandalaNet(
        input_channels=50,
        num_actions=30,
        num_res_blocks=8,
        channels=96
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def run_evaluation(model1_iter, model2_iter, num_games=20, mcts_sims=400, device='mps'):
    """Run evaluation between two models."""

    checkpoint_dir = Path("data/checkpoints")
    elo_file = Path("data/elo_ratings.json")

    # Check checkpoints exist
    ckpt1 = checkpoint_dir / f"model_iter_{model1_iter}.pt"
    ckpt2 = checkpoint_dir / f"model_iter_{model2_iter}.pt"

    if not ckpt1.exists():
        print(f"❌ Checkpoint not found: {ckpt1}")
        return False
    if not ckpt2.exists():
        print(f"❌ Checkpoint not found: {ckpt2}")
        return False

    print("=" * 70)
    print(f"🏆 ELO EVALUATION: Iteration {model1_iter} vs Iteration {model2_iter}")
    print("=" * 70)

    # Load Elo system
    elo = EloRating(initial_rating=1500.0, k_factor=32.0)
    if elo_file.exists():
        elo.load(elo_file)
        print(f"\n📊 Current Elo ratings:")
        print(f"   Iteration {model1_iter}: {elo.get_rating(f'iter_{model1_iter}'):.1f}")
        print(f"   Iteration {model2_iter}: {elo.get_rating(f'iter_{model2_iter}'):.1f}")
    else:
        print(f"\n📊 No existing Elo data - starting fresh")

    # Load models
    print(f"\n⏳ Loading models...")
    model1 = load_model_from_checkpoint(ckpt1, device)
    model2 = load_model_from_checkpoint(ckpt2, device)
    print(f"✅ Models loaded")

    # Setup arena
    game = MandalaGame()
    arena = Arena(game, mcts_simulations=mcts_sims, c_puct=1.0, device=device)

    # Run matches
    print(f"\n🎮 Playing {num_games} games ({mcts_sims} MCTS simulations per move)...")
    print(f"   This will take approximately {num_games * 2.1:.0f} minutes")
    print()

    start_time = time.time()

    results = arena.play_match(
        model1=model1,
        model2=model2,
        num_games=num_games
    )

    wins = results['model1_wins']
    losses = results['model2_wins']
    draws = results['draws']

    elapsed = time.time() - start_time

    # Results
    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Model 1 (iter {model1_iter}): {wins} wins, {draws} draws, {losses} losses")
    print(f"Model 2 (iter {model2_iter}): {losses} wins, {draws} draws, {wins} losses")
    print(f"Win rate: {win_rate*100:.1f}% for model 1")
    print(f"Time taken: {elapsed/60:.1f} minutes ({elapsed/num_games:.1f} sec/game)")

    # Update Elo
    print("\n" + "=" * 70)
    print("🏆 ELO UPDATE")
    print("=" * 70)

    id1 = f'iter_{model1_iter}'
    id2 = f'iter_{model2_iter}'
    old_rating1 = elo.get_rating(id1)
    old_rating2 = elo.get_rating(id2)

    # Update Elo per-game (not aggregate) for correct rating changes
    for _ in range(wins):
        elo.record_match(id1, id2, id1)
    for _ in range(losses):
        elo.record_match(id1, id2, id2)
    for _ in range(draws):
        elo.record_match(id1, id2, None)

    new_rating1 = elo.get_rating(id1)
    new_rating2 = elo.get_rating(id2)

    change1 = new_rating1 - old_rating1
    change2 = new_rating2 - old_rating2

    print(f"Iteration {model1_iter}:")
    print(f"   Old: {old_rating1:.1f}")
    print(f"   New: {new_rating1:.1f}")
    print(f"   Change: {change1:+.1f}")
    print()
    print(f"Iteration {model2_iter}:")
    print(f"   Old: {old_rating2:.1f}")
    print(f"   New: {new_rating2:.1f}")
    print(f"   Change: {change2:+.1f}")

    # Save
    print(f"\n💾 Saving Elo ratings to {elo_file}")
    elo.save(elo_file)

    # Show leaderboard
    print("\n" + "=" * 70)
    print("📈 CURRENT LEADERBOARD")
    print("=" * 70)
    leaderboard = elo.get_leaderboard()
    print(f"{'Rank':<6} {'Model':<15} {'Elo':<10}")
    print("-" * 31)
    for rank, (model_id, rating) in enumerate(leaderboard[:10], 1):
        print(f"{rank:<6} {model_id:<15} {rating:>7.1f}")

    print("\n✅ Evaluation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run single Elo evaluation")
    parser.add_argument('--model1', type=int, default=7, help='First model iteration')
    parser.add_argument('--model2', type=int, default=6, help='Second model iteration')
    parser.add_argument('--num-games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='mps', help='Device (mps/cuda/cpu)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU")
        args.device = 'cpu'

    success = run_evaluation(
        model1_iter=args.model1,
        model2_iter=args.model2,
        num_games=args.num_games,
        mcts_sims=args.mcts_sims,
        device=args.device
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
