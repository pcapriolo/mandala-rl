#!/usr/bin/env python3
"""
Evaluation script for Mandala RL.

Evaluates a checkpoint against baseline and updates Elo ratings.

Usage:
    python scripts/evaluate.py --checkpoint data/checkpoints/model_iter_10.pt
    python scripts/evaluate.py --checkpoint data/checkpoints/model_latest.pt --baseline data/checkpoints/model_iter_5.pt
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.evaluation.arena import Arena
from mandala_rl.evaluation.elo import EloRating


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: Path, device: str) -> MandalaNet:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create network (using default config for now)
    network = MandalaNet(
        input_channels=50,
        num_actions=256,
        num_res_blocks=10,
        channels=128
    )

    network.load_state_dict(checkpoint['model_state_dict'])
    network.to(device)
    network.eval()

    return network


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mandala RL bot')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint to evaluate')
    parser.add_argument('--baseline', type=str, default=None,
                      help='Path to baseline checkpoint (default: previous iteration)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--num-games', type=int, default=None,
                      help='Number of games to play')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device (auto-detect: cuda > mps > cpu)
    device = config.get('device', 'auto')
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create game
    game = MandalaGame()

    # Load models
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {args.checkpoint} not found")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Load baseline
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"Error: Baseline {args.baseline} not found")
            return
        print(f"Loading baseline: {baseline_path}")
        baseline = load_model(baseline_path, device)
    else:
        # Use same model as baseline (sanity check)
        print("No baseline specified, using same model (sanity check)")
        baseline = model

    # Create arena
    arena = Arena(
        game=game,
        mcts_simulations=config['mcts']['num_simulations'],
        c_puct=config['mcts']['c_puct'],
        device=device
    )

    # Play match
    num_games = args.num_games or config['evaluation']['num_games']
    print(f"\nPlaying {num_games} games...")
    print("=" * 60)

    results = arena.play_match(model, baseline, num_games, seed=args.seed)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model wins:    {results['model1_wins']} ({results['model1_wins']/num_games*100:.1f}%)")
    print(f"Baseline wins: {results['model2_wins']} ({results['model2_wins']/num_games*100:.1f}%)")
    print(f"Draws:         {results['draws']} ({results['draws']/num_games*100:.1f}%)")

    # Update Elo ratings
    elo_file = Path(config['paths']['elo_file'])
    elo = EloRating()

    if elo_file.exists():
        elo.load(elo_file)
        print(f"\nLoaded Elo ratings from {elo_file}")

    # Get model IDs from checkpoint names
    model_id = checkpoint_path.stem
    baseline_id = Path(args.baseline).stem if args.baseline else "same"

    if baseline_id != "same":
        # Record match
        if results['model1_wins'] > results['model2_wins']:
            winner = model_id
        elif results['model2_wins'] > results['model1_wins']:
            winner = baseline_id
        else:
            winner = None

        elo.record_match(model_id, baseline_id, winner)

        # Save updated ratings
        elo_file.parent.mkdir(parents=True, exist_ok=True)
        elo.save(elo_file)
        print(f"\nUpdated Elo ratings saved to {elo_file}")

        # Display leaderboard
        print(f"\n{elo}")


if __name__ == '__main__':
    main()
