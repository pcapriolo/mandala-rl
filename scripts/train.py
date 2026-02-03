#!/usr/bin/env python3
"""
Main training script for Mandala RL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest.pt
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Mandala RL bot')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--iterations', type=int, default=None,
                      help='Override number of iterations')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = config.get('device', 'mps')
    if device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    print(f"Using device: {device}")

    # Create game
    game = MandalaGame()
    print("Created Mandala game engine")

    # Create network
    network_config = config['network']
    network = MandalaNet(
        input_channels=network_config['input_channels'],
        num_actions=network_config['num_actions'],
        num_res_blocks=network_config['num_res_blocks'],
        channels=network_config['channels']
    )
    print(f"Created network with {sum(p.numel() for p in network.parameters())} parameters")

    # Prepare training config
    training_config = {
        # MCTS
        'mcts_simulations': config['mcts']['num_simulations'],
        'c_puct': config['mcts']['c_puct'],
        'temperature': config['selfplay']['temperature'],
        'temperature_threshold': config['selfplay']['temperature_threshold'],

        # Training
        'games_per_iteration': config['selfplay']['games_per_iteration'],
        'batch_size': config['training']['batch_size'],
        'epochs_per_iteration': config['training']['epochs_per_iteration'],
        'learning_rate': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'lr_milestones': config['training']['lr_milestones'],
        'lr_gamma': config['training']['lr_gamma'],
        'replay_buffer_size': config['training']['replay_buffer_size'],
        'checkpoint_frequency': config['training']['checkpoint_frequency'],

        # Evaluation
        'eval_frequency': config['evaluation']['eval_frequency'],
        'eval_num_games': config['evaluation']['eval_num_games'],
        'eval_mcts_simulations': config['evaluation']['eval_mcts_simulations'],

        # Network architecture (for evaluation)
        'num_res_blocks': network_config['num_res_blocks'],
        'channels': network_config['channels'],

        # Paths
        'checkpoint_dir': config['paths']['checkpoint_dir'],
        'log_dir': config['paths']['log_dir'],
        'elo_file': config['paths']['elo_file'],
    }

    # Create trainer
    trainer = Trainer(
        game=game,
        network=network,
        config=training_config,
        device=device
    )
    print("Created trainer")

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")

    # Train
    num_iterations = args.iterations or config['training']['num_iterations']
    print(f"\nStarting training for {num_iterations} iterations...")
    print("=" * 60)

    trainer.train(num_iterations)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
