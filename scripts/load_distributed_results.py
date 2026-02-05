"""
Simple script to load distributed results and add to training.

Use this if you want to manually trigger distributed self-play
and then load the results into your training.
"""
import pickle
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.training.trainer import Trainer
import yaml


def main():
    parser = argparse.ArgumentParser(description='Load distributed self-play results')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Config file')
    parser.add_argument('--results-file', type=str, help='Specific results file to load')
    parser.add_argument('--results-dir', type=str, default='data/distributed_results',
                       help='Directory with results')
    parser.add_argument('--train', action='store_true', help='Train network after loading')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize trainer
    device = config.get('device', 'mps')
    game = MandalaGame()
    network = MandalaNet(
        input_channels=config['network']['input_channels'],
        num_actions=config['network']['num_actions'],
        num_res_blocks=config['network']['num_res_blocks'],
        channels=config['network']['channels']
    ).to(device)

    trainer = Trainer(
        game=game,
        network=network,
        config=config['training'],
        paths=config['paths'],
        selfplay_config=config['selfplay'],
        mcts_config=config['mcts'],
        eval_config=config.get('evaluation', {}),
        device=device
    )

    # Load checkpoint if exists
    latest_checkpoint = Path(config['paths']['checkpoint_dir']) / 'model_latest.pt'
    if latest_checkpoint.exists():
        print(f"Loading checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    else:
        print("No checkpoint found, starting fresh")

    print(f"Current replay buffer size: {len(trainer.replay_buffer)}")

    # Find results file
    if args.results_file:
        results_file = Path(args.results_file)
    else:
        # Find most recent results
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("examples_*.pkl"), key=lambda p: p.stat().st_mtime)

        if not result_files:
            print(f"No results found in {results_dir}")
            print("Trigger distributed self-play first!")
            return

        results_file = result_files[-1]
        print(f"Using most recent results: {results_file}")

    # Load examples
    print(f"\nLoading examples from {results_file}...")
    with open(results_file, 'rb') as f:
        examples = pickle.load(f)

    print(f"Loaded {len(examples)} training examples")

    # Add to replay buffer
    trainer.replay_buffer.add_examples(examples)
    print(f"New replay buffer size: {len(trainer.replay_buffer)}")

    # Optionally train
    if args.train:
        print("\nTraining network on new examples...")
        trainer._train_network()

        print("\nSaving checkpoint...")
        trainer._save_checkpoint()
        print("✓ Done!")
    else:
        print("\nExamples loaded but not trained yet.")
        print("Run with --train to train on these examples.")


if __name__ == '__main__':
    main()
