"""
Hybrid training: Distributed self-play + Local GPU training.

This script combines the best of both worlds:
- Fast distributed self-play on cloud workers (offload the slow part)
- Fast GPU training on your MacBook (keep the fast part local)
"""
import argparse
import pickle
import requests
import time
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.training.trainer import Trainer


def create_lightweight_checkpoint(full_checkpoint_path: Path, output_path: Path):
    """Create lightweight checkpoint (model only, no replay buffer)."""
    import torch

    checkpoint = torch.load(full_checkpoint_path, map_location='cpu', weights_only=False)

    lightweight = {
        'iteration': checkpoint['iteration'],
        'model_state_dict': checkpoint['model_state_dict'],
    }

    torch.save(lightweight, output_path)
    print(f"Created lightweight checkpoint: {output_path}")


def wait_for_results(execution_id: str, results_dir: Path, timeout: int = 3600):
    """Wait for distributed workflow results to arrive."""
    print(f"\nWaiting for results from execution {execution_id}...")
    print(f"Looking in: {results_dir}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Look for pickle file with this execution ID
        result_files = list(results_dir.glob(f"examples_{execution_id}_*.pkl"))

        if result_files:
            result_file = result_files[0]
            print(f"\n✓ Results received: {result_file}")
            return result_file

        # Check for error
        error_files = list(results_dir.glob(f"error_{execution_id}_*.json"))
        if error_files:
            print(f"\n✗ Workflow failed. Error file: {error_files[0]}")
            with open(error_files[0], 'r') as f:
                import json
                error_data = json.load(f)
                print(f"Error: {error_data.get('error', 'Unknown')}")
            return None

        # Wait and retry
        time.sleep(5)
        elapsed = time.time() - start_time
        print(f"  Waiting... ({elapsed:.0f}s elapsed)", end='\r')

    print(f"\n✗ Timeout after {timeout}s")
    return None


def trigger_distributed_selfplay(
    webhook_url: str,
    callback_url: str,
    checkpoint_url: str,
    total_games: int,
    num_workers: int,
    mcts_simulations: int,
    num_res_blocks: int,
    channels: int
) -> str:
    """Trigger distributed self-play workflow via webhook."""

    payload = {
        "callback_url": callback_url,
        "checkpoint_url": checkpoint_url,
        "total_games": total_games,
        "num_workers": num_workers,
        "mcts_simulations": mcts_simulations,
        "num_res_blocks": num_res_blocks,
        "channels": channels
    }

    print("\nTriggering distributed self-play workflow...")
    print(f"  Webhook: {webhook_url}")
    print(f"  Callback: {callback_url}")
    print(f"  Checkpoint: {checkpoint_url}")
    print(f"  Games: {total_games} (across {num_workers} workers)")
    print(f"  MCTS sims: {mcts_simulations}")

    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()

    result = response.json()
    execution_id = result.get('executionId')

    print(f"\n✓ Workflow triggered!")
    print(f"  Execution ID: {execution_id}")

    return execution_id


def main():
    parser = argparse.ArgumentParser(description='Hybrid distributed training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--webhook-url', type=str, required=True, help='Flyte webhook URL')
    parser.add_argument('--callback-url', type=str, required=True, help='Callback URL for results')
    parser.add_argument('--checkpoint-url', type=str, required=True, help='Public URL to checkpoint')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    parser.add_argument('--games-per-iteration', type=int, default=100, help='Games per iteration')
    parser.add_argument('--num-workers', type=int, default=10, help='Number of distributed workers')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")

    # Initialize components
    device = config.get('device', 'mps')
    print(f"Using device: {device}")

    game = MandalaGame()
    network = MandalaNet(
        input_channels=config['network']['input_channels'],
        num_actions=config['network']['num_actions'],
        num_res_blocks=config['network']['num_res_blocks'],
        channels=config['network']['channels']
    ).to(device)

    print(f"Created network with {sum(p.numel() for p in network.parameters())} parameters")

    # Create trainer
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
    print("Created trainer")

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint {args.resume} not found")
    else:
        latest_checkpoint = Path(config['paths']['checkpoint_dir']) / 'model_latest.pt'
        if latest_checkpoint.exists():
            print(f"Resuming from latest checkpoint...")
            trainer.load_checkpoint(latest_checkpoint)

    results_dir = Path("data/distributed_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting HYBRID training for {args.iterations} iterations")
    print("=" * 60)

    for iteration in range(args.iterations):
        trainer.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {trainer.iteration}")
        print(f"{'='*60}")

        # 1. Create lightweight checkpoint for workers
        print("\n[1/4] Preparing checkpoint for workers...")
        lightweight_path = Path(config['paths']['checkpoint_dir']) / 'model_for_workers.pt'
        full_checkpoint = Path(config['paths']['checkpoint_dir']) / 'model_latest.pt'

        if full_checkpoint.exists():
            create_lightweight_checkpoint(full_checkpoint, lightweight_path)
        else:
            print("No checkpoint found, using initial weights")
            trainer.save_checkpoint(lightweight_path)

        # 2. Trigger distributed self-play
        print("\n[2/4] Generating self-play games (distributed)...")
        execution_id = trigger_distributed_selfplay(
            webhook_url=args.webhook_url,
            callback_url=args.callback_url,
            checkpoint_url=args.checkpoint_url,
            total_games=args.games_per_iteration,
            num_workers=args.num_workers,
            mcts_simulations=config['mcts']['num_simulations'],
            num_res_blocks=config['network']['num_res_blocks'],
            channels=config['network']['channels']
        )

        # 3. Wait for results
        result_file = wait_for_results(execution_id, results_dir, timeout=3600)

        if result_file is None:
            print("\n✗ Failed to get results. Skipping this iteration.")
            continue

        # 4. Load examples and add to replay buffer
        print("\n[3/4] Loading training examples...")
        with open(result_file, 'rb') as f:
            examples = pickle.load(f)

        print(f"Loaded {len(examples)} training examples")
        trainer.replay_buffer.add_examples(examples)
        print(f"Replay buffer size: {len(trainer.replay_buffer)}")

        # 5. Train network (locally on GPU)
        print("\n[4/4] Training network (local GPU)...")
        trainer._train_network()

        # 6. Save checkpoint
        trainer._save_checkpoint()
        trainer.total_games += args.games_per_iteration

        print(f"\n✓ Iteration {trainer.iteration} complete!")
        print(f"Total games: {trainer.total_games}")

    print("\n" + "="*60)
    print("HYBRID TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
