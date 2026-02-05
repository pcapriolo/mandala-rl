"""
Training script using distributed self-play via Flyte.

This is a modified version of train.py that offloads self-play to cloud workers.
"""
import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flytekit.configuration import Config, PlatformConfig
from workflows.client import DistributedSelfPlayClient
from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train Mandala RL agent with distributed self-play')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--iterations', type=int, help='Number of iterations (overrides config)')
    parser.add_argument('--flyte-endpoint', type=str, required=True, help='Flyte server endpoint')
    parser.add_argument('--num-workers', type=int, default=10, help='Number of distributed workers')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")

    # Override iterations if specified
    if args.iterations:
        config['training']['num_iterations'] = args.iterations

    # Add distributed config
    config['training']['num_workers'] = args.num_workers

    # Setup Flyte
    flyte_config = Config(
        platform=PlatformConfig(
            endpoint=args.flyte_endpoint,
            insecure=False
        )
    )

    # Initialize components
    device = config.get('device', 'mps')
    print(f"Using device: {device}")

    game = MandalaGame()
    print("Created Mandala game engine")

    network = MandalaNet(
        input_channels=config['network']['input_channels'],
        num_actions=config['network']['num_actions'],
        num_res_blocks=config['network']['num_res_blocks'],
        channels=config['network']['channels']
    ).to(device)
    print(f"Created network with {sum(p.numel() for p in network.parameters())} parameters")

    # Create trainer (regular trainer)
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

    # Create distributed client
    distributed_client = DistributedSelfPlayClient(flyte_config)
    print(f"Created distributed client (endpoint: {args.flyte_endpoint})")

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    else:
        # Auto-resume from latest checkpoint if it exists
        latest_checkpoint = Path(config['paths']['checkpoint_dir']) / 'model_latest.pt'
        if latest_checkpoint.exists():
            print(f"Found existing checkpoint, resuming training...")
            trainer.load_checkpoint(latest_checkpoint)
        else:
            print("No checkpoint found, starting from scratch")

    # Override trainer's self-play method with distributed version
    original_generate = trainer._generate_selfplay_games

    def distributed_generate_selfplay():
        """Generate self-play games using distributed workers."""
        num_games = trainer.config.get('games_per_iteration', 100)
        num_workers = trainer.config.get('num_workers', 10)
        mcts_sims = trainer.config.get('mcts_simulations', 800)

        print(f"\n[DISTRIBUTED] Generating {num_games} games using {num_workers} workers...")

        # Save current checkpoint for workers to use
        checkpoint_path = Path(trainer.checkpoint_dir) / "model_for_selfplay.pt"
        trainer.save_checkpoint(checkpoint_path)

        # Generate games via Flyte
        examples = distributed_client.generate_games_distributed(
            checkpoint_path=checkpoint_path,
            num_games=num_games,
            num_workers=num_workers,
            mcts_simulations=mcts_sims
        )

        print(f"Received {len(examples)} training examples from distributed workers")

        # Convert examples back to game format for compatibility
        # (This is a bit inefficient but maintains compatibility with existing code)
        from mandala_rl.selfplay.worker import SelfPlayGame

        # Group examples by game (this is approximate since we lost game boundaries)
        # For now, just return examples directly and modify _add_to_replay_buffer
        return examples

    # Monkey-patch the method
    trainer._generate_selfplay_games = distributed_generate_selfplay

    # Also need to modify _add_to_replay_buffer to handle raw examples
    original_add = trainer._add_to_replay_buffer

    def distributed_add_to_replay_buffer(data):
        """Add examples to replay buffer (handles both games and raw examples)."""
        if data and isinstance(data[0], tuple):
            # Already training examples
            trainer.replay_buffer.add_examples(data)
        else:
            # Games (use original method)
            original_add(data)

    trainer._add_to_replay_buffer = distributed_add_to_replay_buffer

    # Train
    num_iterations = config['training']['num_iterations']
    print(f"\nStarting DISTRIBUTED training for {num_iterations} iterations...")
    print(f"Workers: {num_workers}")
    print(f"Games per iteration: {config['selfplay']['games_per_iteration']}")
    print("=" * 60)

    trainer.train(num_iterations)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
