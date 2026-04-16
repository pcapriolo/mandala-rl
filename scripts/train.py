#!/usr/bin/env python3
"""
Main training script for Mandala RL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest.pt
"""
import argparse
import hashlib
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    parser.add_argument('--game', type=str, default='mandala',
                      choices=['mandala', 'lost_cities', 'dominion'],
                      help='Game to train (default: mandala)')
    parser.add_argument('--flush-buffer', action='store_true',
                      help='Discard replay buffer from checkpoint (keep weights)')
    parser.add_argument('--seed-buffer', type=str, default=None,
                      help='Path to seed buffer (.pkl) to pre-fill replay buffer')
    parser.add_argument('--pretrain-epochs', type=int, default=0,
                      help='Pre-train on seed buffer for N epochs before self-play')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config_hash = hashlib.sha256(yaml.dump(config, sort_keys=True).encode()).hexdigest()[:12]
    print(f"Loaded config from {args.config} (hash: {config_hash})")

    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Create game
    if args.game == 'mandala':
        from mandala_rl.game.engine import MandalaGame
        game = MandalaGame()
        print("Created Mandala game engine")
    elif args.game == 'lost_cities':
        from lost_cities.game.engine import LostCitiesGame
        game = LostCitiesGame()
        print("Created Lost Cities game engine")
    elif args.game == 'dominion':
        from dominion.game.engine import DominionGame
        game = DominionGame()
        print("Created Dominion game engine")

    # Create network
    network_config = config['network']
    network = MandalaNet(
        input_channels=network_config['input_channels'],
        num_actions=network_config['num_actions'],
        num_res_blocks=network_config['num_res_blocks'],
        channels=network_config['channels'],
        belief_size=network_config.get('belief_size', 12),
        phase_aware_policy=network_config.get('phase_aware_policy', False),
        factored_policy=network_config.get('factored_policy', False),
    )
    print(f"Created network with {sum(p.numel() for p in network.parameters())} parameters")

    # Prepare training config
    training_config = {
        # MCTS
        'mcts_simulations': config['mcts']['num_simulations'],
        'c_puct': config['mcts']['c_puct'],
        'dirichlet_alpha': config['mcts'].get('dirichlet_alpha', 0.3),
        'dirichlet_epsilon': config['mcts'].get('dirichlet_epsilon', 0.25),
        'temperature': config['selfplay']['temperature'],
        'temperature_threshold': config['selfplay']['temperature_threshold'],

        # Self-play
        'games_per_iteration': config['selfplay']['games_per_iteration'],
        'parallel_games': config['selfplay'].get('parallel_games', 8),
        'leaves_per_game': config['selfplay'].get('leaves_per_game', 1),
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
        'input_channels': network_config['input_channels'],
        'num_actions': network_config['num_actions'],
        'num_res_blocks': network_config['num_res_blocks'],
        'channels': network_config['channels'],
        'belief_size': network_config.get('belief_size', 12),

        # Paths
        'checkpoint_dir': config['paths']['checkpoint_dir'],
        'replay_dir': config['paths']['replay_dir'],
        'log_dir': config['paths']['log_dir'],
        'elo_file': config['paths']['elo_file'],

        # Replay saving
        'save_replay_frequency': config.get('save_replay_frequency', 10),

        # Disk management
        'prune_old_checkpoints': config['training'].get('prune_old_checkpoints', False),

        # Entropy regularization
        'entropy_weight': config['training'].get('entropy_weight', 0.0),

        # Degenerate game filtering
        'max_discard_rate': config['training'].get('max_discard_rate', 0),

        # Seed buffer re-injection
        'seed_reinject_frequency': config['training'].get('seed_reinject_frequency', 0),

        # LC quality filtering
        'min_play_rate': config['training'].get('min_play_rate', 0),

        # Policy weight schedule (decays from base to 1.0 over 256 iterations)
        'policy_weight': config['training'].get('policy_weight', 1.0),

        # Checkpoint every N games within iteration (for resume)
        'checkpoint_every_n_games': config['training'].get('checkpoint_every_n_games', 10),

        # Opponent diversity (fraction of full-sim games played vs older checkpoint)
        'opponent_diversity_ratio': config['selfplay'].get('opponent_diversity_ratio', 0.0),
        'opponent_iter_min': config['selfplay'].get('opponent_iter_min', None),
        'opponent_iter_max': config['selfplay'].get('opponent_iter_max', None),

        # Explore epsilon (Dominion: Big Money prior blending in buy phase)
        'explore_epsilon': config['selfplay'].get('explore_epsilon', 0.0),

        # Min buffer for training (skip training until buffer has enough diversity)
        'min_buffer_for_training': config['training'].get('min_buffer_for_training', 0),

        # Action exploration boost (Dominion: multiply PLAY priors in ACTION phase)
        'action_explore_boost': config['mcts'].get('action_explore_boost', 0.0),

        # Action buy forcing (Dominion: epsilon-greedy buy of kingdom action cards)
        'action_buy_force_rate': config['mcts'].get('action_buy_force_rate', 0.0),

        # Action play forcing (Dominion: epsilon-greedy play of action cards in hand)
        'action_play_force_rate': config['mcts'].get('action_play_force_rate', 0.0),

        # Curriculum: max action cards in kingdom (0 = Phase 0, no action cards)
        'max_action_cards': config.get('max_action_cards', 10),
        'big_money_force_rate': config.get('big_money_force_rate', 0.0),
        'forced_kingdom_cards': config.get('forced_kingdom_cards', []),
        'disabled_basic_supply': config.get('disabled_basic_supply', []),
        'province_supply': config.get('province_supply', 8),

        # Game rules (Dominion curriculum)
        'draw_penalty': config.get('draw_penalty', 0.0),
        'max_turns': config.get('max_turns', 0),

        # Deploy
        'deploy_frequency': config.get('deploy_frequency', 0),
    }

    # Catch-all: merge top-level scalar/list keys from YAML not already in training_config.
    # This prevents silent config drops (e.g., draw_penalty, max_turns, deploy_frequency)
    # when new keys are added to YAML but not manually listed above.
    for key, value in config.items():
        if not isinstance(value, dict) and key not in training_config:
            training_config[key] = value

    # Create trainer
    trainer = Trainer(
        game=game,
        network=network,
        config=training_config,
        device=device,
        config_path=args.config
    )
    print("Created trainer")

    # Resume from checkpoint if specified or if latest exists
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

    # Flush replay buffer if requested (keep weights, discard poisoned data)
    if args.flush_buffer:
        trainer.replay_buffer.buffer.clear()
        print("Flushed replay buffer (keeping network weights)")

    # Seed replay buffer from bot-generated data
    if args.seed_buffer:
        seed_path = Path(args.seed_buffer)
        if seed_path.exists():
            from mandala_rl.training.replay_buffer import ReplayBuffer
            seed_buf = ReplayBuffer(max_size=200000)
            seed_buf.load(seed_path)
            trainer.replay_buffer.add_examples(seed_buf.get_all_data())
            print(f"Seeded replay buffer with {len(seed_buf)} examples from {seed_path}")
            # Cache for periodic re-injection
            trainer.set_seed_buffer(seed_path)
        else:
            print(f"Warning: Seed buffer {seed_path} not found, skipping")

    # Pre-train on seed data before self-play
    if args.pretrain_epochs > 0 and len(trainer.replay_buffer) > 0:
        print(f"\n{'='*60}")
        print(f"Pre-training on seed data for {args.pretrain_epochs} epochs...")
        print(f"Buffer: {len(trainer.replay_buffer)} examples")
        print(f"{'='*60}")
        original_epochs = trainer.config['epochs_per_iteration']
        trainer.config['epochs_per_iteration'] = args.pretrain_epochs
        trainer._train_network()
        trainer.config['epochs_per_iteration'] = original_epochs
        trainer._save_checkpoint()
        print(f"Pre-training complete.")

    # Train
    num_iterations = args.iterations or config['training']['num_iterations']
    print(f"\nStarting training for {num_iterations} iterations...")
    print("=" * 60)

    trainer.train(num_iterations)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
