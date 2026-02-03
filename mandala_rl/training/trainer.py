"""Main training loop."""
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from ..game.engine import MandalaGame
from ..network.model import MandalaNet
from ..selfplay.worker import SelfPlayWorker
from .replay_buffer import ReplayBuffer
from ..evaluation.elo import EloRating
from ..evaluation.arena import Arena


class Trainer:
    """
    Main training orchestrator.

    Handles:
    - Self-play game generation
    - Network training
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        game: MandalaGame,
        network: MandalaNet,
        config: dict,
        device: str = "mps"
    ):
        """
        Args:
            game: Game engine
            network: Neural network
            config: Training configuration
            device: PyTorch device
        """
        self.game = game
        self.network = network.to(device)
        self.device = device
        self.config = config

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.get('replay_buffer_size', 500000)
        )

        # Self-play worker
        self.selfplay_worker = SelfPlayWorker(
            game=game,
            network=network,
            mcts_simulations=config.get('mcts_simulations', 800),
            temperature=config.get('temperature', 1.0),
            temperature_threshold=config.get('temperature_threshold', 30),
            c_puct=config.get('c_puct', 1.0),
            device=device
        )

        # Optimizer
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_milestones', [100, 200, 300]),
            gamma=config.get('lr_gamma', 0.1)
        )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'data/logs'))

        # Elo rating system
        self.elo = EloRating(initial_rating=1500.0, k_factor=32.0)
        self.elo_file = Path(config.get('elo_file', 'data/elo_ratings.json'))
        if self.elo_file.exists():
            self.elo.load(self.elo_file)

        # Arena for evaluation
        self.arena = Arena(
            game=game,
            mcts_simulations=config.get('eval_mcts_simulations', 400),
            c_puct=config.get('c_puct', 1.0),
            device=device
        )

        # Training state
        self.iteration = 0
        self.total_games = 0
        self.best_checkpoint = None

    def train(self, num_iterations: int):
        """
        Run training loop.

        Each iteration:
        1. Generate self-play games
        2. Add examples to replay buffer
        3. Train network on replay buffer
        4. Save checkpoint

        Args:
            num_iterations: Number of training iterations
        """
        for iteration in range(num_iterations):
            self.iteration += 1
            print(f"\n{'='*60}")
            print(f"Iteration {self.iteration}")
            print(f"{'='*60}")

            # 1. Self-play
            print("\n[1/3] Generating self-play games...")
            games = self._generate_selfplay_games()
            print(f"Generated {len(games)} games")

            # 2. Add to replay buffer
            print("\n[2/3] Adding examples to replay buffer...")
            self._add_to_replay_buffer(games)
            print(f"Replay buffer size: {len(self.replay_buffer)}")

            # 3. Train network
            print("\n[3/3] Training network...")
            self._train_network()

            # 4. Save checkpoint
            self._save_checkpoint()

            # 5. Evaluate (periodically)
            eval_freq = self.config.get('eval_frequency', 10)
            if self.iteration > 0 and self.iteration % eval_freq == 0:
                self._evaluate_checkpoint()

            # Step scheduler
            self.scheduler.step()

    def _generate_selfplay_games(self) -> list:
        """Generate self-play games."""
        num_games = self.config.get('games_per_iteration', 100)

        # Update worker's network to latest
        self.selfplay_worker.network.load_state_dict(self.network.state_dict())

        games = []
        for _ in tqdm(range(num_games), desc="Self-play"):
            game = self.selfplay_worker.play_game()
            games.append(game)

        self.total_games += num_games
        self.writer.add_scalar('Training/TotalGames', self.total_games, self.iteration)

        return games

    def _add_to_replay_buffer(self, games: list):
        """Extract examples from games and add to buffer."""
        all_examples = []
        for game in games:
            examples = self.selfplay_worker.get_training_examples(game)
            all_examples.extend(examples)

        self.replay_buffer.add_examples(all_examples)
        self.writer.add_scalar('Training/BufferSize', len(self.replay_buffer), self.iteration)

    def _train_network(self):
        """Train network on replay buffer."""
        if len(self.replay_buffer) < self.config.get('batch_size', 256):
            print("Not enough examples in buffer yet")
            return

        num_epochs = self.config.get('epochs_per_iteration', 10)
        batch_size = self.config.get('batch_size', 256)

        self.network.train()

        for epoch in range(num_epochs):
            # Sample batch
            states, policies, values = self.replay_buffer.sample(batch_size)

            # Convert to tensors
            states = torch.from_numpy(states).to(self.device)
            policies = torch.from_numpy(policies).to(self.device)
            values = torch.from_numpy(values).to(self.device)

            # Forward pass
            total_loss, policy_loss, value_loss = self.network.get_loss(
                states, policies, values
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Log
            if epoch == 0:
                self.writer.add_scalar('Loss/Total', total_loss.item(), self.iteration)
                self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.iteration)
                self.writer.add_scalar('Loss/Value', value_loss.item(), self.iteration)
                self.writer.add_scalar('Training/LearningRate',
                                     self.optimizer.param_groups[0]['lr'],
                                     self.iteration)

                print(f"Loss - Total: {total_loss.item():.4f}, "
                      f"Policy: {policy_loss.item():.4f}, "
                      f"Value: {value_loss.item():.4f}")

    def _save_checkpoint(self):
        """Save network checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'model_latest.pt')

        # Save periodic checkpoint
        if self.iteration % self.config.get('checkpoint_frequency', 10) == 0:
            torch.save(checkpoint, checkpoint_dir / f'model_iter_{self.iteration}.pt')

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']

        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from iteration {self.iteration}")

    def _evaluate_checkpoint(self):
        """Evaluate current model against previous checkpoint."""
        print(f"\n[EVALUATION] Testing iteration {self.iteration}")

        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        current_model_id = f"iter_{self.iteration}"

        # Find previous checkpoint to compare against
        eval_freq = self.config.get('eval_frequency', 10)
        prev_iteration = self.iteration - eval_freq
        prev_checkpoint = checkpoint_dir / f'model_iter_{prev_iteration}.pt'

        if not prev_checkpoint.exists():
            print(f"No baseline checkpoint found (iter {prev_iteration}), skipping evaluation")
            # Register current model with initial Elo
            self.elo.get_rating(current_model_id)
            self._save_elo()
            return

        # Load both models
        print(f"Playing {current_model_id} vs iter_{prev_iteration}")

        current_model = MandalaNet(
            input_channels=50,
            num_actions=30,
            num_res_blocks=self.config.get('num_res_blocks', 10),
            channels=self.config.get('channels', 128)
        ).to(self.device)
        current_model.load_state_dict(self.network.state_dict())

        prev_model = MandalaNet(
            input_channels=50,
            num_actions=30,
            num_res_blocks=self.config.get('num_res_blocks', 10),
            channels=self.config.get('channels', 128)
        ).to(self.device)
        prev_checkpoint_data = torch.load(prev_checkpoint, map_location=self.device)
        prev_model.load_state_dict(prev_checkpoint_data['model_state_dict'])

        # Play match
        num_eval_games = self.config.get('eval_num_games', 20)
        results = self.arena.play_match(
            current_model,
            prev_model,
            num_games=num_eval_games,
            seed=self.iteration
        )

        # Update Elo
        prev_model_id = f"iter_{prev_iteration}"
        winner = None
        if results['model1_wins'] > results['model2_wins']:
            winner = current_model_id
        elif results['model2_wins'] > results['model1_wins']:
            winner = prev_model_id

        self.elo.record_match(current_model_id, prev_model_id, winner)

        # Log results
        current_elo = self.elo.get_rating(current_model_id)
        prev_elo = self.elo.get_rating(prev_model_id)
        win_rate = results['model1_wins'] / results['total_games']

        print(f"\nResults:")
        print(f"  {current_model_id}: {results['model1_wins']} wins, Elo: {current_elo:.1f}")
        print(f"  {prev_model_id}: {results['model2_wins']} wins, Elo: {prev_elo:.1f}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win rate: {win_rate:.2%}")

        # Log to Tensorboard
        self.writer.add_scalar('Evaluation/WinRate', win_rate, self.iteration)
        self.writer.add_scalar('Evaluation/CurrentElo', current_elo, self.iteration)
        self.writer.add_scalar('Evaluation/Wins', results['model1_wins'], self.iteration)
        self.writer.add_scalar('Evaluation/Losses', results['model2_wins'], self.iteration)

        # Update best checkpoint
        if self.best_checkpoint is None or current_elo > self.elo.get_rating(self.best_checkpoint):
            self.best_checkpoint = current_model_id
            print(f"  ⭐ New best: {current_model_id} (Elo: {current_elo:.1f})")

        # Save Elo ratings
        self._save_elo()

    def _save_elo(self):
        """Save Elo ratings to file."""
        self.elo_file.parent.mkdir(parents=True, exist_ok=True)
        self.elo.save(self.elo_file)
        print(f"Saved Elo ratings to {self.elo_file}")
