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

        # Training state
        self.iteration = 0
        self.total_games = 0

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
