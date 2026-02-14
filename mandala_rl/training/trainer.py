"""Main training loop."""
import torch
import torch.optim as optim
import numpy as np
import subprocess
import sys
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
        device: str = "mps",
        config_path: str = "configs/default.yaml"
    ):
        """
        Args:
            game: Game engine
            network: Neural network
            config: Training configuration
            device: PyTorch device
            config_path: Path to YAML config (for background eval subprocess)
        """
        self.game = game
        self.network = network.to(device)
        self.device = device
        self.config = config
        self._config_path = config_path

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

        # Mixed precision for CUDA (2x throughput on A100/H100)
        self.use_amp = device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # torch.compile for CUDA (20-40% kernel fusion speedup)
        # Store unwrapped reference BEFORE compiling — torch.compile wraps
        # the model, prefixing state_dict keys with '_orig_mod.' which breaks
        # load_state_dict on uncompiled copies (worker, evaluation models).
        self._unwrapped_network = self.network
        if device == 'cuda':
            try:
                self.network = torch.compile(self.network)
                print("torch.compile enabled")
            except Exception:
                pass

        # Training state
        self.iteration = 0
        self.total_games = 0
        self.games_in_current_iteration = 0  # Track progress within iteration
        self.best_checkpoint = None
        self._eval_process = None  # Background eval subprocess

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
        resuming_mid_iteration = self.games_in_current_iteration > 0

        for iteration in range(num_iterations):
            if resuming_mid_iteration:
                resuming_mid_iteration = False
            else:
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

            # 5. Evaluate (async — runs on CPU in background, never blocks training)
            eval_freq = self.config.get('eval_frequency', 10)
            if eval_freq > 0 and self.iteration > 0 and self.iteration % eval_freq == 0:
                self._start_async_eval()

            # 6. Clean up disk
            self._cleanup_checkpoints()

            # Step scheduler
            self.scheduler.step()

    def _generate_selfplay_games(self) -> list:
        """Generate self-play games using batched parallel play."""
        num_games = self.config.get('games_per_iteration', 100)
        replay_dir = Path(self.config.get('replay_dir', 'data/replays'))
        save_replay_freq = self.config.get('save_replay_frequency', 10)
        checkpoint_every_n_games = self.config.get('checkpoint_every_n_games', 10)
        parallel_games = self.config.get('parallel_games', 8)

        # Update worker's network to latest
        self.selfplay_worker.network.load_state_dict(self._unwrapped_network.state_dict())

        # Resume from where we left off if mid-iteration
        start_game = self.games_in_current_iteration
        remaining = num_games - start_game
        if start_game > 0:
            print(f"Resuming from game {start_game + 1}/{num_games} in current iteration")

        games = []
        progress = tqdm(total=num_games, desc="Self-play", initial=start_game)

        def on_game_complete(game_idx, game_record):
            self.games_in_current_iteration = start_game + game_idx + 1
            self.total_games += 1
            progress.update(1)

            # Save checkpoint every N games
            if self.games_in_current_iteration % checkpoint_every_n_games == 0:
                tqdm.write(f"Checkpoint after game {self.games_in_current_iteration}/{num_games}")
                self._save_checkpoint(suffix=f"_game{self.total_games}")

        batch_games = self.selfplay_worker.play_games_batched(
            num_games=remaining,
            batch_size=parallel_games,
            save_dir=replay_dir,
            iteration=self.iteration,
            save_replay_freq=save_replay_freq,
            on_game_complete=on_game_complete,
        )
        games.extend(batch_games)
        progress.close()

        # Reset counter at end of iteration
        self.games_in_current_iteration = 0

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
        batch_size = self.config.get('batch_size', 256)
        if len(self.replay_buffer) < batch_size:
            print("Not enough examples in buffer yet")
            return

        num_epochs = self.config.get('epochs_per_iteration', 10)
        batches_per_epoch = max(1, len(self.replay_buffer) // batch_size)

        self.network.train()

        total_steps = 0
        epoch_total_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0

        for epoch in range(num_epochs):
            for batch_idx in range(batches_per_epoch):
                states, policies, values = self.replay_buffer.sample(batch_size)

                states = torch.from_numpy(states.astype(np.float32)).to(self.device)
                policies = torch.from_numpy(policies.astype(np.float32)).to(self.device)
                values = torch.from_numpy(values.astype(np.float32)).to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    total_loss, policy_loss, value_loss = self.network.get_loss(
                        states, policies, values
                    )

                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                    self.optimizer.step()

                epoch_total_loss = total_loss.item()
                epoch_policy_loss = policy_loss.item()
                epoch_value_loss = value_loss.item()
                total_steps += 1

        # Log final loss values
        self.writer.add_scalar('Loss/Total', epoch_total_loss, self.iteration)
        self.writer.add_scalar('Loss/Policy', epoch_policy_loss, self.iteration)
        self.writer.add_scalar('Loss/Value', epoch_value_loss, self.iteration)
        self.writer.add_scalar('Training/LearningRate',
                             self.optimizer.param_groups[0]['lr'],
                             self.iteration)

        print(f"Loss - Total: {epoch_total_loss:.4f}, "
              f"Policy: {epoch_policy_loss:.4f}, "
              f"Value: {epoch_value_loss:.4f} "
              f"({total_steps} gradient steps)")

    def _save_checkpoint(self, suffix=''):
        """Save network checkpoint and replay buffer.

        Args:
            suffix: Optional suffix for checkpoint filename (e.g., '_game500').
                    Game-level checkpoints are lightweight (no replay buffer).
        """
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'games_in_current_iteration': self.games_in_current_iteration,
            'model_state_dict': self._unwrapped_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        if suffix:
            # Game-level checkpoint: lightweight only (no replay buffer, no model_latest)
            torch.save(checkpoint, checkpoint_dir / f'model_latest{suffix}.pt')
        else:
            # Full checkpoint: save model_latest with replay buffer for resume
            latest_checkpoint = dict(checkpoint)
            latest_checkpoint['replay_buffer'] = self.replay_buffer.get_all_data()
            torch.save(latest_checkpoint, checkpoint_dir / 'model_latest.pt')

            # Save periodic iteration checkpoint (lightweight, no replay buffer)
            if self.iteration % self.config.get('checkpoint_frequency', 10) == 0:
                torch.save(checkpoint, checkpoint_dir / f'model_iter_{self.iteration}.pt')

        status = f"iteration {self.iteration}"
        if self.games_in_current_iteration > 0:
            status += f", game {self.games_in_current_iteration}/{self.config.get('games_per_iteration', 100)}"
        print(f"Saved checkpoint to {checkpoint_dir} ({status}, buffer: {len(self.replay_buffer)})")

    def _cleanup_checkpoints(self):
        """Remove stale checkpoints to prevent disk from filling up."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        if not checkpoint_dir.exists():
            return

        # 1. Delete all game-level checkpoints (only useful during mid-iteration resume)
        game_checkpoints = list(checkpoint_dir.glob('model_latest_game*.pt'))
        for f in game_checkpoints:
            f.unlink()
        if game_checkpoints:
            print(f"Cleaned up {len(game_checkpoints)} game-level checkpoints")

        # 2. Prune old iteration checkpoints, keep last 20
        keep_last = 20
        iter_checkpoints = sorted(
            checkpoint_dir.glob('model_iter_*.pt'),
            key=lambda f: int(f.stem.split('_')[-1])
        )
        if len(iter_checkpoints) > keep_last:
            to_delete = iter_checkpoints[:-keep_last]
            for f in to_delete:
                f.unlink()
            print(f"Pruned {len(to_delete)} old iteration checkpoints")

    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint and replay buffer."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self._unwrapped_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']

        # NEW: Restore game progress within iteration
        self.games_in_current_iteration = checkpoint.get('games_in_current_iteration', 0)

        # Restore replay buffer if present
        if 'replay_buffer' in checkpoint:
            self.replay_buffer.load_data(checkpoint['replay_buffer'])
            print(f"Restored replay buffer with {len(self.replay_buffer)} examples")

        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from iteration {self.iteration}, total games: {self.total_games}")
        if self.games_in_current_iteration > 0:
            print(f"Mid-iteration: will continue from game {self.games_in_current_iteration + 1}")

    def _start_async_eval(self):
        """Spawn background eval subprocess on CPU. Never blocks training."""
        # Skip if previous eval is still running
        if self._eval_process is not None and self._eval_process.poll() is None:
            print(f"[EVAL] Skipping iter {self.iteration} — previous eval still running")
            return

        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        prev_checkpoint = checkpoint_dir / f'model_iter_{self.iteration - 1}.pt'
        curr_checkpoint = checkpoint_dir / f'model_iter_{self.iteration}.pt'

        if not prev_checkpoint.exists() or not curr_checkpoint.exists():
            print(f"[EVAL] Skipping iter {self.iteration} — checkpoint not found")
            return

        eval_script = Path(__file__).parent.parent.parent / 'scripts' / 'eval_worker.py'
        if not eval_script.exists():
            print(f"[EVAL] eval_worker.py not found at {eval_script}")
            return

        cmd = [
            sys.executable, str(eval_script),
            '--config', str(self._config_path),
            '--iteration', str(self.iteration),
            '--checkpoint-dir', str(checkpoint_dir),
            '--elo-file', str(self.elo_file),
            '--log-dir', str(self.config.get('log_dir', 'data/logs')),
        ]

        log_dir = Path(self.config.get('log_dir', 'data/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_log = open(log_dir / 'eval_worker.log', 'a')

        self._eval_process = subprocess.Popen(cmd, stdout=eval_log, stderr=subprocess.STDOUT)
        print(f"[EVAL] Started background eval for iter {self.iteration} (PID {self._eval_process.pid})")

    def _evaluate_checkpoint(self):
        """Evaluate current model against previous checkpoint (synchronous, legacy)."""
        print(f"\n[EVALUATION] Testing iteration {self.iteration}")

        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        current_model_id = f"iter_{self.iteration}"

        # Find previous checkpoint to compare against (always compare against iteration - 1)
        prev_iteration = self.iteration - 1
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
            input_channels=self.config.get('input_channels', 50),
            num_actions=self.config.get('num_actions', 30),
            num_res_blocks=self.config.get('num_res_blocks', 10),
            channels=self.config.get('channels', 128)
        ).to(self.device)
        current_model.load_state_dict(self._unwrapped_network.state_dict())

        prev_model = MandalaNet(
            input_channels=self.config.get('input_channels', 50),
            num_actions=self.config.get('num_actions', 30),
            num_res_blocks=self.config.get('num_res_blocks', 10),
            channels=self.config.get('channels', 128)
        ).to(self.device)
        prev_checkpoint_data = torch.load(prev_checkpoint, map_location=self.device, weights_only=False)
        prev_model.load_state_dict(prev_checkpoint_data['model_state_dict'])

        # Play match
        num_eval_games = self.config.get('eval_num_games', 20)
        results = self.arena.play_match(
            current_model,
            prev_model,
            num_games=num_eval_games,
            seed=self.iteration
        )

        # Batch Elo update (single update using total scores to avoid order bias)
        prev_model_id = f"iter_{prev_iteration}"
        total = results['total_games']
        score_current = results['model1_wins'] + 0.5 * results['draws']
        score_prev = results['model2_wins'] + 0.5 * results['draws']

        rating_current = self.elo.get_rating(current_model_id)
        rating_prev = self.elo.get_rating(prev_model_id)
        expected_current = total / (1.0 + 10 ** ((rating_prev - rating_current) / 400.0))

        self.elo.ratings[current_model_id] = rating_current + self.elo.k_factor * (score_current - expected_current)
        self.elo.ratings[prev_model_id] = rating_prev + self.elo.k_factor * (score_prev - (total - expected_current))

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
