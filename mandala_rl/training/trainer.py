"""Main training loop."""
import json
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import yaml
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
        config_path: str = "configs/default.yaml",
        config_schema: Optional[object] = None,
    ):
        """
        Args:
            game: Game engine
            network: Neural network
            config: Training configuration
            device: PyTorch device
            config_path: Path to YAML config
            config_schema: Typed DominionConfig for schema-based hot-reload
                (Dominion only; None for Mandala/LC uses legacy whitelists).
        """
        self.game = game
        self.network = network.to(device)
        self.device = device
        self.config = config
        self._config_path = config_path
        self._config_schema = config_schema

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.get('replay_buffer_size', 500000)
        )

        # Seed buffer for periodic re-injection
        self._seed_buffer_data = None

        # Self-play worker
        self.selfplay_worker = SelfPlayWorker(
            game=game,
            network=network,
            mcts_simulations=config.get('mcts_simulations', 800),
            temperature=config.get('temperature', 1.0),
            temperature_threshold=config.get('temperature_threshold', 30),
            explore_epsilon=config.get('explore_epsilon', 0.0),
            c_puct=config.get('c_puct', 1.0),
            dirichlet_alpha=config.get('dirichlet_alpha', 0.3),
            dirichlet_epsilon=config.get('dirichlet_epsilon', 0.25),
            device=device,
            leaves_per_game=config.get('leaves_per_game', 1),
            action_explore_boost=config.get('action_explore_boost', 0.0),
            action_buy_force_rate=config.get('action_buy_force_rate', 0.0),
            action_play_force_rate=config.get('action_play_force_rate', 0.0),
            max_action_cards=config.get('max_action_cards', 10),
            big_money_force_rate=config.get('big_money_force_rate', 0.0),
            forced_kingdom_cards=config.get('forced_kingdom_cards', []),
            disabled_basic_supply=config.get('disabled_basic_supply', []),
            province_supply=config.get('province_supply', 8),
            draw_penalty=config.get('draw_penalty', 0.0),
            drop_draws=config.get('drop_draws', False),
            max_turns=config.get('max_turns', 0),
            mcts_leaf_eval_source=config.get('mcts_leaf_eval_source', 'score'),
            opponent_disabled_supply=config.get('opponent_disabled_supply', []),
        )

        # Log curriculum params at startup to catch silent config drops
        curriculum = {"province_supply": config.get("province_supply", 8),
                      "max_action_cards": config.get("max_action_cards", 10),
                      "disabled_basic_supply": config.get("disabled_basic_supply", []),
                      "big_money_force_rate": config.get("big_money_force_rate", 0.0),
                      "forced_kingdom_cards": config.get("forced_kingdom_cards", []),
                      "max_turns": config.get("max_turns", 0)}
        print(f"[CONFIG] Curriculum params: {curriculum}")

        # Optimizer
        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
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

        # Warmup target: skip gradient updates until buffer reaches this size.
        # Set by --warmup-to-full; auto-cleared when threshold crossed.
        self._warmup_target = 0

    def _init_heartbeat_path(self):
        """Cache the heartbeat path once at startup."""
        if not hasattr(self, '_heartbeat_path'):
            self._heartbeat_path = Path(self.config.get('checkpoint_dir', 'data/checkpoints')).parent / 'heartbeat.json'

    def _write_heartbeat(self, phase: str, detail: str = ""):
        """Write a heartbeat JSON so the observer can show live progress."""
        self._init_heartbeat_path()
        num_games = self.config.get('games_per_iteration', 100)
        hb = {
            'iteration': self.iteration,
            'phase': phase,
            'detail': detail,
            'game': self.games_in_current_iteration,
            'games_total': num_games,
            'total_games': self.total_games,
            'timestamp': time.time(),
        }
        try:
            self._heartbeat_path.write_text(json.dumps(hb))
        except Exception as e:
            print(f"[heartbeat] write failed: {e}")

    def _get_lr_for_iteration(self, iteration: int) -> float:
        """Compute LR based on absolute iteration (restart-resilient).

        Default lr_milestones is [] — no silent LR schedule. If milestones are
        explicitly configured and one is crossed, print a tripwire line.
        """
        lr = self.config.get('learning_rate', 1e-3)
        gamma = self.config.get('lr_gamma', 0.3)
        base_lr = lr
        for milestone in sorted(self.config.get('lr_milestones', [])):
            if iteration >= milestone:
                lr *= gamma
        if lr != base_lr and iteration != getattr(self, '_last_logged_lr_iter', -1):
            print(f"[LR] Milestone schedule applied at iter {iteration}: base {base_lr:.2e} → effective {lr:.2e}")
            self._last_logged_lr_iter = iteration
        return lr

    # Tunable config keys: maps YAML path to SelfPlayWorker attribute name
    _TUNABLE_KEYS = {
        ('mcts', 'dirichlet_epsilon'): 'dirichlet_epsilon',
        ('mcts', 'dirichlet_alpha'): 'dirichlet_alpha',
        ('mcts', 'c_puct'): 'c_puct',
        ('mcts', 'action_explore_boost'): 'action_explore_boost',
        ('mcts', 'action_buy_force_rate'): 'action_buy_force_rate',
        ('mcts', 'action_play_force_rate'): 'action_play_force_rate',
        ('selfplay', 'temperature'): 'temperature',
        ('selfplay', 'temperature_threshold'): 'temperature_threshold',
        ('selfplay', 'explore_epsilon'): 'explore_epsilon',
    }

    # Top-level YAML keys that flow to SelfPlayWorker attributes.
    # Curriculum keys change the game/supply distribution — a flip means
    # the replay buffer now mixes examples from two distributions.
    _WORKER_TOP_KEYS = [
        ('big_money_force_rate', 'big_money_force_rate', False),
        ('draw_penalty', 'draw_penalty', False),
        ('max_turns', 'max_turns', False),
        ('province_supply', 'province_supply', True),
        ('max_action_cards', 'max_action_cards', True),
        ('disabled_basic_supply', 'disabled_basic_supply', True),
        ('forced_kingdom_cards', 'forced_kingdom_cards', True),
        ('drop_draws', 'drop_draws', False),
    ]

    # Top-level YAML keys that are read via self.config.get(...) on every use,
    # so updating self.config is enough to hot-reload them.
    _CONFIG_TOP_KEYS = [
        'min_buffer_for_training',
        'batch_size',
        'epochs_per_iteration',
        'eval_frequency',
        'checkpoint_frequency',
        'save_replay_frequency',
        'deploy_frequency',
        'policy_weight',
        'entropy_weight',
        'max_discard_rate',
        'checkpoint_every_n_games',
        'seed_reinject_frequency',
    ]

    # YAML-nested keys that train.py flattens into self.config at startup.
    # Hot-reload must traverse the nesting to pick up changes; without this
    # dict, the _CONFIG_TOP_KEYS loop looks for them at raw top level and
    # silently skips (they live under selfplay:).
    # Map: (yaml_section, yaml_key) -> flattened cfg key in self.config
    _CONFIG_NESTED_KEYS = {
        ('selfplay', 'opponent_diversity_ratio'): 'opponent_diversity_ratio',
        ('selfplay', 'opponent_iter_min'): 'opponent_iter_min',
        ('selfplay', 'opponent_iter_max'): 'opponent_iter_max',
    }

    def _hot_reload_config(self):
        """Re-read YAML and update tunable params.

        Dominion uses the typed schema (single source of truth, fixes the 10
        silent-skip bugs from the legacy whitelists — see DEVLOG #169). Other
        games fall through to the legacy four-whitelist path.
        """
        config_path = Path(self._config_path)
        if not config_path.exists():
            print(f"[HOT-RELOAD ERROR] config not found at {config_path}")
            return

        # Schema-based path (Dominion). No whitelists — every field's
        # hot-reload metadata lives on the dataclass.
        if self._config_schema is not None:
            try:
                changes = self._config_schema.reload_into(
                    str(config_path), self.config, self.selfplay_worker,
                )
            except (yaml.YAMLError, ValueError) as e:
                print(f"[HOT-RELOAD ERROR] {config_path}: {e}")
                return
            if changes:
                print(f"Config reload: {', '.join(changes)}")
            return

        # Legacy path (Mandala, Lost Cities).
        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f)
        except Exception as e:
            print(f"[HOT-RELOAD ERROR] {config_path}: {e}")
            return
        changes = []
        curriculum_changes = []
        for (section, key), attr in self._TUNABLE_KEYS.items():
            new_val = raw.get(section, {}).get(key)
            if new_val is None:
                continue
            old_val = getattr(self.selfplay_worker, attr, None)
            if old_val is not None and old_val != new_val:
                setattr(self.selfplay_worker, attr, new_val)
                changes.append(f"{attr} {old_val} → {new_val}")
        for top_key, attr, is_curriculum in self._WORKER_TOP_KEYS:
            if top_key not in raw:
                continue
            new_val = raw[top_key]
            old_val = getattr(self.selfplay_worker, attr, None)
            if old_val != new_val:
                setattr(self.selfplay_worker, attr, new_val)
                changes.append(f"{attr} {old_val} → {new_val}")
                if is_curriculum:
                    curriculum_changes.append(f"{attr} {old_val} → {new_val}")
        for cfg_key in self._CONFIG_TOP_KEYS:
            if cfg_key not in raw:
                continue
            new_val = raw[cfg_key]
            old_val = self.config.get(cfg_key)
            if old_val != new_val:
                self.config[cfg_key] = new_val
                changes.append(f"config.{cfg_key} {old_val} → {new_val}")
        for (section, yaml_key), cfg_key in self._CONFIG_NESTED_KEYS.items():
            new_val = raw.get(section, {}).get(yaml_key)
            if new_val is None:
                continue
            old_val = self.config.get(cfg_key)
            if old_val != new_val:
                self.config[cfg_key] = new_val
                changes.append(f"config.{cfg_key} {old_val} → {new_val}")
        if changes:
            print(f"Config reload: {', '.join(changes)}")
        if curriculum_changes:
            print(f"[WARNING] Curriculum keys changed mid-run: {', '.join(curriculum_changes)}. "
                  f"Replay buffer now mixes examples from two distributions.")

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

            # 0. Hot-reload tunable config values from YAML
            self._hot_reload_config()

            # 1. Self-play
            print("\n[1/3] Generating self-play games...")
            num_games = self.config.get('games_per_iteration', 100)
            self._write_heartbeat('selfplay', f'0/{num_games}')
            games = self._generate_selfplay_games()
            # Log quality from ALL games (fast+full), train only on full-sim games
            all_games = getattr(self, '_all_games_for_quality', games)
            print(f"Generated {len(all_games)} games ({len(games)} full-sim for training)")
            self._log_game_quality(all_games)

            # 2. Filter degenerate games + add to replay buffer
            print("\n[2/3] Adding examples to replay buffer...")
            self._write_heartbeat('training', 'adding to buffer')
            games = self._filter_degenerate_games(games)

            # Re-inject seed buffer periodically
            reinject_freq = self.config.get('seed_reinject_frequency', 0)
            if reinject_freq > 0 and self.iteration % reinject_freq == 0:
                self._reinject_seed_buffer()

            if games:
                self._add_to_replay_buffer(games)
            else:
                print("  All games filtered — training on buffer only this iteration")
            print(f"Replay buffer size: {len(self.replay_buffer)}")

            # 3. Train network
            print("\n[3/3] Training network...")
            self._write_heartbeat('training', 'gradient updates')
            self._train_network()

            # 4. Save checkpoint
            self._write_heartbeat('checkpoint', 'saving')
            self._save_checkpoint()

            # 5. Auto-export deploy checkpoint
            deploy_freq = self.config.get('deploy_frequency', 0)
            if deploy_freq > 0 and self.iteration % deploy_freq == 0:
                self._export_deploy_checkpoint()

            # 6. Evaluate (async — runs on CPU in background, never blocks training)
            eval_freq = self.config.get('eval_frequency', 10)
            if eval_freq > 0 and self.iteration > 0 and self.iteration % eval_freq == 0:
                self._start_async_eval()

            # 7. Clean up disk
            self._cleanup_checkpoints()

            # Set LR based on absolute iteration (restart-resilient)
            lr = self._get_lr_for_iteration(self.iteration)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

    def set_seed_buffer(self, path):
        """Load and cache seed buffer data for periodic re-injection."""
        seed_buf = ReplayBuffer(max_size=200000)
        seed_buf.load(Path(path))
        self._seed_buffer_data = seed_buf.get_all_data()
        print(f"Cached {len(self._seed_buffer_data)} seed examples for re-injection")

    def _reinject_seed_buffer(self):
        """Re-inject cached seed buffer into replay buffer."""
        if self._seed_buffer_data is None:
            return
        self.replay_buffer.add_examples(self._seed_buffer_data)
        print(f"Re-injected {len(self._seed_buffer_data)} seed examples into buffer")

    def _filter_degenerate_games(self, games: list) -> list:
        """Filter out games with degenerate play patterns.

        For Mandala: reject games where discard rate exceeds max_discard_rate.
        For Lost Cities: no filtering (not needed).
        """
        max_discard = self.config.get('max_discard_rate', 0)
        if max_discard <= 0:
            return games

        kept = []
        for game in games:
            s = game.summary
            if s.get('game_type') == 'mandala':
                mt = sum(s['mountain_plays'][0]) + sum(s['mountain_plays'][1])
                fld = sum(s['field_plays'][0]) + sum(s['field_plays'][1])
                disc = sum(s['discard_plays'][0]) + sum(s['discard_plays'][1])
                total = mt + fld + disc
                disc_rate = disc / max(total, 1)
                if disc_rate <= max_discard:
                    kept.append(game)
            else:
                kept.append(game)

        filtered = len(games) - len(kept)
        if filtered > 0:
            print(f"  Filtered {filtered}/{len(games)} degenerate games "
                  f"(>{max_discard:.0%} discard), kept {len(kept)}")
        return kept

    def _generate_selfplay_games(self) -> list:
        """Generate self-play games using batched parallel play.

        All games run at full MCTS simulations (DEVLOG #145: playout cap
        removed — 50-sim fast games produced noisy policy targets that
        capped Province% at ~50-60%).
        """
        num_games = self.config.get('games_per_iteration', 100)
        replay_dir = Path(self.config.get('replay_dir', 'data/replays'))
        save_replay_freq = self.config.get('save_replay_frequency', 10)
        checkpoint_every_n_games = self.config.get("checkpoint_every_n_games", 10)
        parallel_games = self.config.get('parallel_games', 8)
        full_sims = self.config.get('mcts_simulations', 800)

        # Update worker's network to latest
        self.selfplay_worker.network.load_state_dict(self._unwrapped_network.state_dict())
        self.selfplay_worker.big_money_force_rate = self._get_force_rate()

        # Resume from where we left off if mid-iteration
        start_game = self.games_in_current_iteration
        remaining_total = num_games - start_game
        if start_game > 0:
            print(f"Resuming from game {start_game + 1}/{num_games} in current iteration")

        all_games = []
        progress = tqdm(total=num_games, desc="Self-play", initial=start_game)

        def on_game_complete(game_idx, game_record):
            self.games_in_current_iteration = start_game + len(all_games) + game_idx + 1
            self.total_games += 1
            progress.update(1)
            self._write_heartbeat('selfplay', f'{self.games_in_current_iteration}/{num_games}')

            if checkpoint_every_n_games > 0 and self.games_in_current_iteration % checkpoint_every_n_games == 0:
                tqdm.write(f"Checkpoint after game {self.games_in_current_iteration}/{num_games}")
                self._save_checkpoint(suffix=f"_game{self.total_games}")

        # All games at full simulations
        self.selfplay_worker.mcts_simulations = full_sims
        games = self.selfplay_worker.play_games_batched(
            num_games=remaining_total,
            batch_size=parallel_games,
            save_dir=replay_dir,
            iteration=self.iteration,
            save_replay_freq=save_replay_freq,
            on_game_complete=on_game_complete,
        )
        all_games.extend(games)

        # Opponent diversity games (older checkpoint as opponent)
        n_opponent = 0
        opp_ratio = self.config.get('opponent_diversity_ratio', 0.0)
        if opp_ratio > 0 and self.iteration > 20:
            n_opponent = max(1, int(num_games * opp_ratio))
            opp_path = self._select_opponent_checkpoint()
            if opp_path:
                opp_network = self._load_opponent_network(opp_path)
                if opp_network is not None:
                    opp_iter = opp_path.stem.replace('model_iter_', '')
                    tqdm.write(f"Playing {n_opponent} games vs iter_{opp_iter} opponent")
                    opp_games = self.selfplay_worker.play_games_vs_opponent(
                        num_games=n_opponent,
                        opponent_network=opp_network,
                        batch_size=parallel_games,
                        save_dir=replay_dir,
                        iteration=self.iteration,
                        save_replay_freq=save_replay_freq,
                        on_game_complete=on_game_complete,
                    )
                    all_games.extend(opp_games)
                    del opp_network  # Free GPU memory

        progress.close()

        # Reset counter at end of iteration
        self.games_in_current_iteration = 0

        self.writer.add_scalar('Training/TotalGames', self.total_games, self.iteration)
        self.writer.add_scalar('Training/OpponentGames', n_opponent, self.iteration)

        self._all_games_for_quality = all_games
        return all_games

    def _add_to_replay_buffer(self, games: list):
        """Extract examples from games and add to buffer."""
        all_examples = []
        for game in games:
            examples = self.selfplay_worker.get_training_examples(game)
            all_examples.extend(examples)

        self.replay_buffer.add_examples(all_examples)
        self.writer.add_scalar('Training/BufferSize', len(self.replay_buffer), self.iteration)

    def _get_policy_weight(self) -> float:
        """Return policy weight from config (no decay — DEVLOG #145)."""
        return self.config.get('policy_weight', 1.0)

    def _get_force_rate(self) -> float:
        return self.config.get('big_money_force_rate', 0.0)

    def _train_network(self):
        """Train network on replay buffer."""
        from .replay_buffer import augment_color_permutation_batch

        batch_size = self.config.get('batch_size', 256)
        config_min = self.config.get('min_buffer_for_training', 10000)
        effective_min = max(config_min, self._warmup_target)
        buf_size = len(self.replay_buffer)
        if buf_size < effective_min:
            if self._warmup_target > 0:
                print(f"[WARMUP] Buffer {buf_size}/{self._warmup_target}, skipping training")
            else:
                print(f"Buffer too small for training ({buf_size}/{effective_min}), skipping")
            return
        if self._warmup_target > 0 and buf_size >= self._warmup_target:
            print(f"[WARMUP] Buffer reached capacity ({buf_size}/{self._warmup_target}), resuming gradient updates")
            self._warmup_target = 0

        num_epochs = self.config.get('epochs_per_iteration', 10)
        batches_per_epoch = max(1, len(self.replay_buffer) // batch_size)

        self.network.train()

        total_steps = 0
        epoch_total_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_score_loss = 0.0
        epoch_belief_loss = 0.0

        policy_weight = self._get_policy_weight()

        for epoch in range(num_epochs):
            if num_epochs > 1 and epoch % max(1, num_epochs // 10) == 0:
                print(f"  Epoch {epoch}/{num_epochs}, loss: {epoch_total_loss:.4f}", flush=True)
            for batch_idx in range(batches_per_epoch):
                states, policies, values, scores, beliefs = self.replay_buffer.sample(batch_size)

                # Color permutation augmentation (50% of the time, Mandala only — 6 colors, 137 channels)
                if np.random.random() < 0.5 and states.shape[1] == 137:
                    states, policies, beliefs = augment_color_permutation_batch(
                        states, policies, beliefs)

                states = torch.from_numpy(states.astype(np.float32)).to(self.device)
                policies = torch.from_numpy(policies.astype(np.float32)).to(self.device)
                values = torch.from_numpy(values.astype(np.float32)).to(self.device)
                scores = torch.from_numpy(scores.astype(np.float32)).to(self.device)
                beliefs = torch.from_numpy(beliefs.astype(np.float32)).to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    total_loss, policy_loss, value_loss, score_loss, belief_loss = self.network.get_loss(
                        states, policies, values,
                        target_scores=scores,
                        target_beliefs=beliefs,
                        entropy_weight=self.config.get('entropy_weight', 0.0),
                        policy_weight=policy_weight,
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
                epoch_score_loss = score_loss.item() if hasattr(score_loss, 'item') else float(score_loss)
                epoch_belief_loss = belief_loss.item() if hasattr(belief_loss, 'item') else float(belief_loss)
                total_steps += 1

        # Log final loss values
        self.writer.add_scalar('Loss/Total', epoch_total_loss, self.iteration)
        self.writer.add_scalar('Loss/Policy', epoch_policy_loss, self.iteration)
        self.writer.add_scalar('Loss/Value', epoch_value_loss, self.iteration)
        self.writer.add_scalar('Loss/Score', epoch_score_loss, self.iteration)
        self.writer.add_scalar('Loss/Belief', epoch_belief_loss, self.iteration)
        self.writer.add_scalar('Training/LearningRate',
                             self.optimizer.param_groups[0]['lr'],
                             self.iteration)
        self.writer.add_scalar('Training/PolicyWeight', policy_weight, self.iteration)

        # Log policy entropy and value head diagnostics
        self.network.eval()
        with torch.no_grad():
            policy_logits, pred_values = self.network(states)[:2]
            network_policy = F.softmax(policy_logits, dim=1)
            entropy = -torch.sum(network_policy * torch.log(network_policy + 1e-8), dim=1).mean()
            max_prob = torch.max(network_policy, dim=1)[0].mean()
            self.writer.add_scalar('Network/PolicyEntropy', entropy.item(), self.iteration)
            self.writer.add_scalar('Network/MaxActionProb', max_prob.item(), self.iteration)

            # Value head diagnostics
            pv = pred_values.squeeze()
            tv = values
            val_pred_mean = pv.mean().item()
            val_pred_std = pv.std().item()
            val_target_mean = tv.mean().item()
            val_target_std = tv.std().item()
            self.writer.add_scalar('ValueHead/PredMean', val_pred_mean, self.iteration)
            self.writer.add_scalar('ValueHead/PredStd', val_pred_std, self.iteration)
            self.writer.add_scalar('ValueHead/TargetMean', val_target_mean, self.iteration)
            self.writer.add_scalar('ValueHead/TargetStd', val_target_std, self.iteration)

        print(f"Loss - Total: {epoch_total_loss:.4f}, "
              f"Policy: {epoch_policy_loss:.4f} (w={policy_weight:.2f}), "
              f"Value: {epoch_value_loss:.4f}, "
              f"Score: {epoch_score_loss:.4f}, "
              f"Belief: {epoch_belief_loss:.4f} "
              f"({total_steps} gradient steps)")
        print(f"  Value head: pred {val_pred_mean:+.3f}±{val_pred_std:.3f}, "
              f"target {val_target_mean:+.3f}±{val_target_std:.3f}")

        # Append to losses.jsonl for dashboard charts
        losses_path = Path(self.config.get('checkpoint_dir', 'data/checkpoints')).parent / 'losses.jsonl'
        try:
            entry = {
                'iteration': self.iteration,
                'total': round(epoch_total_loss, 4),
                'policy': round(epoch_policy_loss, 4),
                'value': round(epoch_value_loss, 4),
                'score': round(epoch_score_loss, 4),
                'belief': round(epoch_belief_loss, 4),
                'policy_weight': round(policy_weight, 2),
            }
            if hasattr(self, '_game_quality'):
                entry.update(self._game_quality)
            with open(losses_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    def _log_game_quality(self, games: list):
        """Analyze completed self-play games and produce quality assessment."""
        if not games:
            return

        summaries = [g.summary for g in games if getattr(g, 'summary', None)]
        if not summaries:
            return

        game_type = summaries[0].get('game_type', 'unknown')
        n = len(summaries)

        # Universal metrics
        lengths = [s['game_length'] for s in summaries]
        outcomes = [g.outcome for g in games[:n]]
        scores_p0 = [s.get('score_p0', 0) for s in summaries]
        scores_p1 = [s.get('score_p1', 0) for s in summaries]

        avg_len = sum(lengths) / n
        std_len = (sum((l - avg_len)**2 for l in lengths) / n) ** 0.5
        avg_turns = sum(s.get('turn_number', 0) for s in summaries) / n
        p0_wins = sum(1 for o in outcomes if o > 0)
        p1_wins = sum(1 for o in outcomes if o < 0)
        draws = sum(1 for o in outcomes if o == 0)
        avg_score = (sum(scores_p0) + sum(scores_p1)) / (2 * n)

        # Tensorboard
        self.writer.add_scalar('GameQuality/AvgLength', avg_len, self.iteration)
        self.writer.add_scalar('GameQuality/StdLength', std_len, self.iteration)
        self.writer.add_scalar('GameQuality/P0WinRate', p0_wins / n, self.iteration)
        self.writer.add_scalar('GameQuality/DrawRate', draws / n, self.iteration)
        self.writer.add_scalar('GameQuality/AvgScore', avg_score, self.iteration)

        # For Dominion: turns are the meaningful length unit
        primary_len = avg_turns if game_type == 'dominion' else avg_len
        primary_len_label = 'turns' if game_type == 'dominion' else 'moves'

        # Store for losses.jsonl
        self._game_quality = {
            'avg_len': round(primary_len, 1),
            'std_len': round(std_len, 1),
            'p0_wr': round(p0_wins / n, 3),
            'draw_rate': round(draws / n, 3),
            'avg_score': round(avg_score, 1),
            'force_rate': round(self._get_force_rate(), 2),
        }

        # Game-specific assessment
        assessment = []
        warnings = []

        if game_type == 'lost_cities':
            assessment, warnings = self._assess_lc(summaries, n, avg_len, std_len, p0_wins, p1_wins, draws)
        elif game_type == 'mandala':
            assessment, warnings = self._assess_mandala(summaries, n, avg_len, std_len, p0_wins, p1_wins, draws)
        elif game_type == 'dominion':
            assessment, warnings = self._assess_dominion(summaries, n, avg_turns, p0_wins, p1_wins, draws)

        # Print summary
        print(f"  Game quality: {primary_len:.0f} avg {primary_len_label} (±{std_len:.1f}), "
              f"P0 {p0_wins}/{n} P1 {p1_wins}/{n} Draw {draws}/{n}, "
              f"Avg score: {avg_score:.1f}")
        for line in assessment:
            print(f"  {line}")
        for w in warnings:
            print(f"  WARNING: {w}")

    def _assess_lc(self, summaries, n, avg_len, std_len, p0w, p1w, draws):
        """Qualitative assessment of Lost Cities gameplay."""
        assessment = []
        warnings = []

        avg_exps = sum(s['num_expeditions'][0] + s['num_expeditions'][1]
                       for s in summaries) / (2 * n)
        total_exp_plays = sum(sum(s['expedition_plays'][0]) + sum(s['expedition_plays'][1])
                             for s in summaries) / (2 * n)
        total_discards = sum(sum(s['color_discards'][0]) + sum(s['color_discards'][1])
                            for s in summaries) / (2 * n)
        discard_draw_rate = sum(sum(s['discard_pile_draws'][0]) + sum(s['discard_pile_draws'][1])
                               for s in summaries) / (2 * n)

        play_rate = total_exp_plays / max(1, total_exp_plays + total_discards)

        total_moves_per_player = total_exp_plays + total_discards
        discard_pct = total_discards / max(1, total_moves_per_player)

        self.writer.add_scalar('GameQuality/LC_AvgExpeditions', avg_exps, self.iteration)
        self.writer.add_scalar('GameQuality/LC_PlayRate', play_rate, self.iteration)
        self.writer.add_scalar('GameQuality/LC_DiscardDrawRate', discard_draw_rate, self.iteration)

        self._game_quality['avg_exps'] = round(avg_exps, 2)
        self._game_quality['play_rate'] = round(play_rate, 3)
        self._game_quality['discard_pct'] = round(discard_pct, 3)
        self._game_quality['disc_draw_rate'] = round(discard_draw_rate, 2)

        assessment.append(f"LC: {avg_exps:.1f} expeditions/player, "
                          f"{play_rate:.0%} play rate, "
                          f"{discard_draw_rate:.1f} discard-pile draws/player")

        if avg_exps < 0.5:
            warnings.append("DEGENERATE: <0.5 expeditions/player — bot isn't building anything")
        elif avg_exps < 1.5:
            warnings.append("WEAK: <1.5 expeditions — bot barely commits to expeditions")
        elif avg_exps >= 3.0:
            assessment.append("HEALTHY: 3+ expeditions — bot is actively building")

        if std_len < 1.0 and n >= 20:
            warnings.append(f"ALL GAMES ~{avg_len:.0f} MOVES — degenerate equilibrium (recycling)")

        if play_rate < 0.15:
            warnings.append(f"Play rate {play_rate:.0%} — bot almost never plays to expeditions")

        if n >= 20 and max(p0w, p1w) / n > 0.70:
            dominant = "P0" if p0w > p1w else "P1"
            warnings.append(f"{dominant} wins {max(p0w,p1w)/n:.0%} — severe player imbalance")

        return assessment, warnings

    def _assess_mandala(self, summaries, n, avg_len, std_len, p0w, p1w, draws):
        """Qualitative assessment of Mandala gameplay."""
        assessment = []
        warnings = []

        avg_mt = sum(sum(s['mountain_plays'][0]) + sum(s['mountain_plays'][1])
                     for s in summaries) / (2 * n)
        avg_fld = sum(sum(s['field_plays'][0]) + sum(s['field_plays'][1])
                      for s in summaries) / (2 * n)
        avg_disc = sum(sum(s['discard_plays'][0]) + sum(s['discard_plays'][1])
                       for s in summaries) / (2 * n)
        total_plays = avg_mt + avg_fld + avg_disc

        if total_plays > 0:
            mt_pct = avg_mt / total_plays
            fld_pct = avg_fld / total_plays
            disc_pct = avg_disc / total_plays

            self.writer.add_scalar('GameQuality/M_MountainRate', mt_pct, self.iteration)
            self.writer.add_scalar('GameQuality/M_FieldRate', fld_pct, self.iteration)
            self.writer.add_scalar('GameQuality/M_DiscardRate', disc_pct, self.iteration)

            self._game_quality['mt_pct'] = round(mt_pct, 3)
            self._game_quality['fld_pct'] = round(fld_pct, 3)
            self._game_quality['disc_pct'] = round(disc_pct, 3)

            assessment.append(f"Mandala: {mt_pct:.0%} mountain, {fld_pct:.0%} field, "
                              f"{disc_pct:.0%} discard")

            if disc_pct > 0.50:
                warnings.append(f"Discard rate {disc_pct:.0%} — bot is throwing away too many cards")
            if mt_pct < 0.10:
                warnings.append(f"Mountain rate {mt_pct:.0%} — bot rarely builds mountains")
            if fld_pct < 0.10:
                warnings.append(f"Field rate {fld_pct:.0%} — bot rarely grows fields")

        avg_score = sum(s.get('score_p0', 0) + s.get('score_p1', 0) for s in summaries) / (2 * n)
        assessment.append(f"Avg score: {avg_score:.1f}")

        if std_len < 2.0 and n >= 20:
            warnings.append(f"Low game length variance (±{std_len:.1f}) — possible degenerate pattern")

        if n >= 20 and max(p0w, p1w) / n > 0.70:
            dominant = "P0" if p0w > p1w else "P1"
            warnings.append(f"{dominant} wins {max(p0w,p1w)/n:.0%} — severe player imbalance")

        return assessment, warnings

    def _assess_dominion(self, summaries, n, avg_turns, p0w, p1w, draws):
        """Qualitative assessment of Dominion gameplay."""
        assessment = []
        warnings = []

        def avg_field(field):
            return sum(s.get(field, (0,0))[0] + s.get(field, (0,0))[1]
                       for s in summaries) / (2 * n)

        avg_buys = avg_field('total_buys')
        avg_provinces = avg_field('province_buys')
        avg_duchies = avg_field('duchy_buys')
        avg_estates = avg_field('estate_buys')
        avg_copper = avg_field('copper_buys')
        avg_treasures = avg_field('treasure_buys')  # Silver + Gold
        avg_action_buys = avg_field('action_buys')
        avg_curses = avg_field('curse_buys')
        avg_actions = avg_field('action_plays')

        # Realized purchasing power: avg coins when entering buy phase
        total_coins = sum(s.get('total_coins_at_buy', (0,0))[0] + s.get('total_coins_at_buy', (0,0))[1]
                         for s in summaries)
        total_entries = sum(s.get('buy_phase_entries', (0,0))[0] + s.get('buy_phase_entries', (0,0))[1]
                           for s in summaries)
        avg_coins_at_buy = total_coins / max(1, total_entries)

        action_rate = avg_actions / max(1, avg_turns)  # avg action plays per turn (both players)

        # Action utilization: % of turns with action in hand where player played one
        total_turns_with_action = sum(
            s.get('turns_with_action_in_hand', (0,0))[0] + s.get('turns_with_action_in_hand', (0,0))[1]
            for s in summaries)
        total_turns_action_played = sum(
            s.get('turns_action_played', (0,0))[0] + s.get('turns_action_played', (0,0))[1]
            for s in summaries)
        action_utilization = total_turns_action_played / max(1, total_turns_with_action)

        # Wasted coins: avg unspent coins per buy phase
        total_wasted = sum(
            s.get('coins_wasted', (0,0))[0] + s.get('coins_wasted', (0,0))[1]
            for s in summaries)
        avg_coins_wasted = total_wasted / max(1, total_entries)

        # Select-phase behavioral tracking
        avg_cards_trashed = sum(
            s.get('cards_trashed', (0,0))[0] + s.get('cards_trashed', (0,0))[1]
            for s in summaries) / (2 * n)
        avg_cellar_discards = sum(
            s.get('cards_discarded_cellar', (0,0))[0] + s.get('cards_discarded_cellar', (0,0))[1]
            for s in summaries) / (2 * n)
        avg_empty_selects = sum(
            s.get('done_selecting_empty', (0,0))[0] + s.get('done_selecting_empty', (0,0))[1]
            for s in summaries) / (2 * n)

        # Per-card buy breakdown
        CARD_NAMES = [
            'Copper', 'Silver', 'Gold', 'Estate', 'Duchy', 'Province', 'Curse',
            'Cellar', 'Moat', 'Chapel', 'Harbinger', 'Merchant', 'Vassal',
            'Village', 'Workshop', 'Bureaucrat', 'Gardens', 'Militia',
            'Moneylender', 'Poacher', 'Remodel', 'Smithy', 'Throne Room',
            'Artisan', 'Bandit', 'Council Room', 'Library', 'Market',
            'Mine', 'Sentry', 'Witch',
        ]
        card_buy_totals = {}
        for s in summaries:
            for card_id_str, counts in s.get('card_buys', {}).items():
                cid = int(card_id_str)
                card_buy_totals[cid] = card_buy_totals.get(cid, 0) + counts[0] + counts[1]
        # Average per player per game, sorted by frequency
        card_buy_avg = {cid: total / (2 * n) for cid, total in card_buy_totals.items()}
        avg_silver = card_buy_avg.get(1, 0)
        avg_gold = card_buy_avg.get(2, 0)
        top_buys = sorted(card_buy_avg.items(), key=lambda x: -x[1])

        self.writer.add_scalar('GameQuality/DOM_AvgBuys', avg_buys, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgProvinces', avg_provinces, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgDuchies', avg_duchies, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgEstates', avg_estates, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCopper', avg_copper, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgTreasures', avg_treasures, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgSilver', avg_silver, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgGold', avg_gold, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgActionBuys', avg_action_buys, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCurses', avg_curses, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_ActionRate', action_rate, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_ActionUtilization', action_utilization, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCoinsWasted', avg_coins_wasted, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgTurns', avg_turns, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCoinsAtBuy', avg_coins_at_buy, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCardsTrashed', avg_cards_trashed, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgCellarDiscards', avg_cellar_discards, self.iteration)
        self.writer.add_scalar('GameQuality/DOM_AvgEmptySelects', avg_empty_selects, self.iteration)
        # Log top action card buys to tensorboard
        for cid, avg in top_buys:
            if cid >= 7:  # action/kingdom cards only
                self.writer.add_scalar(f'CardBuys/{CARD_NAMES[cid]}', avg, self.iteration)

        self._game_quality['avg_buys'] = round(avg_buys, 2)
        self._game_quality['avg_provinces'] = round(avg_provinces, 2)
        self._game_quality['avg_duchies'] = round(avg_duchies, 2)
        self._game_quality['avg_estates'] = round(avg_estates, 2)
        self._game_quality['avg_copper'] = round(avg_copper, 2)
        self._game_quality['avg_treasures'] = round(avg_treasures, 2)
        self._game_quality['avg_silver'] = round(avg_silver, 2)
        self._game_quality['avg_gold'] = round(avg_gold, 2)
        self._game_quality['avg_action_buys'] = round(avg_action_buys, 2)
        self._game_quality['avg_curses'] = round(avg_curses, 2)
        self._game_quality['action_rate'] = round(action_rate, 3)
        self._game_quality['action_utilization'] = round(action_utilization, 3)
        self._game_quality['avg_coins_wasted'] = round(avg_coins_wasted, 2)
        self._game_quality['avg_turns'] = round(avg_turns, 1)
        self._game_quality['avg_coins_at_buy'] = round(avg_coins_at_buy, 2)
        # Raw MCTS Province visit % (before epsilon blend) — tracks organic learning
        mcts_prov_vals = [s.get('mcts_province_pct', 0) for s in summaries if s.get('mcts_province_pct') is not None]
        self._game_quality['mcts_province_pct'] = round(sum(mcts_prov_vals) / max(len(mcts_prov_vals), 1) * 100, 1)
        # Province argmax %: how often Province has the MOST visits when affordable (deterministic preference)
        argmax_vals = [s.get('mcts_province_argmax_pct', 0) for s in summaries if s.get('mcts_province_argmax_pct') is not None]
        if argmax_vals:
            self._game_quality['mcts_province_argmax_pct'] = round(sum(argmax_vals) / len(argmax_vals) * 100, 1)
        self._game_quality['avg_cards_trashed'] = round(avg_cards_trashed, 2)
        self._game_quality['avg_cellar_discards'] = round(avg_cellar_discards, 2)
        self._game_quality['avg_empty_selects'] = round(avg_empty_selects, 2)
        # Top action card buys (kingdom cards only, sorted by frequency)
        action_buys_breakdown = {CARD_NAMES[cid]: round(avg, 2)
                                 for cid, avg in top_buys if cid >= 7 and avg >= 0.05}
        if action_buys_breakdown:
            self._game_quality['top_buys'] = action_buys_breakdown

        # Turn-bucketed buy curve (10 buckets of 5 turns each)
        NUM_BUY_BUCKETS = 10
        bucket_totals = [{} for _ in range(NUM_BUY_BUCKETS)]
        has_bucketed = False
        for s in summaries:
            bucketed = s.get('bucketed_buys', [])
            if not bucketed:
                continue
            has_bucketed = True
            for b_idx, bucket in enumerate(bucketed):
                if b_idx >= NUM_BUY_BUCKETS:
                    break
                for card_id_str, counts in bucket.items():
                    cid = int(card_id_str)
                    bucket_totals[b_idx][cid] = bucket_totals[b_idx].get(cid, 0) + counts[0] + counts[1]
        if has_bucketed:
            buy_curve = []
            for b_idx in range(NUM_BUY_BUCKETS):
                bucket_avg = {CARD_NAMES[cid]: round(total / (2 * n), 3)
                              for cid, total in bucket_totals[b_idx].items()
                              if cid < len(CARD_NAMES) and total > 0}
                buy_curve.append(bucket_avg)
            self._game_quality['buy_curve'] = buy_curve

        # Average turn each card is bought (buy timing)
        turn_sum_totals = {}
        has_turn_sums = False
        for s in summaries:
            for card_id_str, counts in s.get('buy_turn_sum', {}).items():
                has_turn_sums = True
                cid = int(card_id_str)
                turn_sum_totals[cid] = turn_sum_totals.get(cid, 0) + counts[0] + counts[1]
        if has_turn_sums:
            buy_timing = {}
            for cid, turn_sum in turn_sum_totals.items():
                buy_count = card_buy_totals.get(cid, 0)
                if buy_count > 0 and cid < len(CARD_NAMES):
                    buy_timing[CARD_NAMES[cid]] = round(turn_sum / buy_count, 1)
            if buy_timing:
                self._game_quality['buy_timing'] = buy_timing

        assessment.append(f"Dominion: {avg_buys:.1f} buys/player, "
                          f"{avg_provinces:.1f} prov, {avg_duchies:.1f} duchy, {avg_estates:.1f} estate, "
                          f"{avg_copper:.1f} copper, {avg_treasures:.1f} silver/gold, "
                          f"{avg_action_buys:.1f} action, {avg_curses:.1f} curse")
        assessment.append(f"  {avg_turns:.0f} turns, {action_rate:.0%} action play rate, "
                          f"{action_utilization:.0%} action utilization, "
                          f"${avg_coins_at_buy:.1f} avg purchasing power, "
                          f"${avg_coins_wasted:.1f} wasted/buy")
        assessment.append(f"  {avg_cards_trashed:.1f} trashed, {avg_cellar_discards:.1f} cellar discards, "
                          f"{avg_empty_selects:.1f} empty selects/player")
        if top_buys:
            action_cards = [(CARD_NAMES[cid], avg) for cid, avg in top_buys if cid >= 7 and avg >= 0.05]
            if action_cards:
                parts = [f"{name} {avg:.1f}" for name, avg in action_cards[:6]]
                assessment.append(f"  Top kingdom buys: {', '.join(parts)}")

        if avg_provinces < 0.5 and avg_turns > 15:
            warnings.append("LOW PROVINCES: bot rarely buys provinces — no winning strategy")
        if avg_buys < 2.0:
            warnings.append(f"FEW BUYS ({avg_buys:.1f}/player) — bot not buying enough cards")
        if avg_actions < 0.5 and avg_turns > 10:
            warnings.append("NO ACTIONS: bot rarely plays action cards")

        if n >= 20 and max(p0w, p1w) / n > 0.70:
            dominant = "P0" if p0w > p1w else "P1"
            warnings.append(f"{dominant} wins {max(p0w,p1w)/n:.0%} — severe player imbalance")

        return assessment, warnings

    def _select_opponent_checkpoint(self):
        """Select opponent checkpoint, biased toward efficient-buying era if configured."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        checkpoints = sorted(checkpoint_dir.glob('model_iter_*.pt'))
        if len(checkpoints) < 2:
            return None
        # Filter to target iter range if configured
        target_min = self.config.get('opponent_iter_min', None)
        target_max = self.config.get('opponent_iter_max', None)
        if target_min is not None and target_max is not None:
            candidates = []
            for cp in checkpoints:
                try:
                    it = int(cp.stem.split('_')[-1])
                    if target_min <= it <= target_max:
                        candidates.append(cp)
                except ValueError:
                    continue
            if candidates:
                return candidates[np.random.randint(len(candidates))]
        # Fallback: full history except latest
        candidates = checkpoints[:-1]
        if not candidates:
            return None
        return candidates[np.random.randint(len(candidates))]

    def _load_opponent_network(self, path):
        """Load a checkpoint into a fresh MandalaNet for use as opponent."""
        try:
            net_cfg = self.config.get('network', {})
            belief_size = net_cfg.get('belief_size', self.config.get('belief_size', 12))
            opponent = MandalaNet(
                input_channels=net_cfg.get('input_channels', self.config.get('input_channels', 50)),
                num_actions=net_cfg.get('num_actions', self.config.get('num_actions', 30)),
                num_res_blocks=net_cfg.get('num_res_blocks', self.config.get('num_res_blocks', 10)),
                channels=net_cfg.get('channels', self.config.get('channels', 128)),
                belief_size=belief_size,
                phase_aware_policy=net_cfg.get('phase_aware_policy', False),
                factored_policy=net_cfg.get('factored_policy', False),
            ).to(self.device)
            data = torch.load(path, map_location=self.device, weights_only=False)
            # Handle architecture mismatches — old checkpoints may have different sizes
            saved_state = data['model_state_dict']
            model_state = opponent.state_dict()
            for key in ['conv_input.weight', 'fc_belief.weight', 'fc_belief.bias']:
                if key in saved_state and key in model_state:
                    if saved_state[key].shape != model_state[key].shape:
                        raise RuntimeError(f"Shape mismatch for {key}: saved {saved_state[key].shape} vs expected {model_state[key].shape}")
            opponent.load_state_dict(saved_state)
            opponent.eval()
            return opponent
        except Exception as e:
            print(f"Warning: Failed to load opponent {path}: {e}")
            return None

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
            'warmup_target': self._warmup_target,
            'model_state_dict': self._unwrapped_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if suffix:
            # Game-level checkpoint: lightweight only
            torch.save(checkpoint, checkpoint_dir / f'model_latest{suffix}.pt')
        else:
            # Atomic write (temp file + rename) to prevent 0-byte corruption on crash
            tmp_path = checkpoint_dir / 'model_latest.pt.tmp'
            torch.save(checkpoint, tmp_path)
            tmp_path.rename(checkpoint_dir / 'model_latest.pt')

            # Save replay buffer separately (not in checkpoint — avoids OOM from 3x memory spike)
            try:
                buf_tmp = checkpoint_dir / 'buffer_latest.pkl.tmp'
                self.replay_buffer.save(buf_tmp)
                buf_final = checkpoint_dir / 'buffer_latest.pkl'
                buf_tmp.rename(buf_final)
                buf_size_mb = buf_final.stat().st_size / 1e6
                if buf_size_mb < 1:
                    print(f"  WARNING: buffer file suspiciously small ({buf_size_mb:.1f} MB)")
            except Exception as e:
                print(f"  WARNING: buffer save failed ({e}), training continues")

            # Save periodic iteration checkpoint (lightweight, no replay buffer)
            if self.iteration % self.config.get('checkpoint_frequency', 10) == 0:
                torch.save(checkpoint, checkpoint_dir / f'model_iter_{self.iteration}.pt')

            # Archive every Nth checkpoint permanently (never pruned)
            archive_freq = self.config.get('archive_checkpoint_frequency', 100)
            if archive_freq > 0 and self.iteration % archive_freq == 0:
                archive_dir = checkpoint_dir / 'archive'
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f'model_iter_{self.iteration}.pt'
                torch.save(checkpoint, archive_path)
                print(f"  Archived checkpoint: {archive_path}")

        status = f"iteration {self.iteration}"
        if self.games_in_current_iteration > 0:
            status += f", game {self.games_in_current_iteration}/{self.config.get('games_per_iteration', 100)}"
        print(f"Saved checkpoint to {checkpoint_dir} ({status}, buffer: {len(self.replay_buffer)})")

    def _export_deploy_checkpoint(self):
        """Export lightweight checkpoint to data/deploy/ for Railway."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'data/checkpoints'))
        game_name = 'lost_cities' if self.config.get('network', {}).get('num_actions', 30) != 30 else 'mandala'
        deploy_dir = checkpoint_dir.parent / 'deploy' / game_name
        deploy_dir.mkdir(parents=True, exist_ok=True)

        lightweight = {
            'model_state_dict': self._unwrapped_network.state_dict(),
            'iteration': self.iteration,
            'total_games': self.total_games,
        }
        torch.save(lightweight, deploy_dir / 'model.pt')
        print(f"[DEPLOY] Exported deploy checkpoint: iter {self.iteration} -> {deploy_dir / 'model.pt'}")

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

        # 2. Prune old iteration checkpoints (only if enabled in config)
        if not self.config.get('prune_old_checkpoints', False):
            return
        iter_files = sorted(checkpoint_dir.glob('model_iter_*.pt'), key=lambda f: int(f.stem.split('_')[2]))
        if len(iter_files) > 40:
            keep = set()
            for f in iter_files:
                i = int(f.stem.split('_')[2])
                if i % 50 == 0:
                    keep.add(f)
            for f in iter_files[-30:]:
                keep.add(f)
            to_delete = [f for f in iter_files if f not in keep]
            for f in to_delete:
                f.unlink()
            if to_delete:
                print(f"Pruned {len(to_delete)} old iteration checkpoints (kept {len(keep)})")

    def load_checkpoint(self, filepath: Path):
        """Load training checkpoint and replay buffer."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Handle policy head size migration (108 → 150 actions)
        saved_state = checkpoint['model_state_dict']
        model_state = self._unwrapped_network.state_dict()
        policy_migrated = False
        if 'fc_policy.weight' in saved_state and 'fc_policy.weight' in model_state:
            old_out = saved_state['fc_policy.weight'].shape[0]
            new_out = model_state['fc_policy.weight'].shape[0]
            if old_out != new_out:
                policy_migrated = True
                print(f"Migrating policy head: {old_out} → {new_out} actions")
                old_w = saved_state['fc_policy.weight']
                old_b = saved_state['fc_policy.bias']
                new_w = torch.zeros_like(model_state['fc_policy.weight'])
                new_b = torch.zeros_like(model_state['fc_policy.bias'])
                # Copy BUILD_MOUNTAIN (0-11) and GROW_FIELD (12-95) as-is
                new_w[:96] = old_w[:96]
                new_b[:96] = old_b[:96]
                # Remap old DISCARD (96+color) → new DISCARD (96+color*8+count)
                # Copy old per-color weights to all count variants
                for c in range(6):
                    if 96 + c < old_out:
                        for cnt in range(8):
                            new_w[96 + c * 8 + cnt] = old_w[96 + c]
                            new_b[96 + c * 8 + cnt] = old_b[96 + c]
                # Remap old CLAIM (102+color) → new CLAIM (144+color)
                for c in range(6):
                    if 102 + c < old_out:
                        new_w[144 + c] = old_w[102 + c]
                        new_b[144 + c] = old_b[102 + c]
                saved_state['fc_policy.weight'] = new_w
                saved_state['fc_policy.bias'] = new_b
        # Handle input channel migration (e.g., 156 → 218)
        if 'conv_input.weight' in saved_state and 'conv_input.weight' in model_state:
            old_in = saved_state['conv_input.weight'].shape[1]
            new_in = model_state['conv_input.weight'].shape[1]
            if old_in != new_in:
                policy_migrated = True  # Also skip optimizer state
                print(f"Migrating input channels: {old_in} → {new_in}")
                new_w = torch.zeros_like(model_state['conv_input.weight'])
                new_w[:, :old_in, :, :] = saved_state['conv_input.weight']
                saved_state['conv_input.weight'] = new_w
        # Handle belief head migration (e.g., 12 → 31)
        if 'fc_belief.weight' in saved_state and 'fc_belief.weight' in model_state:
            old_out = saved_state['fc_belief.weight'].shape[0]
            new_out = model_state['fc_belief.weight'].shape[0]
            if old_out != new_out:
                policy_migrated = True  # Skip optimizer state
                print(f"Migrating belief head: {old_out} → {new_out} (reinitializing)")
                saved_state['fc_belief.weight'] = model_state['fc_belief.weight']
                saved_state['fc_belief.bias'] = model_state['fc_belief.bias']
        self._unwrapped_network.load_state_dict(saved_state)
        if policy_migrated:
            print("Optimizer state skipped (policy head migrated, sizes incompatible)")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (ValueError, KeyError, RuntimeError):
                print("Optimizer state incompatible, starting fresh")
        self.iteration = checkpoint['iteration']
        # Set LR based on absolute iteration (restart-resilient)
        lr = self._get_lr_for_iteration(self.iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        print(f"LR set to {lr:.6f} for iteration {self.iteration}")
        self.total_games = checkpoint['total_games']

        # NEW: Restore game progress within iteration
        self.games_in_current_iteration = checkpoint.get('games_in_current_iteration', 0)
        self._warmup_target = checkpoint.get('warmup_target', 0)
        if self._warmup_target > 0:
            print(f"[WARMUP] Resuming warmup: target {self._warmup_target} examples")

        # Try loading buffer from separate file first, then legacy embedded buffer
        buffer_path = Path(filepath).parent / 'buffer_latest.pkl'
        if 'replay_buffer' in checkpoint:
            self.replay_buffer.load_data(checkpoint['replay_buffer'])
            print(f"Restored replay buffer with {len(self.replay_buffer)} examples (legacy)")
        elif buffer_path.exists():
            buf_size_mb = buffer_path.stat().st_size / 1e6
            print(f"Loading buffer from {buffer_path.name} ({buf_size_mb:.0f} MB)...")
            try:
                self.replay_buffer.load(buffer_path)
                print(f"Restored replay buffer with {len(self.replay_buffer)} examples")
            except Exception as e:
                print(f"WARNING: Buffer load failed ({e}), starting empty")
        else:
            print("Replay buffer starts empty (rebuilds from self-play)")
        if len(self.replay_buffer) == 0 and self.iteration > 0:
            print(f"WARNING: Buffer empty at iter {self.iteration} — network vulnerable to overfitting")

        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from iteration {self.iteration}, total games: {self.total_games}")
        if self.games_in_current_iteration > 0:
            print(f"Mid-iteration: will continue from game {self.games_in_current_iteration + 1}")

    def _start_async_eval(self):
        """No-op: eval is handled by the standalone eval_daemon.py process."""
        print(f"[EVAL] Checkpoint saved — eval_daemon will pick up iter {self.iteration}")

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

        net_cfg = self.config.get('network', {})
        current_model = MandalaNet(
            input_channels=net_cfg.get('input_channels', self.config.get('input_channels', 50)),
            num_actions=net_cfg.get('num_actions', self.config.get('num_actions', 30)),
            num_res_blocks=net_cfg.get('num_res_blocks', self.config.get('num_res_blocks', 10)),
            channels=net_cfg.get('channels', self.config.get('channels', 128)),
            belief_size=net_cfg.get('belief_size', self.config.get('belief_size', 12)),
            phase_aware_policy=net_cfg.get('phase_aware_policy', False),
            factored_policy=net_cfg.get('factored_policy', False),
        ).to(self.device)
        current_model.load_state_dict(self._unwrapped_network.state_dict())

        prev_model = MandalaNet(
            input_channels=net_cfg.get('input_channels', self.config.get('input_channels', 50)),
            num_actions=net_cfg.get('num_actions', self.config.get('num_actions', 30)),
            num_res_blocks=net_cfg.get('num_res_blocks', self.config.get('num_res_blocks', 10)),
            channels=net_cfg.get('channels', self.config.get('channels', 128)),
            belief_size=net_cfg.get('belief_size', self.config.get('belief_size', 12)),
            phase_aware_policy=net_cfg.get('phase_aware_policy', False),
            factored_policy=net_cfg.get('factored_policy', False),
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
