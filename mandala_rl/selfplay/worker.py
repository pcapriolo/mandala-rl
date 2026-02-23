"""
Self-play worker for generating training games using C++ MCTS engine.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import mcts_cpp
from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from ..network.model import MandalaNet


class SelfPlayGame:
    """Represents a completed self-play game."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.current_players: List[int] = []
        self.belief_labels: List[np.ndarray] = []  # Ground-truth opponent hand/cup
        self.outcome: float = 0.0  # Final game outcome from player 0's perspective
        self.score_p0: int = 0     # Player 0's raw score
        self.score_p1: int = 0     # Player 1's raw score
        self.summary: dict = {}   # Game quality stats from C++ engine

    def __len__(self):
        return len(self.states)


class SelfPlayWorker:
    """
    Worker for generating self-play games.

    Uses C++ MCTS engine with batched neural network inference.
    """

    def __init__(
        self,
        game,
        network: MandalaNet,
        mcts_simulations: int = 800,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "mps",
        leaves_per_game: int = 1
    ):
        self.game = game
        self.network = network.to(device)
        self.network.eval()
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device
        self.leaves_per_game = leaves_per_game

        # Detect game type for C++ engine
        self._game_type = "mandala" if network.num_actions in (108, 150) else "lost_cities"

        # Mixed precision for CUDA inference
        self.use_amp = device == 'cuda'

    def generate_games(self, num_games: int) -> List[SelfPlayGame]:
        """Generate multiple self-play games."""
        return self.play_games_batched(num_games)

    def get_training_examples(self, game: SelfPlayGame) -> List[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]]:
        """
        Convert game to training examples.

        Each example is (state, policy, value, score, belief) where:
        - value: final outcome from current player's perspective
        - score: normalized score margin from current player's perspective
        - belief: 12-element binary vector (opp hand colors + opp cup colors)
        """
        examples = []
        outcome = game.outcome

        # Normalize score margin to ~[-1, 1]
        score_margin = game.score_p0 - game.score_p1
        max_margin = 60.0 if self._game_type == 'mandala' else 200.0

        has_beliefs = len(game.belief_labels) == len(game.states)

        for i, (state, policy, player) in enumerate(zip(game.states, game.policies, game.current_players)):
            value = outcome if player == 0 else -outcome
            score = score_margin / max_margin if player == 0 else -score_margin / max_margin

            # Policy target pruning: zero out actions with <2% visits
            policy = policy.copy()
            policy[policy < 0.02] = 0.0
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum

            belief = game.belief_labels[i] if has_beliefs else np.zeros(12, dtype=np.float32)
            examples.append((state, policy, value, score, belief))

        return examples

    def play_games_batched(self, num_games: int, batch_size: int = 8,
                           save_dir: Optional[Path] = None, iteration: int = 0,
                           save_replay_freq: int = 10,
                           on_game_complete=None) -> List[SelfPlayGame]:
        """Play multiple games with C++ MCTS engine and batched NN inference."""
        mgr = mcts_cpp.BatchedMCTS(
            game_type=self._game_type,
            seed=int(np.random.randint(0, 2**31)),
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            leaves_per_game=self.leaves_per_game,
        )
        mgr.init_games(num_games)

        self.network.eval()
        completed = []

        while not mgr.all_done():
            # Root expansion: get canonical state tensors
            root_tensors = mgr.begin_move()
            if len(root_tensors) == 0:
                break

            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                batch = torch.from_numpy(np.stack(root_tensors)).to(self.device)
                logits = self.network(batch)[0]  # policy logits only
                policies = F.softmax(logits, dim=1).cpu().numpy()
            mgr.set_root_policies(policies)

            # Run MCTS simulations (each step collects leaves_per_game leaves per game)
            sim_steps = max(1, self.mcts_simulations // self.leaves_per_game)
            for _ in range(sim_steps):
                leaf_tensors = mgr.simulate_step()
                if len(leaf_tensors) > 0:
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                        batch = torch.from_numpy(np.stack(leaf_tensors)).to(self.device)
                        logits, vals, _scores, _beliefs = self.network(batch)
                        pols = F.softmax(logits, dim=1).cpu().numpy()
                        vals_np = vals.cpu().numpy()[:, 0]
                    mgr.apply_nn_results(pols, vals_np)

            # Select actions, advance games
            done_indices = mgr.finish_move()
            for idx in done_indices:
                game_data = mgr.get_game_data(idx)
                record = SelfPlayGame()
                record.states = [np.array(s) for s in game_data[0]]
                record.policies = [np.array(p) for p in game_data[1]]
                record.current_players = [int(p) for p in game_data[2]]
                record.outcome = float(game_data[3])
                record.score_p0 = int(game_data[4])
                record.score_p1 = int(game_data[5])
                if len(game_data) > 6:
                    record.belief_labels = [np.array(b, dtype=np.float32) for b in game_data[6]]
                record.summary = dict(mgr.get_game_summary(idx))
                completed.append(record)

                # Save replay for dashboard monitoring
                game_num = len(completed)
                if save_dir and save_replay_freq > 0 and game_num % save_replay_freq == 0:
                    self._save_replay(save_dir, record, iteration)

                if on_game_complete:
                    on_game_complete(idx, record)

        return completed

    @staticmethod
    def _save_replay(save_dir: Path, game: 'SelfPlayGame', iteration: int):
        """Save minimal replay JSON for dashboard monitoring."""
        save_dir.mkdir(parents=True, exist_ok=True)
        game_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        winner = 0 if game.outcome > 0 else (1 if game.outcome < 0 else None)
        replay = {
            'game_id': game_id,
            'metadata': {'iteration': iteration},
            'moves': [{'move_num': i, 'player': int(p)}
                       for i, p in enumerate(game.current_players)],
            'final_score': [game.summary.get('score_p0'), game.summary.get('score_p1')]
                           if game.summary else None,
            'winner': winner,
            'summary': {k: v for k, v in game.summary.items()
                        if k not in ('outcome',)} if game.summary else None,
        }
        filepath = save_dir / f"game_{game_id}.json"
        with open(filepath, 'w') as f:
            json.dump(replay, f)
