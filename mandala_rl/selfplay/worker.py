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
        explore_epsilon: float = 0.0,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "mps",
        leaves_per_game: int = 1,
        action_explore_boost: float = 0.0,
        action_buy_force_rate: float = 0.0,
        action_play_force_rate: float = 0.0,
        max_action_cards: int = 10,
        big_money_force_rate: float = 0.0,
        forced_kingdom_cards: list = None,
        disabled_basic_supply: list = None,
        province_supply: int = 8,
        draw_penalty: float = 0.0,
        drop_draws: bool = False,
        max_turns: int = 0,
        mcts_leaf_eval_source: str = "score",
        opponent_disabled_supply: list = None,
    ):
        self.game = game
        self.network = network.to(device)
        self.network.eval()
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.explore_epsilon = explore_epsilon
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device
        self.leaves_per_game = leaves_per_game
        self.action_explore_boost = action_explore_boost
        self.action_buy_force_rate = action_buy_force_rate
        self.action_play_force_rate = action_play_force_rate
        self.max_action_cards = max_action_cards
        self.big_money_force_rate = big_money_force_rate
        self.forced_kingdom_cards = forced_kingdom_cards or []
        self.disabled_basic_supply = disabled_basic_supply or []
        self.province_supply = province_supply
        self.draw_penalty = draw_penalty
        self.drop_draws = drop_draws
        self.max_turns = max_turns
        if mcts_leaf_eval_source not in ("score", "value"):
            raise ValueError(f"mcts_leaf_eval_source must be 'score' or 'value', got {mcts_leaf_eval_source!r}")
        self.mcts_leaf_eval_source = mcts_leaf_eval_source
        # DEVLOG #170: card IDs the reference (opponent) network's policy must
        # treat as illegal during inference. Applied as -inf on the BUY action
        # logits before softmax, so the reference structurally never expands
        # those branches in MCTS. Supply stays globally available — only the
        # reference's selection is constrained. Current agent is unaffected.
        self.opponent_disabled_supply = opponent_disabled_supply or []

        # Detect game type for C++ engine
        if network.num_actions in (108, 150):
            self._game_type = "mandala"
        elif network.num_actions == 131:
            self._game_type = "dominion"
        else:
            self._game_type = "lost_cities"

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

        if self.drop_draws and outcome == 0.0:
            return []

        # Normalize score margin to ~[-1, 1]
        score_margin = game.score_p0 - game.score_p1
        if self._game_type == 'mandala':
            max_margin = 60.0
        elif self._game_type == 'dominion':
            max_margin = 30.0
        else:
            max_margin = 200.0

        has_beliefs = len(game.belief_labels) == len(game.states)

        for i, (state, policy, player) in enumerate(zip(game.states, game.policies, game.current_players)):
            score = score_margin / max_margin if player == 0 else -score_margin / max_margin
            # DEVLOG #123: Phase 1 uses binary win/loss (not score margin).
            # Score margin gave tiny ±0.07 targets (Province = Estate at same VP) → value head blind.
            # Binary outcome: +1 win / -1 loss / 0 draw → strong gradient for Province-buying strategies.
            value = outcome if player == 0 else -outcome
            # Draw penalty: teach both players that draws are slightly bad,
            # preventing the "Silver-spam → safe draw" equilibrium (training targets only, not MCTS).
            if value == 0.0 and self.draw_penalty > 0:
                value = -self.draw_penalty

            # Policy target pruning: zero out actions with <2% visits
            policy = policy.copy()
            policy[policy < 0.02] = 0.0
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum

            if has_beliefs:
                belief = game.belief_labels[i]
            else:
                bl_size = len(game.belief_labels[0]) if game.belief_labels else 12
                belief = np.zeros(bl_size, dtype=np.float32)
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
            explore_epsilon=self.explore_epsilon,
            leaves_per_game=self.leaves_per_game,
            action_explore_boost=self.action_explore_boost,
            action_buy_force_rate=self.action_buy_force_rate,
            action_play_force_rate=self.action_play_force_rate,
            max_action_cards=self.max_action_cards,
            big_money_force_rate=self.big_money_force_rate,
            forced_kingdom_cards=self.forced_kingdom_cards,
            disabled_basic_supply=self.disabled_basic_supply,
            province_supply=self.province_supply,
            max_turns=self.max_turns,
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
                        logits, vals, scores, _beliefs = self.network(batch)
                        pols = F.softmax(logits, dim=1).cpu().numpy()
                        if self.mcts_leaf_eval_source == "value":
                            # Value head trained on binary ±1 outcomes — wider leaf eval range
                            # than score head (narrow VP-margin targets). See DEVLOG #163.
                            vals_np = vals.squeeze(-1).cpu().numpy()
                        else:
                            # score: score head (VP margin), bounded by tanh
                            vals_np = torch.tanh(scores.squeeze(-1)).cpu().numpy()
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

    def play_games_vs_opponent(self, num_games: int, opponent_network,
                                batch_size: int = 8,
                                save_dir: Optional[Path] = None,
                                iteration: int = 0,
                                save_replay_freq: int = 10,
                                on_game_complete=None) -> List[SelfPlayGame]:
        """Play games with current network vs an opponent network.

        Game layout: even game_idx -> current=P0, odd -> current=P1.
        Both players' positions are recorded for training (value targets are
        always correct; diverse policy targets aid exploration).
        """
        mgr = mcts_cpp.BatchedMCTS(
            game_type=self._game_type,
            seed=int(np.random.randint(0, 2**31)),
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            explore_epsilon=self.explore_epsilon,
            leaves_per_game=self.leaves_per_game,
            action_explore_boost=self.action_explore_boost,
            action_buy_force_rate=self.action_buy_force_rate,
            action_play_force_rate=self.action_play_force_rate,
            max_action_cards=self.max_action_cards,
            big_money_force_rate=self.big_money_force_rate,
            forced_kingdom_cards=self.forced_kingdom_cards,
            disabled_basic_supply=self.disabled_basic_supply,
            province_supply=self.province_supply,
            max_turns=self.max_turns,
        )
        mgr.init_games(num_games)

        self.network.eval()
        opponent_network.eval()
        models = [self.network, opponent_network]
        completed = []

        while not mgr.all_done():
            root_tensors = mgr.begin_move()
            if len(root_tensors) == 0:
                break

            active_players = mgr.get_active_players()
            active_indices = mgr.get_active_game_indices()
            model_map = self._build_two_model_map(active_indices, active_players)

            policies = self._eval_two_models(
                root_tensors, model_map, models, policy_only=True
            )
            mgr.set_root_policies(policies)

            # Build game_idx → model_index lookup for leaf routing
            game_to_model = {}
            for batch_pos, game_idx in enumerate(active_indices):
                game_to_model[game_idx] = model_map[batch_pos]

            sim_steps = max(1, self.mcts_simulations // self.leaves_per_game)
            for _ in range(sim_steps):
                leaf_tensors = mgr.simulate_step()
                if len(leaf_tensors) == 0:
                    continue
                pending = mgr.get_pending_game_indices()
                leaf_model_map = [game_to_model[gi] for gi in pending]
                pols, vals = self._eval_two_models(
                    leaf_tensors, leaf_model_map, models, policy_only=False
                )
                mgr.apply_nn_results(pols, vals)

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

                game_num = len(completed)
                if save_dir and save_replay_freq > 0 and game_num % save_replay_freq == 0:
                    self._save_replay(save_dir, record, iteration)

                if on_game_complete:
                    on_game_complete(idx, record)

        return completed

    @staticmethod
    def _build_two_model_map(active_indices, active_players):
        """Map each active game to model index: 0=current, 1=opponent.

        Even game_idx → current is P0, odd → current is P1.
        """
        model_map = []
        for game_idx, player in zip(active_indices, active_players):
            current_is_p0 = (game_idx % 2 == 0)
            if current_is_p0:
                model_map.append(0 if player == 0 else 1)
            else:
                model_map.append(1 if player == 0 else 0)
        return model_map

    def _eval_two_models(self, tensors, model_map, models, policy_only=False):
        """Group tensors by model, evaluate each batch, recombine in original order."""
        n = len(tensors)
        if n == 0:
            if policy_only:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)

        stacked = np.stack(tensors)

        groups = {}
        for i, m_idx in enumerate(model_map):
            groups.setdefault(m_idx, []).append(i)

        all_policies = None
        all_values = np.empty(n, dtype=np.float32) if not policy_only else None

        for m_idx, indices in groups.items():
            batch = torch.from_numpy(stacked[indices]).to(self.device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, vals, scores, *_ = models[m_idx](batch)
                # DEVLOG #170: per-reference policy masking. When the opponent
                # (m_idx == 1) has opponent_disabled_supply set, force -inf on
                # the BUY action logits for those card IDs before softmax so
                # MCTS never expands those branches on the reference side.
                # Current agent (m_idx == 0) is untouched.
                if m_idx == 1 and self.opponent_disabled_supply:
                    # BUY[card_id] action index = 34 + card_id
                    # (dominion/game/state.py: BUY_OFFSET = 34)
                    buy_idx = [34 + cid for cid in self.opponent_disabled_supply]
                    logits[:, buy_idx] = float('-inf')
                pols = F.softmax(logits, dim=1).cpu().numpy()
                if policy_only:
                    vals_np = None
                elif self.mcts_leaf_eval_source == "value":
                    vals_np = vals.squeeze(-1).cpu().numpy()
                else:
                    vals_np = torch.tanh(scores.squeeze(-1)).cpu().numpy()

            if all_policies is None:
                all_policies = np.empty((n, pols.shape[1]), dtype=np.float32)

            for out_i, src_i in enumerate(indices):
                all_policies[src_i] = pols[out_i]
                if all_values is not None:
                    all_values[src_i] = vals_np[out_i]

        if all_policies is None:
            if policy_only:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)

        if policy_only:
            return all_policies
        return all_policies, all_values

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
