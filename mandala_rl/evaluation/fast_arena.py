"""
Fast evaluation arena using C++ BatchedMCTS engine with GPU inference.

Replaces the Python MCTS Arena for ~50-100x faster Elo evaluation.
All eval games run in parallel through the C++ engine with batched NN calls.
"""
import numpy as np
import torch
import torch.nn.functional as F
import mcts_cpp
from typing import Optional, List, Tuple


class FastArena:
    """
    Arena using C++ BatchedMCTS for fast model evaluation.

    Plays all games in parallel. For each move, determines which model
    should evaluate each game based on the current player and model assignment.
    """

    def __init__(
        self,
        game_type: str,
        mcts_simulations: int = 400,
        c_puct: float = 1.0,
        device: str = "cuda"
    ):
        self.game_type = game_type
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device
        self.use_amp = device == 'cuda'

    def play_match(
        self,
        model1,
        model2,
        num_games: int,
        seed: Optional[int] = None
    ) -> dict:
        """Play a match between two models. Delegates to play_tournament."""
        results = self.play_tournament(
            current_model=model1,
            opponents=[(model2, "_opponent")],
            games_per_opponent=num_games,
            seed=seed,
        )
        r = results["_opponent"]
        return {
            'model1_wins': r['current_wins'],
            'model2_wins': r['opponent_wins'],
            'draws': r['draws'],
            'total_games': r['total_games'],
        }

    def play_tournament(
        self,
        current_model,
        opponents: List[Tuple],
        games_per_opponent: int = 5,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Run all tournament games in one BatchedMCTS session.

        current_model: the checkpoint being evaluated
        opponents: list of (model, name) tuples
        games_per_opponent: games per opponent (alternating colors)

        Game layout: game_idx = opp_idx * games_per_opponent + repeat_idx
        Within each group: even repeats → current=P0, odd → opponent=P0

        Returns: {name: {current_wins, opponent_wins, draws, total_games}}
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        num_opponents = len(opponents)
        total_games = num_opponents * games_per_opponent

        # models[0] = current, models[1..N] = opponents
        models = [current_model] + [opp[0] for opp in opponents]
        for m in models:
            m.eval()

        # Map each game_idx to its opponent model index (1-based in models list)
        game_to_opp_model = {}
        for opp_idx in range(num_opponents):
            for repeat in range(games_per_opponent):
                game_idx = opp_idx * games_per_opponent + repeat
                game_to_opp_model[game_idx] = opp_idx + 1  # 1-based

        mgr = mcts_cpp.BatchedMCTS(
            game_type=self.game_type,
            seed=seed if seed is not None else int(np.random.randint(0, 2**31)),
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            temperature=0.0,
            temperature_threshold=0,
            leaves_per_game=1,
        )
        mgr.init_games(total_games)

        while not mgr.all_done():
            root_tensors = mgr.begin_move()
            if len(root_tensors) == 0:
                break

            active_players = mgr.get_active_players()
            active_indices = mgr.get_active_game_indices()

            model_map = self._build_tournament_model_map(
                active_indices, active_players, game_to_opp_model, games_per_opponent
            )
            policies = self._eval_multi_model(
                root_tensors, model_map, models, policy_only=True
            )
            mgr.set_root_policies(policies)

            # Build game_idx → model_index lookup for leaf routing
            game_to_model = {}
            for batch_pos, game_idx in enumerate(active_indices):
                game_to_model[game_idx] = model_map[batch_pos]

            for _ in range(self.mcts_simulations):
                leaf_tensors = mgr.simulate_step()
                if len(leaf_tensors) == 0:
                    continue
                pending = mgr.get_pending_game_indices()
                leaf_model_map = [game_to_model[gi] for gi in pending]
                pols, vals = self._eval_multi_model(
                    leaf_tensors, leaf_model_map, models, policy_only=False
                )
                mgr.apply_nn_results(pols, vals)

            mgr.finish_move()

        # Collect results per opponent — both aggregated and per-game
        results = {}
        for opp_idx, (_, name) in enumerate(opponents):
            r = {'current_wins': 0, 'opponent_wins': 0, 'draws': 0,
                 'total_games': games_per_opponent, 'game_results': []}
            for repeat in range(games_per_opponent):
                game_idx = opp_idx * games_per_opponent + repeat
                _, _, _, outcome = mgr.get_game_data(game_idx)
                outcome = float(outcome)  # From player 0's perspective
                current_is_p0 = (game_idx % 2 == 0)
                if outcome > 0:
                    winner = 'current' if current_is_p0 else 'opponent'
                elif outcome < 0:
                    winner = 'opponent' if current_is_p0 else 'current'
                else:
                    winner = 'draw'
                if winner == 'current':
                    r['current_wins'] += 1
                elif winner == 'opponent':
                    r['opponent_wins'] += 1
                else:
                    r['draws'] += 1
                r['game_results'].append(winner)
            results[name] = r

        return results

    def _build_tournament_model_map(self, active_indices, active_players,
                                     game_to_opp_model, games_per_opponent):
        """Map each active game to its model index for this move.

        model 0 = current_model, model N = Nth opponent.
        Within each opponent group, even repeats have current as P0.
        """
        model_map = []
        for game_idx, player in zip(active_indices, active_players):
            current_is_p0 = (game_idx % 2 == 0)
            if current_is_p0:
                model_map.append(0 if player == 0 else game_to_opp_model[game_idx])
            else:
                model_map.append(game_to_opp_model[game_idx] if player == 0 else 0)
        return model_map

    def _eval_multi_model(self, tensors, model_map, models, policy_only=False):
        """Group tensors by model, evaluate each batch, recombine in original order."""
        n = len(tensors)
        if n == 0:
            if policy_only:
                return np.zeros((0, 0), dtype=np.float32)
            return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.float32)

        stacked = np.stack(tensors)

        # Group indices by model
        groups = {}
        for i, m_idx in enumerate(model_map):
            groups.setdefault(m_idx, []).append(i)

        all_policies = None
        all_values = np.empty(n, dtype=np.float32) if not policy_only else None

        for m_idx, indices in groups.items():
            batch = torch.from_numpy(stacked[indices]).to(self.device)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, vals = models[m_idx](batch)
                pols = F.softmax(logits, dim=1).cpu().numpy()
                vals_np = vals.cpu().numpy()[:, 0] if not policy_only else None

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
