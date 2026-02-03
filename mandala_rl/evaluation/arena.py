"""
Arena for evaluating models against each other.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..game.engine import MandalaGame
from ..network.model import MandalaNet
from ..mcts.search import MCTS


class Arena:
    """
    Arena for playing matches between models.

    Uses deterministic play (temperature=0) for evaluation.
    """

    def __init__(
        self,
        game: MandalaGame,
        mcts_simulations: int = 800,
        c_puct: float = 1.0,
        device: str = "mps"
    ):
        """
        Args:
            game: Game engine
            mcts_simulations: Number of MCTS simulations per move
            c_puct: MCTS exploration constant
            device: PyTorch device
        """
        self.game = game
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.device = device

    def play_match(
        self,
        model1: MandalaNet,
        model2: MandalaNet,
        num_games: int,
        seed: Optional[int] = None
    ) -> dict:
        """
        Play a match between two models.

        Args:
            model1: First model
            model2: Second model
            num_games: Number of games to play (should be even for fairness)
            seed: Random seed for reproducibility

        Returns:
            dict with keys:
                - 'model1_wins': Number of wins for model1
                - 'model2_wins': Number of wins for model2
                - 'draws': Number of draws
                - 'total_games': Total games played
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        model1.eval()
        model2.eval()

        # Create MCTS instances
        mcts1 = MCTS(
            game=self.game,
            network=model1.predict,
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            temperature=0.0  # Deterministic for evaluation
        )

        mcts2 = MCTS(
            game=self.game,
            network=model2.predict,
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            temperature=0.0
        )

        results = {
            'model1_wins': 0,
            'model2_wins': 0,
            'draws': 0,
            'total_games': num_games
        }

        # Play games (alternate who plays first)
        for game_num in tqdm(range(num_games), desc="Arena match"):
            # Alternate starting player
            if game_num % 2 == 0:
                # Model1 plays as player 0
                winner = self._play_game(mcts1, mcts2)
                if winner == 0:
                    results['model1_wins'] += 1
                elif winner == 1:
                    results['model2_wins'] += 1
                else:
                    results['draws'] += 1
            else:
                # Model2 plays as player 0
                winner = self._play_game(mcts2, mcts1)
                if winner == 0:
                    results['model2_wins'] += 1
                elif winner == 1:
                    results['model1_wins'] += 1
                else:
                    results['draws'] += 1

        return results

    def _play_game(self, mcts_player0: MCTS, mcts_player1: MCTS) -> Optional[int]:
        """
        Play a single game between two MCTS players.

        Args:
            mcts_player0: MCTS for player 0
            mcts_player1: MCTS for player 1

        Returns:
            Winner: 0, 1, or None for draw
        """
        state = self.game.get_initial_state()
        mcts_players = [mcts_player0, mcts_player1]

        while not self.game.is_terminal(state):
            current_mcts = mcts_players[state.current_player]

            # Get action (deterministic)
            canonical_state = state.get_canonical_form()
            action_probs, _ = current_mcts.get_action_prob(
                canonical_state,
                temperature=0.0,
                add_noise=False  # No exploration noise in evaluation
            )

            # Select best action
            action = np.argmax(action_probs)

            # Apply action
            state = self.game.get_next_state(state, action)

        # Determine winner
        reward_p0 = self.game.get_reward(state, player=0)
        if reward_p0 > 0:
            return 0
        elif reward_p0 < 0:
            return 1
        else:
            return None

    def evaluate_checkpoint(
        self,
        new_model: MandalaNet,
        baseline_model: MandalaNet,
        num_games: int = 100,
        win_threshold: float = 0.55
    ) -> bool:
        """
        Evaluate if new model is stronger than baseline.

        Args:
            new_model: New model to evaluate
            baseline_model: Baseline model
            num_games: Number of games to play
            win_threshold: Win rate threshold to consider new model better

        Returns:
            bool: True if new model exceeds win threshold
        """
        results = self.play_match(new_model, baseline_model, num_games)

        win_rate = results['model1_wins'] / results['total_games']

        print(f"\nEvaluation Results:")
        print(f"  New model wins: {results['model1_wins']}")
        print(f"  Baseline wins: {results['model2_wins']}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win rate: {win_rate:.2%}")
        print(f"  Threshold: {win_threshold:.2%}")

        return win_rate >= win_threshold
