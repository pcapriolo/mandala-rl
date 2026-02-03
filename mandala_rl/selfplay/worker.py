"""
Self-play worker for generating training games.
"""
import numpy as np
import torch
from typing import List, Tuple
from ..game.engine import MandalaGame
from ..game.state import GameState
from ..mcts.search import MCTS
from ..network.model import MandalaNet


class SelfPlayGame:
    """Represents a completed self-play game."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.current_players: List[int] = []
        self.outcome: float = 0.0  # Final game outcome from player 0's perspective

    def __len__(self):
        return len(self.states)


class SelfPlayWorker:
    """
    Worker for generating self-play games.

    Plays games using MCTS + neural network and collects training data.
    """

    def __init__(
        self,
        game: MandalaGame,
        network: MandalaNet,
        mcts_simulations: int = 800,
        temperature: float = 1.0,
        temperature_threshold: int = 30,  # Move to switch to temp=0
        c_puct: float = 1.0,
        device: str = "mps"
    ):
        """
        Args:
            game: Game engine
            network: Neural network
            mcts_simulations: Number of MCTS simulations per move
            temperature: Temperature for action sampling
            temperature_threshold: Move number to switch to deterministic play
            c_puct: MCTS exploration constant
            device: PyTorch device (mps for Apple Silicon)
        """
        self.game = game
        self.network = network.to(device)
        self.network.eval()
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.c_puct = c_puct
        self.device = device

        # Create MCTS instance
        self.mcts = MCTS(
            game=game,
            network=self.network.predict,
            num_simulations=mcts_simulations,
            c_puct=c_puct
        )

    def play_game(self) -> SelfPlayGame:
        """
        Play a single self-play game.

        Returns:
            SelfPlayGame: Completed game with states, policies, and outcome
        """
        game_record = SelfPlayGame()
        state = self.game.get_initial_state()
        move_count = 0

        while not self.game.is_terminal(state):
            # Get canonical state (from current player's perspective)
            canonical_state = state.get_canonical_form()

            # Determine temperature
            temp = self.temperature if move_count < self.temperature_threshold else 0.0

            # Run MCTS
            action_probs, visit_counts = self.mcts.get_action_prob(
                canonical_state,
                temperature=temp,
                add_noise=True
            )

            # Record state and policy
            game_record.states.append(canonical_state.to_tensor())
            game_record.policies.append(action_probs)
            game_record.current_players.append(state.current_player)

            # Sample action
            action = np.random.choice(len(action_probs), p=action_probs)

            # Apply action
            state = self.game.get_next_state(state, action)
            move_count += 1

        # Record outcome
        outcome_p0 = self.game.get_reward(state, player=0)
        outcome_p1 = self.game.get_reward(state, player=1)
        game_record.outcome = outcome_p0

        return game_record

    def generate_games(self, num_games: int) -> List[SelfPlayGame]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to play

        Returns:
            List of completed games
        """
        games = []
        for _ in range(num_games):
            game = self.play_game()
            games.append(game)
        return games

    def get_training_examples(self, game: SelfPlayGame) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert game to training examples.

        Each example is (state, policy, value) where value is the final
        outcome from the perspective of the current player in that state.

        Args:
            game: Completed self-play game

        Returns:
            List of (state_tensor, policy, value) tuples
        """
        examples = []
        outcome = game.outcome

        for i, (state, policy, player) in enumerate(
            zip(game.states, game.policies, game.current_players)
        ):
            # Value from current player's perspective
            value = outcome if player == 0 else -outcome

            examples.append((state, policy, value))

        return examples
