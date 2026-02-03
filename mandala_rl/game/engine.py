"""
Mandala game engine implementing rules and move generation.
"""
import numpy as np
from typing import List, Tuple, Optional
from .state import GameState, Card


class MandalaGame:
    """
    Game engine for Mandala.

    Handles:
    - Move generation
    - Move validation
    - State transitions
    - Terminal state detection
    - Scoring
    """

    # Mountain completion thresholds by card count
    MOUNTAIN_THRESHOLD = 6

    def get_initial_state(self) -> GameState:
        """Set up a new game with dealt cards."""
        state = GameState()
        # Deal initial hands (typically 6 cards each)
        np.random.shuffle(state.deck)
        state.player_hands = (
            state.deck[:6].copy(),
            state.deck[6:12].copy()
        )
        state.deck = state.deck[12:]
        return state

    def get_valid_moves(self, state: GameState) -> np.ndarray:
        """
        Get binary mask of valid moves.

        Move encoding:
        - Play card i to mountain j: move_id = i * 3 + j
        - Collect from field/river: separate indices

        Returns:
            np.ndarray: Binary mask of shape (num_possible_moves,)
        """
        num_cards_in_hand = len(state.player_hands[state.current_player])
        num_mountains = 3

        # TODO: Implement full move generation
        # For now, allow playing any card to any mountain
        valid_moves = np.zeros(num_cards_in_hand * num_mountains + 10, dtype=np.float32)

        if not state.game_over:
            # Allow playing any card in hand to any mountain
            for card_idx in range(num_cards_in_hand):
                for mountain_idx in range(num_mountains):
                    move_id = card_idx * num_mountains + mountain_idx
                    valid_moves[move_id] = 1.0

        return valid_moves

    def get_next_state(self, state: GameState, action: int) -> GameState:
        """
        Apply action to state and return new state.

        Args:
            state: Current game state
            action: Move index

        Returns:
            GameState: New state after move
        """
        new_state = state.copy()

        # TODO: Implement full move execution
        # Decode action, apply it, check for mountain completion, etc.

        # Switch player
        new_state.current_player = 1 - state.current_player

        # Check if game is over
        if self.is_terminal(new_state):
            new_state.game_over = True

        return new_state

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended."""
        # Game ends when all cards are scored
        total_scored = len(state.cups[0]) + len(state.cups[1])
        return total_scored == 36 or state.game_over

    def get_reward(self, state: GameState, player: int) -> float:
        """
        Get reward for player in terminal state.

        Args:
            state: Terminal game state
            player: Player index (0 or 1)

        Returns:
            float: 1.0 for win, -1.0 for loss, 0.0 for draw
        """
        if not self.is_terminal(state):
            return 0.0

        # Score is sum of card values in cups
        score_0 = sum(card.value for card in state.cups[0])
        score_1 = sum(card.value for card in state.cups[1])

        if score_0 > score_1:
            return 1.0 if player == 0 else -1.0
        elif score_1 > score_0:
            return 1.0 if player == 1 else -1.0
        else:
            return 0.0

    def get_symmetries(self, state: GameState, policy: np.ndarray) -> List[Tuple[GameState, np.ndarray]]:
        """
        Get symmetrically equivalent states for data augmentation.

        For Mandala, symmetries are limited (card colors are distinct).
        May include perspective flip.
        """
        # TODO: Implement if useful symmetries exist
        return [(state, policy)]

    def state_to_string(self, state: GameState) -> str:
        """Convert state to human-readable string."""
        lines = []
        lines.append(f"Player {state.current_player}'s turn")
        lines.append(f"Hands: P0={len(state.player_hands[0])}, P1={len(state.player_hands[1])}")
        lines.append(f"Mountains: L={len(state.mountains[0])}, C={len(state.mountains[1])}, R={len(state.mountains[2])}")
        lines.append(f"River: {len(state.river)} cards")
        lines.append(f"Cups: P0={len(state.cups[0])}, P1={len(state.cups[1])}")
        return "\n".join(lines)
