"""
Mandala game state representation.

Mandala is a 2-player card game with 6 colors and values 1-6.
Players play cards to 3 mountains (left, center, right).
When a mountain is completed, cards flow to fields and players collect sets.
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Card:
    """Represents a card with color (0-5) and value (1-6)."""
    color: int  # 0-5
    value: int  # 1-6

    def __hash__(self):
        return hash((self.color, self.value))

    def __eq__(self, other):
        return self.color == other.color and self.value == other.value


class GameState:
    """
    Immutable game state representation.

    State includes:
    - Player hands (cards in hand)
    - Three mountains (left, center, right)
    - Two fields (one per player)
    - River (shared discard area)
    - Cups (scored cards per player)
    - Current player
    """

    def __init__(self):
        # 6 colors × 6 values = 36 cards total
        self.deck: List[Card] = self._create_deck()
        self.player_hands: Tuple[List[Card], List[Card]] = ([], [])
        self.mountains: Tuple[List[Card], List[Card], List[Card]] = ([], [], [])
        self.fields: Tuple[List[Card], List[Card]] = ([], [])
        self.river: List[Card] = []
        self.cups: Tuple[List[Card], List[Card]] = ([], [])
        self.current_player: int = 0  # 0 or 1
        self.game_over: bool = False

    def _create_deck(self) -> List[Card]:
        """Create a standard Mandala deck."""
        return [Card(color, value) for color in range(6) for value in range(1, 7)]

    def copy(self) -> 'GameState':
        """Create a deep copy of the state."""
        new_state = GameState()
        new_state.player_hands = (
            self.player_hands[0].copy(),
            self.player_hands[1].copy()
        )
        new_state.mountains = (
            self.mountains[0].copy(),
            self.mountains[1].copy(),
            self.mountains[2].copy()
        )
        new_state.fields = (self.fields[0].copy(), self.fields[1].copy())
        new_state.river = self.river.copy()
        new_state.cups = (self.cups[0].copy(), self.cups[1].copy())
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        return new_state

    def to_tensor(self) -> np.ndarray:
        """
        Convert state to neural network input tensor.

        Returns:
            np.ndarray: State tensor of shape (C, H, W) where:
                - Planes for each card (6 colors × 6 values = 36 planes)
                - Planes for each location (hand, mountains, fields, river, cups)
                - Planes for player indicator and game phase
        """
        # TODO: Implement efficient tensor representation
        # For now, return placeholder
        return np.zeros((50, 8, 8), dtype=np.float32)

    def get_canonical_form(self) -> 'GameState':
        """Return state from current player's perspective."""
        if self.current_player == 0:
            return self
        # Swap player perspectives
        state = self.copy()
        state.player_hands = (state.player_hands[1], state.player_hands[0])
        state.fields = (state.fields[1], state.fields[0])
        state.cups = (state.cups[1], state.cups[0])
        state.current_player = 0
        return state

    def __hash__(self):
        """Hash for transposition table."""
        return hash((
            tuple(self.player_hands[0]),
            tuple(self.player_hands[1]),
            tuple(self.mountains[0]),
            tuple(self.mountains[1]),
            tuple(self.mountains[2]),
            self.current_player
        ))
