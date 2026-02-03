"""
Mandala game state representation.

Mandala is a 2-player card game with 6 colors (18 cards each = 108 total).
Players play cards to 2 Mandalas, each with a Mountain and 2 Fields.
When a Mandala completes (all 6 colors present), players claim cards to River and Cup.
"""
import numpy as np
from typing import List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Card:
    """Represents a card with just a color (0-5). Cards have no inherent value."""
    color: int  # 0-5

    def __hash__(self):
        return hash(self.color)

    def __eq__(self, other):
        return isinstance(other, Card) and self.color == other.color

    def __repr__(self):
        colors = ['⚪', '🟢', '🟣', '🟡', '🔴', '🟠']
        return colors[self.color] if 0 <= self.color < 6 else f'C{self.color}'


class GameState:
    """
    Mandala game state.

    Components:
    - 2 Mandalas (left=0, right=1), each with:
      - Mountain (central area)
      - 2 Fields (one per player)
    - Each player has:
      - Hand (max 8 cards)
      - River (ordered list of up to 6 cards, one per color, determines scoring)
      - Cup (face-down pile of scored cards)
    - Shared:
      - Deck (draw pile)
      - Discard pile
    """

    def __init__(self):
        # Deck: 18 cards per color × 6 colors = 108 cards
        self.deck: List[Card] = []
        self.discard: List[Card] = []

        # 2 Mandalas, each with Mountain + 2 Fields
        self.mountains: List[List[Card]] = [[], []]  # [mandala_0, mandala_1]
        self.fields: List[List[List[Card]]] = [
            [[], []],  # Mandala 0: [player_0_field, player_1_field]
            [[], []]   # Mandala 1: [player_0_field, player_1_field]
        ]

        # Player data
        self.hands: List[List[Card]] = [[], []]  # [player_0_hand, player_1_hand]
        self.rivers: List[List[Card]] = [[], []]  # [player_0_river, player_1_river] (ordered, max 6)
        self.cups: List[List[Card]] = [[], []]    # [player_0_cup, player_1_cup]

        # Game state
        self.current_player: int = 0  # 0 or 1
        self.game_over: bool = False
        self.deck_reshuffled: bool = False  # Track if we've reshuffled once

    def _create_deck(self) -> List[Card]:
        """Create a standard Mandala deck: 18 cards per color."""
        return [Card(color) for color in range(6) for _ in range(18)]

    def copy(self) -> 'GameState':
        """Create a deep copy of the state."""
        new_state = GameState()
        new_state.deck = self.deck.copy()
        new_state.discard = self.discard.copy()

        new_state.mountains = [m.copy() for m in self.mountains]
        new_state.fields = [[f.copy() for f in mandala] for mandala in self.fields]

        new_state.hands = [h.copy() for h in self.hands]
        new_state.rivers = [r.copy() for r in self.rivers]
        new_state.cups = [c.copy() for c in self.cups]

        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.deck_reshuffled = self.deck_reshuffled

        return new_state

    def get_colors_in_mandala(self, mandala_idx: int) -> Set[int]:
        """Get all colors present in a Mandala (Mountain + both Fields)."""
        colors = set()
        # Mountain
        colors.update(card.color for card in self.mountains[mandala_idx])
        # Both fields
        colors.update(card.color for card in self.fields[mandala_idx][0])
        colors.update(card.color for card in self.fields[mandala_idx][1])
        return colors

    def is_mandala_complete(self, mandala_idx: int) -> bool:
        """Check if Mandala has all 6 colors."""
        return len(self.get_colors_in_mandala(mandala_idx)) == 6

    def get_colors_in_river(self, player: int) -> Set[int]:
        """Get set of colors in player's River."""
        return {card.color for card in self.rivers[player]}

    def to_tensor(self) -> np.ndarray:
        """
        Convert state to neural network input tensor.

        Encoding (50 planes × 8×8):
        - 6 planes: card counts in each Mountain (by color)
        - 12 planes: card counts in Fields (6 colors × 2 mandalas)
        - 6 planes: my hand card counts (by color)
        - 6 planes: opponent hand size indicator
        - 6 planes: my River colors
        - 6 planes: opponent River colors
        - 2 planes: my Cup size / opponent Cup size
        - 6 planes: deck presence indicators
        """
        tensor = np.zeros((50, 8, 8), dtype=np.float32)

        # Simplified encoding for now
        # TODO: Implement full tensor encoding

        return tensor

    def get_canonical_form(self) -> 'GameState':
        """Return state from current player's perspective."""
        if self.current_player == 0:
            return self

        # Swap player perspectives
        state = self.copy()
        state.hands = [state.hands[1], state.hands[0]]
        state.rivers = [state.rivers[1], state.rivers[0]]
        state.cups = [state.cups[1], state.cups[0]]

        # Swap fields
        state.fields = [
            [state.fields[0][1], state.fields[0][0]],
            [state.fields[1][1], state.fields[1][0]]
        ]

        state.current_player = 0
        return state

    def __hash__(self):
        """Hash for transposition table."""
        return hash((
            tuple(c.color for c in self.hands[0]),
            tuple(c.color for c in self.hands[1]),
            tuple(c.color for c in self.mountains[0]),
            tuple(c.color for c in self.mountains[1]),
            tuple(c.color for c in self.fields[0][0]),
            tuple(c.color for c in self.fields[0][1]),
            tuple(c.color for c in self.fields[1][0]),
            tuple(c.color for c in self.fields[1][1]),
            self.current_player
        ))
