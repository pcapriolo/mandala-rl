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
        self.game_ends_next_mandala: bool = False  # Game ends after next Mandala completion

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
        new_state.game_ends_next_mandala = self.game_ends_next_mandala

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

        State is assumed to be in canonical form (current player = index 0).

        Encoding (83 planes × 8×8, values broadcast across spatial dims):
          Ch 0-5:   My hand card counts per color (/18)
          Ch 6-11:  Mountain 0 card counts per color (/18)
          Ch 12-17: Mountain 1 card counts per color (/18)
          Ch 18-23: My Field Mandala 0 counts per color (/18)
          Ch 24-29: My Field Mandala 1 counts per color (/18)
          Ch 30-35: Opp Field Mandala 0 counts per color (/18)
          Ch 36-41: Opp Field Mandala 1 counts per color (/18)
          Ch 42-47: My River value per color ((position+1)/6, 0 if absent)
          Ch 48-53: Opp River value per color ((position+1)/6, 0 if absent)
          Ch 54:    My cup size / 20
          Ch 55:    Opp cup size / 20
          Ch 56:    Opponent hand size / 8
          Ch 57:    Deck size / 108
          Ch 58:    Game ends next mandala flag (0 or 1)
          Ch 59-64: Belief channels — P(opp has ≥1 of color c)
          Ch 65-70: Opp mountain play frequency per color
          Ch 71-76: Opp field play frequency per color
          Ch 77-82: Opp discard frequency per color
        """
        tensor = np.zeros((83, 8, 8), dtype=np.float32)

        CARDS_PER_COLOR = 18

        def color_counts(cards):
            counts = np.zeros(6, dtype=np.float32)
            for card in cards:
                counts[card.color] += 1
            return counts / 18.0

        areas = [
            (0, self.hands[0]),          # My hand
            (6, self.mountains[0]),       # Mountain 0
            (12, self.mountains[1]),      # Mountain 1
            (18, self.fields[0][0]),      # My Field M0
            (24, self.fields[1][0]),      # My Field M1
            (30, self.fields[0][1]),      # Opp Field M0
            (36, self.fields[1][1]),      # Opp Field M1
        ]
        for start_ch, cards in areas:
            counts = color_counts(cards)
            for c in range(6):
                tensor[start_ch + c] = counts[c]

        # Ch 42-47: My River scoring values
        for pos, card in enumerate(self.rivers[0]):
            tensor[42 + card.color] = (pos + 1) / 6.0

        # Ch 48-53: Opponent River scoring values
        for pos, card in enumerate(self.rivers[1]):
            tensor[48 + card.color] = (pos + 1) / 6.0

        # Ch 54-55: Cup sizes
        tensor[54] = len(self.cups[0]) / 20.0
        tensor[55] = len(self.cups[1]) / 20.0

        # Ch 56: Opponent hand size
        tensor[56] = len(self.hands[1]) / 8.0
        # Ch 57: Deck size
        tensor[57] = len(self.deck) / 108.0
        # Ch 58: Game ends next mandala flag
        tensor[58] = 1.0 if self.game_ends_next_mandala else 0.0

        # Ch 59-64: Belief channels — P(opp has ≥1 of color c)
        # Count all known cards per color
        known = np.zeros(6, dtype=np.int32)
        for card in self.hands[0]:
            known[card.color] += 1
        for m in range(2):
            for card in self.mountains[m]:
                known[card.color] += 1
            for card in self.fields[m][0]:
                known[card.color] += 1
            for card in self.fields[m][1]:
                known[card.color] += 1
        for card in self.rivers[0]:
            known[card.color] += 1
        for card in self.rivers[1]:
            known[card.color] += 1

        remaining = np.maximum(CARDS_PER_COLOR - known, 0)
        total_unseen = int(remaining.sum())
        opp_hand_sz = len(self.hands[1])

        for c in range(6):
            belief = 0.0
            if remaining[c] > 0 and total_unseen > 0 and opp_hand_sz > 0:
                # P(none of color c in opp hand) via hypergeometric
                p_none = 1.0
                N = total_unseen
                k = int(remaining[c])
                for i in range(opp_hand_sz):
                    if (N - i) > 0:
                        p_none *= (N - k - i) / (N - i)
                belief = 1.0 - p_none
            tensor[59 + c] = belief

        # Ch 65-82: Behavioral inference (opp play frequency per color)
        # These require move-history tracking not available in the Python game state.
        # Default to 0.0 — the model was trained with these but they're only
        # informative after many moves. Zero is a safe neutral value.
        # (Ch 65-70: mountain plays, Ch 71-76: field plays, Ch 77-82: discards)

        return tensor

    def get_canonical_form(self) -> 'GameState':
        """Return state from current player's perspective."""
        if self.current_player == 0:
            # IMPORTANT: Must return a copy to prevent state corruption
            # MCTS reuses state objects in tree, so original must not be mutated
            return self.copy()

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
            tuple(c.color for c in self.deck),  # Include deck for deterministic shuffling
            tuple(c.color for c in self.discard),
            self.current_player,
            self.deck_reshuffled,
            self.game_ends_next_mandala
        ))
