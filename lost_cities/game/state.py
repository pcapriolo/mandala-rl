"""
Lost Cities game state representation.

Lost Cities is a 2-player card game with 5 expedition colors.
Each color has 3 wager cards and numbered cards 2-10 (12 per color, 60 total).
Players play cards to expeditions in ascending order, trying to maximize score.
"""
import numpy as np
from typing import List, Set
from dataclasses import dataclass


NUM_COLORS = 5
CARDS_PER_COLOR = 12  # 3 wagers + 9 numbered (2-10)
TOTAL_CARDS = 60
HAND_SIZE = 8
INITIAL_DECK_SIZE = TOTAL_CARDS - 2 * HAND_SIZE  # 44
MAX_TURNS = 60  # Safety limit to prevent infinite discard loops (normal games ~22 turns)


@dataclass
class LCCard:
    """A Lost Cities card with color and value."""
    color: int  # 0-4 (5 expedition colors)
    value: int  # 0 = wager, 2-10 = numbered card

    def __hash__(self):
        return hash((self.color, self.value))

    def __eq__(self, other):
        return isinstance(other, LCCard) and self.color == other.color and self.value == other.value

    def __lt__(self, other):
        return (self.color, self.value) < (other.color, other.value)

    def __le__(self, other):
        return (self.color, self.value) <= (other.color, other.value)

    def __repr__(self):
        colors = ['R', 'G', 'B', 'W', 'Y']
        if self.value == 0:
            return f'{colors[self.color]}W'
        return f'{colors[self.color]}{self.value}'


class LostCitiesState:
    """
    Lost Cities game state.

    Components:
    - 5 expedition colors (Red, Green, Blue, White, Yellow)
    - Each player has:
      - Hand (8 cards, kept sorted by color then value)
      - 5 expedition piles (cards played in ascending order)
    - Shared:
      - Deck (draw pile)
      - 5 discard piles (one per color, top card visible)
    """

    def __init__(self):
        self.deck: List[LCCard] = []
        self.hands: List[List[LCCard]] = [[], []]
        self.expeditions: List[List[List[LCCard]]] = [
            [[] for _ in range(NUM_COLORS)],  # Player 0
            [[] for _ in range(NUM_COLORS)],  # Player 1
        ]
        self.discard_piles: List[List[LCCard]] = [[] for _ in range(NUM_COLORS)]
        self.current_player: int = 0
        self.game_over: bool = False
        self.turns_played: int = 0

    def copy(self) -> 'LostCitiesState':
        """Create a deep copy of the state."""
        new = LostCitiesState()
        new.deck = self.deck.copy()
        new.hands = [h.copy() for h in self.hands]
        new.expeditions = [
            [exp.copy() for exp in player_exps]
            for player_exps in self.expeditions
        ]
        new.discard_piles = [p.copy() for p in self.discard_piles]
        new.current_player = self.current_player
        new.game_over = self.game_over
        new.turns_played = self.turns_played
        return new

    def compute_score(self, player: int) -> int:
        """Compute score for a player based on current expeditions."""
        total = 0
        for color in range(NUM_COLORS):
            exp = self.expeditions[player][color]
            if not exp:
                continue
            wagers = sum(1 for c in exp if c.value == 0)
            card_sum = sum(c.value for c in exp)
            total += (card_sum - 20) * (1 + wagers)
            if len(exp) >= 8:
                total += 20
        return total

    def to_tensor(self) -> np.ndarray:
        """
        Convert state to neural network input tensor.
        State is assumed to be in canonical form (current player = index 0).

        86 channels x 8x8 (values broadcast across spatial dims):
          Ch 0-4:   My hand count per color (/8)
          Ch 5-9:   My hand wager count per color (/3)
          Ch 10-14: My expedition top value per color (/10)
          Ch 15-19: My expedition length per color (/12)
          Ch 20-24: My expedition wager count per color (/3)
          Ch 25-29: Opp expedition top value per color (/10)
          Ch 30-34: Opp expedition length per color (/12)
          Ch 35-39: Opp expedition wager count per color (/3)
          Ch 40-44: Discard pile top value per color (/10)
          Ch 45:    Deck size / 44
          Ch 46:    Opp hand size / 8
          Ch 47:    My score / 200 + 0.5
          Ch 48:    Opp score / 200 + 0.5
          Ch 49:    Game progress (turns_played / 150)
          Ch 50-57: Hand position 0-7 card color ((color+1)/6, 0=empty)
          Ch 58-65: Hand position 0-7 card value ((value+1)/11, 0=empty)
          Ch 66-70: Belief channels — P(opp has ≥1 of color c)
          Ch 71-75: Opp discard frequency per color (default 0.0)
          Ch 76-80: My played card frequency per color (default 0.0)
          Ch 81-85: Opp played card frequency per color (default 0.0)
        """
        tensor = np.zeros((86, 8, 8), dtype=np.float32)

        # My hand counts and wager counts per color
        for card in self.hands[0]:
            tensor[card.color] += 1.0 / HAND_SIZE
            if card.value == 0:
                tensor[5 + card.color] += 1.0 / 3.0

        # My expeditions
        for c in range(NUM_COLORS):
            exp = self.expeditions[0][c]
            if exp:
                tensor[10 + c] = max(card.value for card in exp) / 10.0
                tensor[15 + c] = len(exp) / CARDS_PER_COLOR
                tensor[20 + c] = sum(1 for card in exp if card.value == 0) / 3.0

        # Opponent expeditions
        for c in range(NUM_COLORS):
            exp = self.expeditions[1][c]
            if exp:
                tensor[25 + c] = max(card.value for card in exp) / 10.0
                tensor[30 + c] = len(exp) / CARDS_PER_COLOR
                tensor[35 + c] = sum(1 for card in exp if card.value == 0) / 3.0

        # Discard pile tops
        for c in range(NUM_COLORS):
            if self.discard_piles[c]:
                tensor[40 + c] = self.discard_piles[c][-1].value / 10.0

        # Scalars
        tensor[45] = len(self.deck) / INITIAL_DECK_SIZE
        tensor[46] = len(self.hands[1]) / HAND_SIZE
        tensor[47] = np.clip(self.compute_score(0) / 200.0 + 0.5, 0.0, 1.0)
        tensor[48] = np.clip(self.compute_score(1) / 200.0 + 0.5, 0.0, 1.0)
        tensor[49] = self.turns_played / MAX_TURNS

        # Per-position hand card identity (so network knows what's at each hand_pos)
        for i, card in enumerate(self.hands[0]):
            tensor[50 + i] = (card.color + 1) / 6.0   # 0=empty, 0.17-0.83 for colors
            tensor[58 + i] = (card.value + 1) / 11.0   # 0=empty, 0.09=wager, 0.27-1.0 for 2-10

        # Ch 66-70: Belief channels — P(opp has ≥1 of color c)
        # Count all known cards per color
        known = np.zeros(NUM_COLORS, dtype=np.int32)
        for card in self.hands[0]:
            known[card.color] += 1
        for c in range(NUM_COLORS):
            for card in self.expeditions[0][c]:
                known[card.color] += 1
            for card in self.expeditions[1][c]:
                known[card.color] += 1
            for card in self.discard_piles[c]:
                known[card.color] += 1

        remaining = np.maximum(CARDS_PER_COLOR - known, 0)
        total_unseen = int(remaining.sum())
        opp_hand_sz = len(self.hands[1])

        for c in range(NUM_COLORS):
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
            tensor[66 + c] = belief

        # Ch 71-85: Behavioral inference (play frequencies per color)
        # These require move-history tracking not available in the Python game state.
        # Default to 0.0 — the model was trained with these but they're only
        # informative after many moves. Zero is a safe neutral value.
        # (Ch 71-75: discard freq, Ch 76-80: my play freq, Ch 81-85: opp play freq)

        return tensor

    def get_canonical_form(self) -> 'LostCitiesState':
        """Return state from current player's perspective."""
        if self.current_player == 0:
            return self.copy()

        state = self.copy()
        state.hands = [state.hands[1], state.hands[0]]
        state.expeditions = [state.expeditions[1], state.expeditions[0]]
        state.current_player = 0
        return state

    def __hash__(self):
        """Hash for MCTS transposition table."""
        return hash((
            tuple((c.color, c.value) for c in self.hands[0]),
            tuple((c.color, c.value) for c in self.hands[1]),
            tuple(
                tuple((c.color, c.value) for c in exp)
                for player_exps in self.expeditions
                for exp in player_exps
            ),
            tuple(
                tuple((c.color, c.value) for c in pile)
                for pile in self.discard_piles
            ),
            tuple((c.color, c.value) for c in self.deck),
            self.current_player,
        ))
