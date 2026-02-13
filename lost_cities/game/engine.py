"""
Lost Cities game engine implementing complete rules.

Action encoding (96 compound actions):
  action = hand_pos * 12 + dest * 6 + draw_src
  hand_pos: 0-7 (index into sorted hand)
  dest:     0 = play to expedition, 1 = discard to color pile
  draw_src: 0 = deck, 1-5 = discard pile for color 0-4
"""
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from .state import LostCitiesState, LCCard, NUM_COLORS, CARDS_PER_COLOR, HAND_SIZE, MAX_TURNS

NUM_ACTIONS = 96  # 8 * 2 * 6
COLOR_NAMES = ['Red', 'Green', 'Blue', 'White', 'Yellow']


class LostCitiesGame:
    """Complete Lost Cities game engine."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _create_deck() -> List[LCCard]:
        """Create a standard Lost Cities deck: 60 cards."""
        cards = []
        for color in range(NUM_COLORS):
            for _ in range(3):
                cards.append(LCCard(color=color, value=0))
            for value in range(2, 11):
                cards.append(LCCard(color=color, value=value))
        return cards

    def get_initial_state(self) -> LostCitiesState:
        """Set up a new game."""
        state = LostCitiesState()
        deck = self._create_deck()
        random.shuffle(deck)

        state.hands[0] = sorted(deck[:HAND_SIZE])
        state.hands[1] = sorted(deck[HAND_SIZE:2 * HAND_SIZE])
        state.deck = deck[2 * HAND_SIZE:]
        state.current_player = 0
        return state

    @staticmethod
    def _can_play_on_expedition(card: LCCard, expedition: List[LCCard]) -> bool:
        """Check if a card can be legally played on an expedition."""
        if not expedition:
            return True
        last_value = expedition[-1].value
        if last_value == 0:
            return True  # Any card OK after wagers
        return card.value > last_value  # Must be strictly ascending

    def get_valid_moves(self, state: LostCitiesState) -> np.ndarray:
        """
        Get binary mask of valid moves (96 actions).

        Validity checks:
        - hand_pos must be in range
        - dest=0 (play): card must be legal for that expedition
        - dest=1 (discard): always legal
        - draw_src=0 (deck): deck must not be empty
        - draw_src=1-5 (pile): pile must not be empty
        - Can't draw from the pile you just discarded to
        """
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if state.game_over:
            return mask

        hand = state.hands[state.current_player]

        for hand_pos in range(len(hand)):
            card = hand[hand_pos]

            for dest in range(2):
                # Check play legality
                if dest == 0:
                    expedition = state.expeditions[state.current_player][card.color]
                    if not self._can_play_on_expedition(card, expedition):
                        continue

                for draw_src in range(6):
                    # Can't draw from pile you just discarded to
                    if dest == 1 and draw_src == card.color + 1:
                        continue

                    # Check draw source availability
                    if draw_src == 0:
                        if len(state.deck) == 0:
                            continue
                    else:
                        pile_color = draw_src - 1
                        if len(state.discard_piles[pile_color]) == 0:
                            continue

                    action = hand_pos * 12 + dest * 6 + draw_src
                    mask[action] = 1.0

        return mask

    def get_next_state(self, state: LostCitiesState, action: int) -> LostCitiesState:
        """Apply compound action and return new state."""
        new_state = state.copy()
        player = new_state.current_player

        # Decode action
        hand_pos = action // 12
        dest = (action % 12) // 6
        draw_src = action % 6

        card = new_state.hands[player][hand_pos]

        # Remove card from hand
        new_state.hands[player].pop(hand_pos)

        # Play or discard
        if dest == 0:
            new_state.expeditions[player][card.color].append(card)
        else:
            new_state.discard_piles[card.color].append(card)

        # Draw
        if draw_src == 0:
            drawn = new_state.deck.pop()
            new_state.hands[player].append(drawn)
            if len(new_state.deck) == 0:
                new_state.game_over = True
        else:
            pile_color = draw_src - 1
            drawn = new_state.discard_piles[pile_color].pop()
            new_state.hands[player].append(drawn)

        # Re-sort hand
        new_state.hands[player].sort()

        # Advance turn
        new_state.turns_played += 1
        if new_state.turns_played >= MAX_TURNS:
            new_state.game_over = True

        new_state.current_player = 1 - player
        return new_state

    def is_terminal(self, state: LostCitiesState) -> bool:
        """Check if game has ended."""
        return state.game_over

    def get_reward(self, state: LostCitiesState, player: int) -> float:
        """Get reward in [-1, 1] for player in terminal state."""
        if not self.is_terminal(state):
            return 0.0

        score_p = state.compute_score(player)
        score_opp = state.compute_score(1 - player)

        if score_p > score_opp:
            return 1.0
        elif score_p < score_opp:
            return -1.0
        return 0.0

    def _calculate_score(self, state: LostCitiesState, player: int) -> int:
        """Calculate score for a player. Used by web interfaces and replay."""
        return state.compute_score(player)

    def get_symmetries(self, state: LostCitiesState, policy: np.ndarray) -> List[Tuple]:
        """No useful symmetries for Lost Cities."""
        return [(state, policy)]

    def action_to_string(self, action: int) -> str:
        """Convert action ID to human-readable string."""
        hand_pos = action // 12
        dest = (action % 12) // 6
        draw_src = action % 6

        dest_str = "Play" if dest == 0 else "Discard"
        if draw_src == 0:
            draw_str = "Deck"
        else:
            draw_str = f"{COLOR_NAMES[draw_src - 1]} pile"

        return f"Hand[{hand_pos}] {dest_str}, draw {draw_str}"

    def state_to_summary(self, state: LostCitiesState) -> Dict[str, Any]:
        """Convert state to JSON-serializable summary for replays."""
        def exp_to_list(exp):
            return [{'color': c.color, 'value': c.value} for c in exp]

        return {
            'current_player': state.current_player,
            'hands': [
                [{'color': c.color, 'value': c.value} for c in state.hands[0]],
                [{'color': c.color, 'value': c.value} for c in state.hands[1]],
            ],
            'expeditions': [
                [exp_to_list(state.expeditions[p][c]) for c in range(NUM_COLORS)]
                for p in range(2)
            ],
            'discard_piles': [
                exp_to_list(state.discard_piles[c]) for c in range(NUM_COLORS)
            ],
            'deck_size': len(state.deck),
            'scores': [state.compute_score(0), state.compute_score(1)],
        }

    def state_to_string(self, state: LostCitiesState) -> str:
        """Convert state to human-readable string."""
        lines = [f"{'='*50}", f"Player {state.current_player}'s turn  (Turn {state.turns_played})", f"{'='*50}"]

        for p in range(2):
            lines.append(f"\nPlayer {p} (score: {state.compute_score(p)}):")
            lines.append(f"  Hand: {state.hands[p]}")
            for c in range(NUM_COLORS):
                exp = state.expeditions[p][c]
                if exp:
                    lines.append(f"  {COLOR_NAMES[c]}: {exp}")

        lines.append(f"\nDiscard piles:")
        for c in range(NUM_COLORS):
            pile = state.discard_piles[c]
            if pile:
                lines.append(f"  {COLOR_NAMES[c]}: {pile} (top: {pile[-1]})")

        lines.append(f"\nDeck: {len(state.deck)} cards")

        if state.game_over:
            lines.append(f"\n{'='*50}")
            lines.append("GAME OVER")
            for p in range(2):
                lines.append(f"Player {p}: {state.compute_score(p)} points")

        return "\n".join(lines)
