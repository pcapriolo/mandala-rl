"""
Mandala game engine implementing complete rules.

Action space (108 actions):
  Actions   0-11:  BUILD_MOUNTAIN = mandala * 6 + color
  Actions  12-95:  GROW_FIELD     = 12 + mandala * 42 + color * 7 + (count - 1)
  Actions 96-101:  DISCARD        = 96 + color
  Actions 102-107: CLAIM_COLOR    = 102 + color
"""
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from .state import GameState, Card

NUM_ACTIONS = 150

# Phase constants
PHASE_PLAY = 0
PHASE_CLAIM = 1


class Action:
    """Represents a game action."""
    BUILD_MOUNTAIN = 0
    GROW_FIELD = 1
    DISCARD = 2
    CLAIM_COLOR = 3

    def __init__(self, action_type: int, card_indices: List[int], target: int):
        self.action_type = action_type
        self.card_indices = card_indices
        self.target = target

    def __repr__(self):
        types = ["BUILD_MOUNTAIN", "GROW_FIELD", "DISCARD", "CLAIM_COLOR"]
        return f"Action({types[self.action_type]}, cards={self.card_indices}, target={self.target})"


class MandalaGame:
    """
    Complete Mandala game engine.

    Implements all rules from the official Lookout Games rulebook.
    """

    MAX_HAND_SIZE = 8
    NUM_COLORS = 6
    CARDS_PER_COLOR = 18

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def get_initial_state(self) -> GameState:
        """Set up a new game according to official rules."""
        state = GameState()

        state.deck = state._create_deck()
        random.shuffle(state.deck)

        # Deal 2 cards to each Mountain
        state.mountains[0] = [state.deck.pop(), state.deck.pop()]
        state.mountains[1] = [state.deck.pop(), state.deck.pop()]

        # Each player: 6 cards in hand, 2 cards in Cup
        for player in range(2):
            state.hands[player] = [state.deck.pop() for _ in range(6)]
            state.cups[player] = [state.deck.pop() for _ in range(2)]

        state.current_player = 0
        return state

    def get_valid_moves(self, state: GameState) -> np.ndarray:
        """
        Get binary mask of valid moves.

        Returns:
            np.ndarray: Binary mask of shape (108,)
        """
        valid = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if state.game_over:
            return valid

        # CLAIM phase: only CLAIM_COLOR actions (144-149)
        if state.phase == PHASE_CLAIM:
            for color in state.claiming_colors_available:
                valid[144 + color] = 1.0
            return valid

        # PLAY phase
        hand = state.hands[state.current_player]
        if not hand:
            return valid

        hand_color_count = {}
        for card in hand:
            hand_color_count[card.color] = hand_color_count.get(card.color, 0) + 1
        hand_size = len(hand)

        # BUILD_MOUNTAIN: action = mandala * 6 + color (0-11)
        for color in hand_color_count:
            for mandala_idx in range(2):
                if self._can_play_to_mountain(state, color, mandala_idx):
                    valid[mandala_idx * 6 + color] = 1.0

        # GROW_FIELD: action = 12 + mandala * 42 + color * 7 + (count - 1) (12-95)
        for color, count in hand_color_count.items():
            max_count = min(count, hand_size - 1)  # Must keep >=1 card
            if max_count < 1:
                continue
            for mandala_idx in range(2):
                if self._can_play_to_field(state, color, mandala_idx, state.current_player):
                    for c in range(1, max_count + 1):
                        valid[12 + mandala_idx * 42 + color * 7 + (c - 1)] = 1.0

        # DISCARD: action = 96 + color*8 + (count-1) (96-143)
        # Only available when no constructive play (mountain/field) exists
        if not valid[:96].any():
            for color, count in hand_color_count.items():
                for c in range(1, count + 1):
                    valid[96 + color * 8 + (c - 1)] = 1.0

        return valid

    def _can_play_to_mountain(self, state: GameState, color: int, mandala_idx: int) -> bool:
        """Check if color can be played to Mountain (Rule of Color).

        Color must not appear in either field, but CAN add more of a color
        already in the mountain.
        """
        colors_in_fields = set()
        colors_in_fields.update(c.color for c in state.fields[mandala_idx][0])
        colors_in_fields.update(c.color for c in state.fields[mandala_idx][1])
        return color not in colors_in_fields

    def _can_play_to_field(self, state: GameState, color: int, mandala_idx: int, player: int) -> bool:
        """Check if color can be played to Field (Rule of Color)."""
        colors_in_mountain = {c.color for c in state.mountains[mandala_idx]}
        opponent = 1 - player
        colors_in_opp_field = {c.color for c in state.fields[mandala_idx][opponent]}
        return color not in colors_in_mountain and color not in colors_in_opp_field

    def get_next_state(self, state: GameState, action_id: int) -> GameState:
        """
        Apply action and return new state.

        Action encoding:
        - 0-11:    BUILD_MOUNTAIN (mandala * 6 + color)
        - 12-95:   GROW_FIELD (12 + mandala * 42 + color * 7 + (count - 1))
        - 96-143:  DISCARD (96 + color * 8 + (count - 1))
        - 144-149: CLAIM_COLOR (144 + color)
        """
        new_state = state.copy()

        # Handle CLAIM phase actions (144-149)
        if action_id >= 144:
            return self._process_claim(new_state, action_id - 144)

        player = state.current_player
        hand = new_state.hands[player]
        completed_mandala = None

        if action_id < 12:
            # BUILD_MOUNTAIN
            mandala_idx = action_id // 6
            color = action_id % 6

            # Find and remove one card of this color from hand
            card_to_play = None
            for i, card in enumerate(hand):
                if card.color == color:
                    card_to_play = hand.pop(i)
                    break

            if card_to_play is None:
                raise ValueError(f"Invalid action: no card of color {color} in hand")

            new_state.mountains[mandala_idx].append(card_to_play)

            # Draw up to 3 cards (max hand size 8)
            cards_to_draw = min(3, self.MAX_HAND_SIZE - len(hand))
            for _ in range(cards_to_draw):
                self._check_and_reshuffle_deck(new_state)
                if new_state.deck:
                    new_state.hands[player].append(new_state.deck.pop())

            if new_state.is_mandala_complete(mandala_idx):
                completed_mandala = mandala_idx

        elif action_id < 96:
            # GROW_FIELD: decode mandala, color, count
            a = action_id - 12
            mandala_idx = a // 42
            r = a % 42
            color = r // 7
            count = r % 7 + 1

            # Remove exactly `count` cards of this color from hand
            cards_to_play = []
            remaining = []
            for card in hand:
                if card.color == color and len(cards_to_play) < count:
                    cards_to_play.append(card)
                else:
                    remaining.append(card)

            new_state.hands[player] = remaining
            new_state.fields[mandala_idx][player].extend(cards_to_play)

            # No drawing for GROW_FIELD

            if new_state.is_mandala_complete(mandala_idx):
                completed_mandala = mandala_idx

        else:
            # DISCARD: action = 96 + color*8 + (count-1)
            a = action_id - 96
            color = a // 8
            count = a % 8 + 1

            cards_to_discard = []
            remaining = []
            for card in hand:
                if card.color == color and len(cards_to_discard) < count:
                    cards_to_discard.append(card)
                else:
                    remaining.append(card)

            new_state.hands[player] = remaining
            new_state.discard.extend(cards_to_discard)

            # Draw equal number of cards
            for _ in range(len(cards_to_discard)):
                self._check_and_reshuffle_deck(new_state)
                if new_state.deck:
                    new_state.hands[player].append(new_state.deck.pop())

        # Handle completed Mandala — enter CLAIM phase
        if completed_mandala is not None:
            self._start_claiming(new_state, completed_mandala, player)
            return new_state

        # Switch player
        new_state.current_player = 1 - player

        # End game if next player has no cards and deck is exhausted
        if not new_state.hands[new_state.current_player]:
            self._check_and_reshuffle_deck(new_state)
            if not new_state.deck:
                new_state.game_over = True

        return new_state

    def _start_claiming(self, state: GameState, mandala_idx: int, triggering_player: int):
        """
        Enter CLAIM phase after a mandala completes.

        Sets up interactive distribution where players alternate picking colors.
        """
        # Determine who picks first
        fc0 = len(state.fields[mandala_idx][0])
        fc1 = len(state.fields[mandala_idx][1])

        if fc0 > fc1:
            first_player = 0
        elif fc1 > fc0:
            first_player = 1
        else:
            # Tie: player who did NOT play last card chooses first
            first_player = 1 - triggering_player

        # Get unique colors available in mountain
        colors = list(set(c.color for c in state.mountains[mandala_idx]))
        colors.sort()  # Deterministic order for display

        state.phase = PHASE_CLAIM
        state.claiming_mandala = mandala_idx
        state.claiming_colors_available = colors
        state.triggering_player = triggering_player
        state.current_player = first_player

    def _process_claim(self, state: GameState, color: int) -> GameState:
        """
        Process a CLAIM_COLOR action during distribution phase.

        Awards one color's mountain cards to the current player, then
        switches to the other player. When all colors are claimed,
        cleans up and returns to PLAY phase.
        """
        mandala_idx = state.claiming_mandala
        claimer = state.current_player

        # Gather all mountain cards of this color
        cards = [c for c in state.mountains[mandala_idx] if c.color == color]

        # Award cards: check if claimer has field contribution
        fc = len(state.fields[mandala_idx][claimer])
        if fc == 0:
            # No field contribution — cards are discarded
            state.discard.extend(cards)
        else:
            river_colors = state.get_colors_in_river(claimer)
            if color in river_colors:
                # Color already in river: all to cup
                state.cups[claimer].extend(cards)
            else:
                # New color: first card to river, rest to cup
                if cards:
                    state.rivers[claimer].append(cards[0])
                    if len(cards) > 1:
                        state.cups[claimer].extend(cards[1:])

        # Remove color from available list
        state.claiming_colors_available.remove(color)

        # Check if distribution is complete
        if not state.claiming_colors_available:
            # All colors claimed — clean up
            self._finish_claiming(state)
        else:
            # Switch to other player for next pick
            state.current_player = 1 - claimer

        return state

    def _finish_claiming(self, state: GameState):
        """Clean up after all colors have been claimed from a mandala."""
        mandala_idx = state.claiming_mandala
        triggering_player = state.triggering_player

        # Discard field cards (per rules: "place all field cards in the discard pile")
        field_cards = state.fields[mandala_idx][0] + state.fields[mandala_idx][1]
        state.discard.extend(field_cards)
        state.fields[mandala_idx][0] = []
        state.fields[mandala_idx][1] = []

        # Clear mountain (remaining cards already distributed)
        state.mountains[mandala_idx] = []

        # Reset CLAIM phase
        state.phase = PHASE_PLAY
        state.claiming_mandala = -1
        state.claiming_colors_available = []
        state.triggering_player = -1

        # Check if game should end
        if self._should_game_end(state):
            state.game_over = True
            state.current_player = 1 - triggering_player
            return

        # Refill mountain with 2 cards
        self._check_and_reshuffle_deck(state)
        if state.deck:
            state.mountains[mandala_idx].append(state.deck.pop())
        self._check_and_reshuffle_deck(state)
        if state.deck:
            state.mountains[mandala_idx].append(state.deck.pop())

        # Switch to other player for next turn
        state.current_player = 1 - triggering_player

        # End game if next player has no cards and deck is exhausted
        if not state.hands[state.current_player]:
            self._check_and_reshuffle_deck(state)
            if not state.deck:
                state.game_over = True

    def _check_and_reshuffle_deck(self, state: GameState):
        """Check if deck needs reshuffling and handle it."""
        if not state.deck and not state.deck_reshuffled and state.discard:
            state.deck = state.discard.copy()
            state.discard = []
            shuffle_seed = hash(state) & 0x7FFFFFFF
            random.Random(shuffle_seed).shuffle(state.deck)
            state.deck_reshuffled = True
            state.game_ends_next_mandala = True

    def _should_game_end(self, state: GameState) -> bool:
        """Check if game should end after Mandala destruction."""
        if len(state.rivers[0]) >= 6 or len(state.rivers[1]) >= 6:
            return True
        if state.game_ends_next_mandala:
            return True
        return False

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended."""
        return state.game_over

    def get_reward(self, state: GameState, player: int) -> float:
        """
        Get reward for player in terminal state.

        Returns:
            float: 1.0 for win, -1.0 for loss, 0.0 for draw
        """
        if not self.is_terminal(state):
            return 0.0

        scores = [self._calculate_score(state, 0), self._calculate_score(state, 1)]

        if scores[player] > scores[1 - player]:
            return 1.0
        elif scores[player] < scores[1 - player]:
            return -1.0
        else:
            # Tiebreaker: more cards in Cup wins
            if len(state.cups[player]) > len(state.cups[1 - player]):
                return 1.0
            elif len(state.cups[player]) < len(state.cups[1 - player]):
                return -1.0
            else:
                return 0.0

    def _calculate_score(self, state: GameState, player: int) -> int:
        """Calculate score for a player."""
        score = 0
        river = state.rivers[player]
        cup = state.cups[player]

        color_values = {}
        for position, card in enumerate(river):
            color_values[card.color] = position + 1  # 1-indexed

        for card in cup:
            if card.color in color_values:
                score += color_values[card.color]

        return score

    def state_to_string(self, state: GameState) -> str:
        """Convert state to human-readable string."""
        lines = []
        phase_str = "CLAIM" if state.phase == PHASE_CLAIM else "PLAY"
        lines.append(f"{'='*60}")
        lines.append(f"Player {state.current_player}'s turn ({phase_str} phase)")
        lines.append(f"{'='*60}")

        if state.phase == PHASE_CLAIM:
            colors = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']
            avail = [colors[c] for c in state.claiming_colors_available]
            lines.append(f"  Claiming from Mandala {state.claiming_mandala}: {avail}")

        for player in range(2):
            lines.append(f"\nPlayer {player}:")
            lines.append(f"  Hand: {len(state.hands[player])} cards - {[str(c) for c in state.hands[player]]}")
            lines.append(f"  River: {[str(c) for c in state.rivers[player]]} ({len(state.rivers[player])}/6)")
            lines.append(f"  Cup: {len(state.cups[player])} cards - {[str(c) for c in state.cups[player]]}")

        lines.append(f"\nMandalas:")
        for i in range(2):
            colors_present = state.get_colors_in_mandala(i)
            lines.append(f"  Mandala {i}: {len(colors_present)}/6 colors")
            lines.append(f"    Mountain: {[str(c) for c in state.mountains[i]]} ({len(state.mountains[i])} cards)")
            lines.append(f"    Field P0: {[str(c) for c in state.fields[i][0]]} ({len(state.fields[i][0])} cards)")
            lines.append(f"    Field P1: {[str(c) for c in state.fields[i][1]]} ({len(state.fields[i][1])} cards)")

        lines.append(f"\nDeck: {len(state.deck)} cards, Discard: {len(state.discard)} cards")
        lines.append(f"Deck reshuffled: {state.deck_reshuffled}")

        if state.game_over:
            lines.append(f"\n{'='*60}")
            lines.append("GAME OVER")
            score0 = self._calculate_score(state, 0)
            score1 = self._calculate_score(state, 1)
            lines.append(f"Player 0: {score0} points")
            lines.append(f"Player 1: {score1} points")
            if score0 > score1:
                lines.append("Winner: Player 0")
            elif score1 > score0:
                lines.append("Winner: Player 1")
            else:
                lines.append("Draw!")

        return "\n".join(lines)

    def action_to_string(self, action_id: int) -> str:
        """Convert action ID to readable string."""
        colors = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']
        if action_id < 12:
            color = action_id % 6
            mandala = action_id // 6
            return f"BUILD_MOUNTAIN: {colors[color]} -> Mandala {mandala}"
        elif action_id < 96:
            a = action_id - 12
            mandala = a // 42
            r = a % 42
            color = r // 7
            count = r % 7 + 1
            return f"GROW_FIELD: {count}x {colors[color]} -> Field {mandala}"
        elif action_id < 102:
            color = action_id - 96
            return f"DISCARD: {colors[color]}"
        else:
            color = action_id - 102
            return f"CLAIM_COLOR: {colors[color]}"

    def state_to_summary(self, state: GameState) -> Dict:
        """Convert game state to JSON-serializable summary for replays."""
        summary = {
            'current_player': state.current_player,
            'phase': state.phase,
            'hands': [
                [c.color for c in state.hands[0]],
                [c.color for c in state.hands[1]]
            ],
            'rivers': [
                [c.color for c in state.rivers[0]],
                [c.color for c in state.rivers[1]]
            ],
            'cups': [
                [c.color for c in state.cups[0]],
                [c.color for c in state.cups[1]]
            ],
            'mountains': [
                [c.color for c in state.mountains[0]],
                [c.color for c in state.mountains[1]]
            ],
            'fields': [
                [[c.color for c in state.fields[0][0]], [c.color for c in state.fields[0][1]]],
                [[c.color for c in state.fields[1][0]], [c.color for c in state.fields[1][1]]]
            ],
            'deck_size': len(state.deck),
            'discard_size': len(state.discard)
        }
        if state.phase == PHASE_CLAIM:
            summary['claiming_mandala'] = state.claiming_mandala
            summary['claiming_colors_available'] = state.claiming_colors_available
        return summary

    def get_symmetries(self, state: GameState, policy: np.ndarray) -> List[Tuple[GameState, np.ndarray]]:
        """
        Get symmetrically equivalent states.

        For Mandala, there's mandala swap symmetry (left <-> right).
        Skip during CLAIM phase (claiming_mandala is fixed).
        """
        symmetries = [(state, policy)]

        # No symmetry during CLAIM phase
        if state.phase == PHASE_CLAIM:
            return symmetries

        # Swap mandalas
        sym_state = state.copy()
        sym_state.mountains = [sym_state.mountains[1], sym_state.mountains[0]]
        sym_state.fields = [sym_state.fields[1], sym_state.fields[0]]

        sym_policy = policy.copy()

        # BUILD_MOUNTAIN: swap mandala 0 <-> mandala 1 (actions 0-5 <-> 6-11)
        for color in range(6):
            sym_policy[0 * 6 + color], sym_policy[1 * 6 + color] = \
                policy[1 * 6 + color], policy[0 * 6 + color]

        # GROW_FIELD: swap mandala 0 (12-53) <-> mandala 1 (54-95)
        for color in range(6):
            for count_minus_1 in range(7):
                idx_m0 = 12 + 0 * 42 + color * 7 + count_minus_1
                idx_m1 = 12 + 1 * 42 + color * 7 + count_minus_1
                sym_policy[idx_m0], sym_policy[idx_m1] = policy[idx_m1], policy[idx_m0]

        # DISCARD (96-101) and CLAIM_COLOR (102-107): no change needed

        symmetries.append((sym_state, sym_policy))

        return symmetries
