"""
Mandala game engine implementing complete rules.
"""
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from .state import GameState, Card


class Action:
    """Represents a game action."""
    # Action types
    BUILD_MOUNTAIN = 0
    GROW_FIELD = 1
    DISCARD = 2

    def __init__(self, action_type: int, card_indices: List[int], target: int):
        """
        Args:
            action_type: BUILD_MOUNTAIN, GROW_FIELD, or DISCARD
            card_indices: Indices of cards in hand to play
            target: For BUILD_MOUNTAIN/GROW_FIELD: mandala index (0 or 1)
                   For DISCARD: ignored
        """
        self.action_type = action_type
        self.card_indices = card_indices  # Indices in current player's hand
        self.target = target  # Mandala index

    def __repr__(self):
        types = ["BUILD_MOUNTAIN", "GROW_FIELD", "DISCARD"]
        return f"Action({types[self.action_type]}, cards={self.card_indices}, target={self.target})"


class MandalaGame:
    """
    Complete Mandala game engine.

    Implements all rules from the official rulebook.
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

        # Create and shuffle deck
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

        Move encoding:
        - Actions 0-11: BUILD_MOUNTAIN (6 colors × 2 mandalas)
        - Actions 12-23: GROW_FIELD (6 colors × 2 mandalas)
        - Actions 24-29: DISCARD (6 colors)
        Total: 30 action types

        Returns:
            np.ndarray: Binary mask of shape (30,)
        """
        valid = np.zeros(30, dtype=np.float32)

        if state.game_over:
            return valid

        hand = state.hands[state.current_player]
        if not hand:
            return valid

        hand_colors = {}
        for idx, card in enumerate(hand):
            if card.color not in hand_colors:
                hand_colors[card.color] = []
            hand_colors[card.color].append(idx)

        # BUILD_MOUNTAIN actions (play 1 card to mountain)
        for color in hand_colors:
            for mandala_idx in range(2):
                if self._can_play_to_mountain(state, color, mandala_idx):
                    action_id = mandala_idx * 6 + color  # Correct encoding
                    valid[action_id] = 1.0

        # GROW_FIELD actions (play 1+ cards of same color to field)
        for color in hand_colors:
            # Can only play if have at least 1 card but not all cards
            # (must keep at least 1 card in hand)
            if len(hand_colors[color]) < len(hand):
                for mandala_idx in range(2):
                    if self._can_play_to_field(state, color, mandala_idx, state.current_player):
                        action_id = 12 + mandala_idx * 6 + color  # Correct encoding
                        valid[action_id] = 1.0

        # DISCARD actions (always valid for any color in hand)
        for color in hand_colors:
            action_id = 24 + color
            valid[action_id] = 1.0

        return valid

    def _can_play_to_mountain(self, state: GameState, color: int, mandala_idx: int) -> bool:
        """Check if color can be played to Mountain (Rule of Color)."""
        # Color cannot already be in either Field of this Mandala
        colors_in_fields = set()
        colors_in_fields.update(c.color for c in state.fields[mandala_idx][0])
        colors_in_fields.update(c.color for c in state.fields[mandala_idx][1])
        return color not in colors_in_fields

    def _can_play_to_field(self, state: GameState, color: int, mandala_idx: int, player: int) -> bool:
        """Check if color can be played to Field (Rule of Color)."""
        # Color cannot be in Mountain or opponent's Field
        colors_in_mountain = {c.color for c in state.mountains[mandala_idx]}
        opponent = 1 - player
        colors_in_opp_field = {c.color for c in state.fields[mandala_idx][opponent]}

        return color not in colors_in_mountain and color not in colors_in_opp_field

    def get_next_state(self, state: GameState, action_id: int) -> GameState:
        """
        Apply action and return new state.

        Action encoding:
        - 0-11: BUILD_MOUNTAIN (mandala * 6 + color)
        - 12-23: GROW_FIELD (12 + mandala * 6 + color)
        - 24-29: DISCARD (24 + color)
        """
        new_state = state.copy()
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
                # Debug: log the state when invalid action occurs
                hand_colors = [c.color for c in hand]
                print(f"\n=== INVALID ACTION DEBUG ===")
                print(f"Attempted action: BUILD_MOUNTAIN color={color}, mandala={mandala_idx}")
                print(f"Current player: {player}")
                print(f"Hand colors: {hand_colors}")
                print(f"Hand size: {len(hand)}")
                print(f"===========================\n")
                raise ValueError(f"Invalid action: no card of color {color} in hand")

            # Add to mountain
            new_state.mountains[mandala_idx].append(card_to_play)

            # Draw up to 3 cards (max hand size 8)
            cards_to_draw = min(3, self.MAX_HAND_SIZE - len(hand))
            for _ in range(cards_to_draw):
                self._check_and_reshuffle_deck(new_state)
                if new_state.deck:
                    new_state.hands[player].append(new_state.deck.pop())

            # Check if mandala completed
            if new_state.is_mandala_complete(mandala_idx):
                completed_mandala = mandala_idx

        elif action_id < 24:
            # GROW_FIELD
            action_id -= 12
            mandala_idx = action_id // 6
            color = action_id % 6

            # Remove all cards of this color from hand (but keep at least 1 card total)
            cards_to_play = []
            remaining = []
            for card in hand:
                if card.color == color:
                    cards_to_play.append(card)
                else:
                    remaining.append(card)

            # If this would empty hand, only play all but one
            if not remaining:
                remaining.append(cards_to_play.pop())

            new_state.hands[player] = remaining

            # Add to field
            new_state.fields[mandala_idx][player].extend(cards_to_play)

            # No drawing for GROW_FIELD action

            # Check if mandala completed
            if new_state.is_mandala_complete(mandala_idx):
                completed_mandala = mandala_idx

        else:
            # DISCARD
            color = action_id - 24

            # Remove all cards of this color from hand
            cards_to_discard = []
            remaining = []
            for card in hand:
                if card.color == color:
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

        # Handle completed Mandala
        if completed_mandala is not None:
            new_state = self._destroy_mandala(new_state, completed_mandala)

            # Check if game should end
            if self._should_game_end(new_state):
                new_state.game_over = True
                new_state.current_player = 1 - player
                return new_state

            # Refill mountain with 2 cards
            self._check_and_reshuffle_deck(new_state)
            if new_state.deck:
                new_state.mountains[completed_mandala].append(new_state.deck.pop())
            self._check_and_reshuffle_deck(new_state)
            if new_state.deck:
                new_state.mountains[completed_mandala].append(new_state.deck.pop())

        # Switch player
        new_state.current_player = 1 - player

        # End game if next player has no cards and deck is exhausted
        if not new_state.hands[new_state.current_player]:
            self._check_and_reshuffle_deck(new_state)
            if not new_state.deck:
                new_state.game_over = True

        return new_state

    def _destroy_mandala(self, state: GameState, mandala_idx: int) -> GameState:
        """
        Handle Mandala completion: players claim cards from Mountain.

        Returns modified state.
        """
        # Determine who chooses first
        field_counts = [
            len(state.fields[mandala_idx][0]),
            len(state.fields[mandala_idx][1])
        ]

        if field_counts[0] > field_counts[1]:
            first_player = 0
        elif field_counts[1] > field_counts[0]:
            first_player = 1
        else:
            # Tie: player who did NOT play last card chooses first
            # (current_player just played, so other player chooses first)
            first_player = 1 - state.current_player

        # Get colors available in mountain
        mountain_colors = {}
        for card in state.mountains[mandala_idx]:
            if card.color not in mountain_colors:
                mountain_colors[card.color] = []
            mountain_colors[card.color].append(card)

        # Players alternate claiming colors
        current_claimer = first_player
        colors_to_claim = list(mountain_colors.keys())

        # Deterministic claiming order (sorted by color for reproducibility)
        colors_to_claim.sort()

        for color in colors_to_claim:
            cards = mountain_colors[color]

            # Check if player has cards in field (if not, discard all)
            if field_counts[current_claimer] == 0:
                state.discard.extend(cards)
            else:
                # Check if color already in River
                river_colors = state.get_colors_in_river(current_claimer)

                if color in river_colors:
                    # Color already in River: all cards go to Cup
                    state.cups[current_claimer].extend(cards)
                else:
                    # New color: 1 card to River, rest to Cup
                    if cards:
                        state.rivers[current_claimer].append(cards[0])
                        if len(cards) > 1:
                            state.cups[current_claimer].extend(cards[1:])

            # Alternate claimer
            current_claimer = 1 - current_claimer

        # Return field cards to bottom of deck (per official rules)
        field_cards = state.fields[mandala_idx][0] + state.fields[mandala_idx][1]
        state.deck = field_cards + state.deck  # Insert at bottom (index 0), deck pops from end
        state.fields[mandala_idx][0] = []
        state.fields[mandala_idx][1] = []

        # Clear mountain
        state.mountains[mandala_idx] = []

        return state

    def _check_and_reshuffle_deck(self, state: GameState):
        """
        Check if deck needs reshuffling and handle it.

        Called when trying to draw from an empty deck.
        Per official rules: When deck is exhausted, reshuffle discard pile once.
        Game then continues until the next Mandala completion.
        """
        if not state.deck and not state.deck_reshuffled and state.discard:
            # First time deck is exhausted - reshuffle discard pile
            state.deck = state.discard.copy()
            state.discard = []
            # Make shuffle deterministic based on game state for MCTS
            shuffle_seed = hash(state) & 0x7FFFFFFF
            random.Random(shuffle_seed).shuffle(state.deck)
            state.deck_reshuffled = True
            # Set flag: game will end after NEXT Mandala completion
            state.game_ends_next_mandala = True

    def _should_game_end(self, state: GameState) -> bool:
        """
        Check if game should end.

        Called after Mandala destruction to check end conditions.
        """
        # End if someone has 6 colors in River
        if len(state.rivers[0]) >= 6 or len(state.rivers[1]) >= 6:
            return True

        # CRITICAL: End if game_ends_next_mandala flag is set
        # This is set when deck is exhausted, and game ends after NEXT Mandala completion
        if state.game_ends_next_mandala:
            return True

        return False

    def is_terminal(self, state: GameState) -> bool:
        """Check if game has ended."""
        return state.game_over

    def get_reward(self, state: GameState, player: int) -> float:
        """
        Get reward for player in terminal state.

        Scoring:
        - Each card in Cup scores points based on matching River position
        - River position 1 = 1pt, position 2 = 2pt, ..., position 6 = 6pt
        - Cards with no matching River color = 0pts

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

        # Create color -> position mapping (position determines value)
        color_values = {}
        for position, card in enumerate(river):
            color_values[card.color] = position + 1  # 1-indexed

        # Score each card in Cup
        for card in cup:
            if card.color in color_values:
                score += color_values[card.color]

        return score

    def state_to_string(self, state: GameState) -> str:
        """Convert state to human-readable string."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"Player {state.current_player}'s turn")
        lines.append(f"{'='*60}")

        for player in range(2):
            lines.append(f"\nPlayer {player}:")
            lines.append(f"  Hand: {len(state.hands[player])} cards - {[str(c) for c in state.hands[player]]}")
            lines.append(f"  River: {[str(c) for c in state.rivers[player]]} ({len(state.rivers[player])}/6)")
            lines.append(f"  Cup: {len(state.cups[player])} cards")

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
        elif action_id < 24:
            a = action_id - 12
            color = a % 6
            mandala = a // 6
            return f"GROW_FIELD: {colors[color]} -> Field {mandala}"
        else:
            color = action_id - 24
            return f"DISCARD: {colors[color]}"

    def state_to_summary(self, state: GameState) -> Dict:
        """Convert game state to JSON-serializable summary for replays."""
        return {
            'current_player': state.current_player,
            'hands': [
                [c.color for c in state.hands[0]],
                [c.color for c in state.hands[1]]
            ],
            'rivers': [
                [c.color for c in state.rivers[0]],
                [c.color for c in state.rivers[1]]
            ],
            'cups': [len(state.cups[0]), len(state.cups[1])],
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

    def get_symmetries(self, state: GameState, policy: np.ndarray) -> List[Tuple[GameState, np.ndarray]]:
        """
        Get symmetrically equivalent states.

        For Mandala, there's mandala swap symmetry (left <-> right).
        """
        symmetries = [(state, policy)]

        # Swap mandalas
        sym_state = state.copy()
        sym_state.mountains = [sym_state.mountains[1], sym_state.mountains[0]]
        sym_state.fields = [sym_state.fields[1], sym_state.fields[0]]

        # Swap policy for mandala-specific actions
        sym_policy = policy.copy()
        # BUILD_MOUNTAIN: swap mandala index (mandala 0 <-> mandala 1)
        for color in range(6):
            sym_policy[0 * 6 + color], sym_policy[1 * 6 + color] = policy[1 * 6 + color], policy[0 * 6 + color]
        # GROW_FIELD: swap mandala index
        for color in range(6):
            sym_policy[12 + 0 * 6 + color], sym_policy[12 + 1 * 6 + color] = \
                policy[12 + 1 * 6 + color], policy[12 + 0 * 6 + color]
        # DISCARD: no change needed

        symmetries.append((sym_state, sym_policy))

        return symmetries
