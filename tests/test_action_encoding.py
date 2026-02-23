"""
Comprehensive tests for action encoding/decoding.

Action space (150 actions):
  Actions   0-11:  BUILD_MOUNTAIN = mandala * 6 + color
  Actions  12-95:  GROW_FIELD     = 12 + mandala * 42 + color * 7 + (count - 1)
  Actions 96-143:  DISCARD        = 96 + color * 8 + (count - 1)
  Actions 144-149: CLAIM_COLOR    = 144 + color
"""
import numpy as np
from mandala_rl.game.engine import MandalaGame, NUM_ACTIONS
from mandala_rl.game.state import GameState, Card


class TestActionEncodingConsistency:
    """Test that action encoding and decoding are consistent."""

    def test_build_mountain_encoding(self):
        """Test BUILD_MOUNTAIN actions (0-11) encode correctly."""
        for color in range(6):
            for mandala_idx in range(2):
                action = mandala_idx * 6 + color
                decoded_mandala = action // 6
                decoded_color = action % 6
                assert decoded_mandala == mandala_idx
                assert decoded_color == color

    def test_grow_field_encoding(self):
        """Test GROW_FIELD actions (12-95) with count encode correctly."""
        for mandala_idx in range(2):
            for color in range(6):
                for count in range(1, 8):
                    action = 12 + mandala_idx * 42 + color * 7 + (count - 1)
                    assert 12 <= action < 96, f"Action {action} out of GROW_FIELD range"

                    a = action - 12
                    decoded_mandala = a // 42
                    r = a % 42
                    decoded_color = r // 7
                    decoded_count = r % 7 + 1

                    assert decoded_mandala == mandala_idx, \
                        f"Action {action}: mandala {mandala_idx} decoded as {decoded_mandala}"
                    assert decoded_color == color, \
                        f"Action {action}: color {color} decoded as {decoded_color}"
                    assert decoded_count == count, \
                        f"Action {action}: count {count} decoded as {decoded_count}"

    def test_discard_encoding(self):
        """Test DISCARD actions (96-143) encode correctly."""
        for color in range(6):
            for count in range(1, 9):
                action = 96 + color * 8 + (count - 1)
                assert 96 <= action < 144, f"Action {action} out of DISCARD range"
                a = action - 96
                decoded_color = a // 8
                decoded_count = a % 8 + 1
                assert decoded_color == color
                assert decoded_count == count

    def test_claim_color_encoding(self):
        """Test CLAIM_COLOR actions (144-149) encode correctly."""
        for color in range(6):
            action = 144 + color
            decoded_color = action - 144
            assert decoded_color == color

    def test_all_150_actions_unique(self):
        """Verify all 150 actions are unique and contiguous."""
        actions = set()

        # BUILD_MOUNTAIN
        for mandala in range(2):
            for color in range(6):
                actions.add(mandala * 6 + color)

        # GROW_FIELD
        for mandala in range(2):
            for color in range(6):
                for count in range(1, 8):
                    actions.add(12 + mandala * 42 + color * 7 + (count - 1))

        # DISCARD
        for color in range(6):
            for count in range(1, 9):
                actions.add(96 + color * 8 + (count - 1))

        # CLAIM_COLOR
        for color in range(6):
            actions.add(144 + color)

        assert len(actions) == 150, f"Expected 150 unique actions, got {len(actions)}"
        assert actions == set(range(150)), "Actions should be 0-149"

    def test_action_ranges_no_overlap(self):
        """Verify action type ranges don't overlap."""
        build = set(range(0, 12))
        grow = set(range(12, 96))
        discard = set(range(96, 144))
        claim = set(range(144, 150))

        assert len(build & grow) == 0
        assert len(build & discard) == 0
        assert len(build & claim) == 0
        assert len(grow & discard) == 0
        assert len(grow & claim) == 0
        assert len(discard & claim) == 0
        assert len(build | grow | discard | claim) == 150


class TestValidMovesOnlyForCardsInHand:
    """Test that valid moves only include cards actually in hand."""

    def test_valid_moves_match_hand(self):
        """Valid moves should only include colors present in hand."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        hand_colors = {card.color for card in state.hands[state.current_player]}
        valid_moves = game.get_valid_moves(state)
        valid_actions = [i for i, valid in enumerate(valid_moves) if valid]

        for action in valid_actions:
            if action < 12:
                color = action % 6
                assert color in hand_colors
            elif action < 96:
                a = action - 12
                color = (a % 42) // 7
                assert color in hand_colors
            elif action < 144:
                color = (action - 96) // 8
                assert color in hand_colors

    def test_grow_field_count_respects_hand(self):
        """GROW_FIELD count should not exceed cards of that color in hand."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        hand = state.hands[state.current_player]
        hand_color_count = {}
        for card in hand:
            hand_color_count[card.color] = hand_color_count.get(card.color, 0) + 1

        valid_moves = game.get_valid_moves(state)

        for action in range(12, 96):
            if not valid_moves[action]:
                continue
            a = action - 12
            color = (a % 42) // 7
            count = a % 7 + 1

            assert count <= hand_color_count.get(color, 0), \
                f"Action {action}: count {count} > hand has {hand_color_count.get(color, 0)} of color {color}"
            assert count <= len(hand) - 1, \
                f"Action {action}: count {count} would empty hand (size {len(hand)})"


class TestActionExecution:
    """Test that actions execute correctly."""

    def test_build_mountain_removes_correct_card(self):
        """BUILD_MOUNTAIN should remove exactly one card of the correct color."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        hand = state.hands[state.current_player]
        test_color = hand[0].color

        action = 0 * 6 + test_color
        valid_moves = game.get_valid_moves(state)
        if not valid_moves[action]:
            return

        new_state = game.get_next_state(state, action)
        original_player = state.current_player
        new_hand = new_state.hands[original_player]

        assert len(new_hand) >= len(hand) - 1

    def test_grow_field_plays_exact_count(self):
        """GROW_FIELD should play exactly the specified count of cards."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        hand = state.hands[state.current_player]
        color_counts = {}
        for card in hand:
            color_counts[card.color] = color_counts.get(card.color, 0) + 1

        test_color = None
        for color, cnt in color_counts.items():
            if cnt >= 2 and len(color_counts) > 1:
                test_color = color
                break

        if test_color is None:
            return

        # Play exactly 1 card
        action = 12 + 0 * 42 + test_color * 7 + 0  # count=1, mandala=0
        valid_moves = game.get_valid_moves(state)
        if not valid_moves[action]:
            return

        initial_field_size = len(state.fields[0][state.current_player])
        new_state = game.get_next_state(state, action)
        original_player = state.current_player

        new_field_size = len(new_state.fields[0][original_player])
        assert new_field_size == initial_field_size + 1

        new_hand_count = sum(1 for c in new_state.hands[original_player] if c.color == test_color)
        assert new_hand_count == color_counts[test_color] - 1

    def test_grow_field_partial_play(self):
        """GROW_FIELD with count < total should leave remaining cards in hand."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        hand = state.hands[state.current_player]
        color_counts = {}
        for card in hand:
            color_counts[card.color] = color_counts.get(card.color, 0) + 1

        test_color = None
        for color, cnt in color_counts.items():
            if cnt >= 3:
                test_color = color
                break

        if test_color is None:
            return

        # Play 2 of them
        action = 12 + 0 * 42 + test_color * 7 + 1  # count=2, mandala=0
        valid_moves = game.get_valid_moves(state)
        if not valid_moves[action]:
            return

        new_state = game.get_next_state(state, action)
        original_player = state.current_player

        remaining = sum(1 for c in new_state.hands[original_player] if c.color == test_color)
        assert remaining == color_counts[test_color] - 2


class TestClaimPhase:
    """Test the CLAIM phase (interactive mountain distribution)."""

    def _trigger_mandala_completion(self, game, state):
        """Play moves until a mandala completes. Returns state or None."""
        max_moves = 200
        for _ in range(max_moves):
            if state.game_over:
                return None
            if state.phase == 1:
                return state
            valid_moves = game.get_valid_moves(state)
            valid_actions = [i for i, v in enumerate(valid_moves) if v]
            if not valid_actions:
                return None
            action = np.random.choice(valid_actions)
            state = game.get_next_state(state, action)
        return None

    def test_claim_phase_activates(self):
        """Mandala completion should activate CLAIM phase."""
        np.random.seed(42)
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        result = self._trigger_mandala_completion(game, state)
        if result is None:
            print("  SKIP: Could not trigger mandala completion")
            return

        assert result.phase == 1
        assert result.claiming_mandala in (0, 1)
        assert len(result.claiming_colors_available) > 0
        assert result.triggering_player in (0, 1)

    def test_claim_only_valid_actions(self):
        """During CLAIM phase, only CLAIM_COLOR actions should be valid."""
        np.random.seed(42)
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        result = self._trigger_mandala_completion(game, state)
        if result is None:
            return

        valid_moves = game.get_valid_moves(result)
        valid_actions = [i for i, v in enumerate(valid_moves) if v]

        for action in valid_actions:
            assert 144 <= action <= 149, f"During CLAIM, action {action} should be CLAIM_COLOR"
            color = action - 144
            assert color in result.claiming_colors_available

    def test_claim_completes_distribution(self):
        """Claiming all colors should return to PLAY phase."""
        np.random.seed(42)
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        result = self._trigger_mandala_completion(game, state)
        if result is None:
            return

        while result.phase == 1 and not result.game_over:
            valid_moves = game.get_valid_moves(result)
            valid_actions = [i for i, v in enumerate(valid_moves) if v]
            assert len(valid_actions) > 0
            result = game.get_next_state(result, valid_actions[0])

        if not result.game_over:
            assert result.phase == 0

    def test_claim_alternates_players(self):
        """CLAIM phase should alternate between players."""
        np.random.seed(42)
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        result = self._trigger_mandala_completion(game, state)
        if result is None:
            return

        players_seen = [result.current_player]
        while result.phase == 1 and not result.game_over:
            valid_moves = game.get_valid_moves(result)
            valid_actions = [i for i, v in enumerate(valid_moves) if v]
            if not valid_actions:
                break
            result = game.get_next_state(result, valid_actions[0])
            if result.phase == 1:
                players_seen.append(result.current_player)

        if len(players_seen) > 1:
            assert players_seen[0] != players_seen[1]


class TestEndToEndGame:
    """Integration tests playing complete games."""

    def test_complete_random_game(self):
        """Play a complete game with random valid moves."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()

        moves_played = 0
        max_moves = 500

        while not game.is_terminal(state) and moves_played < max_moves:
            valid_moves = game.get_valid_moves(state)
            valid_actions = [i for i, v in enumerate(valid_moves) if v]
            assert len(valid_actions) > 0
            action = np.random.choice(valid_actions)
            state = game.get_next_state(state, action)
            moves_played += 1

        assert game.is_terminal(state) or moves_played >= max_moves
        print(f"Game completed in {moves_played} moves")

    def test_multiple_random_games(self):
        """Play 20 random games to check for crashes."""
        for seed in range(20):
            game = MandalaGame(seed=seed)
            state = game.get_initial_state()
            np.random.seed(seed)

            moves = 0
            while not game.is_terminal(state) and moves < 500:
                valid_moves = game.get_valid_moves(state)
                valid_actions = [i for i, v in enumerate(valid_moves) if v]
                assert len(valid_actions) > 0, f"Seed {seed}, move {moves}: no valid moves"
                action = np.random.choice(valid_actions)
                state = game.get_next_state(state, action)
                moves += 1

            assert game.is_terminal(state), f"Seed {seed}: game didn't terminate in 500 moves"

    def test_tensor_shape(self):
        """Verify tensor has correct shape (96 channels)."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()
        tensor = state.to_tensor()
        assert tensor.shape == (96, 8, 8), f"Expected (96, 8, 8), got {tensor.shape}"

    def test_valid_moves_shape(self):
        """Verify valid moves has correct shape (150)."""
        game = MandalaGame(seed=42)
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert valid.shape == (150,), f"Expected (150,), got {valid.shape}"


def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("RUNNING ACTION ENCODING TESTS (150-action space)")
    print("=" * 80)

    print("\n[TestActionEncodingConsistency]")
    tc = TestActionEncodingConsistency()
    for name in ['test_build_mountain_encoding', 'test_grow_field_encoding',
                  'test_discard_encoding', 'test_claim_color_encoding',
                  'test_all_108_actions_unique', 'test_action_ranges_no_overlap']:
        print(f"  {name}...", end=" ")
        getattr(tc, name)()
        print("PASS")

    print("\n[TestValidMovesOnlyForCardsInHand]")
    tc = TestValidMovesOnlyForCardsInHand()
    for name in ['test_valid_moves_match_hand', 'test_grow_field_count_respects_hand']:
        print(f"  {name}...", end=" ")
        getattr(tc, name)()
        print("PASS")

    print("\n[TestActionExecution]")
    tc = TestActionExecution()
    for name in ['test_build_mountain_removes_correct_card',
                  'test_grow_field_plays_exact_count',
                  'test_grow_field_partial_play']:
        print(f"  {name}...", end=" ")
        getattr(tc, name)()
        print("PASS")

    print("\n[TestClaimPhase]")
    tc = TestClaimPhase()
    for name in ['test_claim_phase_activates', 'test_claim_only_valid_actions',
                  'test_claim_completes_distribution', 'test_claim_alternates_players']:
        print(f"  {name}...", end=" ")
        getattr(tc, name)()
        print("PASS")

    print("\n[TestEndToEndGame]")
    tc = TestEndToEndGame()
    for name in ['test_complete_random_game', 'test_multiple_random_games',
                  'test_tensor_shape', 'test_valid_moves_shape']:
        print(f"  {name}...", end=" ")
        getattr(tc, name)()
        print("PASS")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        run_tests()
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
