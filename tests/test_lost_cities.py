"""Basic tests for Lost Cities game engine."""
import numpy as np
from lost_cities.game.engine import LostCitiesGame
from lost_cities.game.state import LostCitiesState, HAND_SIZE, NUM_COLORS


def test_initial_state():
    """Test initial game state setup."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    assert len(state.hands[0]) == HAND_SIZE
    assert len(state.hands[1]) == HAND_SIZE
    assert len(state.deck) == 60 - 2 * HAND_SIZE
    assert state.current_player == 0
    assert not state.game_over

    # Hands should be sorted
    for p in range(2):
        for i in range(len(state.hands[p]) - 1):
            assert state.hands[p][i] <= state.hands[p][i + 1]


def test_valid_moves():
    """Test move generation."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    valid_moves = game.get_valid_moves(state)
    assert len(valid_moves) == 96
    assert np.sum(valid_moves) > 0


def test_state_copy():
    """Test state copying."""
    game = LostCitiesGame()
    state = game.get_initial_state()
    state_copy = state.copy()

    assert state is not state_copy
    assert state.current_player == state_copy.current_player
    assert len(state.hands[0]) == len(state_copy.hands[0])
    assert len(state.deck) == len(state_copy.deck)


def test_canonical_form():
    """Test canonical state representation."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    # Player 0's perspective
    canonical = state.get_canonical_form()
    assert canonical.current_player == 0

    # Player 1's perspective
    state.current_player = 1
    canonical = state.get_canonical_form()
    assert canonical.current_player == 0


def test_to_tensor():
    """Test tensor encoding produces non-zero output."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    tensor = state.to_tensor()
    assert tensor.shape == (86, 8, 8)
    assert np.any(tensor != 0), "Tensor should not be all zeros"

    # Hand channels should have values
    assert np.any(tensor[0:5] != 0), "Hand count channels should have values"
    # Deck size should be ~1.0
    assert tensor[45, 0, 0] > 0.9, "Deck should be nearly full at start"


def test_play_to_expedition():
    """Test playing a card to an expedition."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    valid = game.get_valid_moves(state)
    # Find a play-to-expedition action
    play_actions = [a for a in range(96) if valid[a] and (a % 12) // 6 == 0]
    assert len(play_actions) > 0, "Should have at least one play action"

    action = play_actions[0]
    new_state = game.get_next_state(state, action)

    # Hand should still be 8 (played 1, drew 1)
    assert len(new_state.hands[0]) == HAND_SIZE
    # Turn should switch
    assert new_state.current_player == 1


def test_discard_action():
    """Test discarding a card."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    valid = game.get_valid_moves(state)
    # Find a discard action (dest=1)
    discard_actions = [a for a in range(96) if valid[a] and (a % 12) // 6 == 1]
    assert len(discard_actions) > 0, "Should have at least one discard action"

    action = discard_actions[0]
    hand_pos = action // 12
    card = state.hands[0][hand_pos]
    new_state = game.get_next_state(state, action)

    assert len(new_state.hands[0]) == HAND_SIZE
    assert new_state.current_player == 1
    # Discard pile for that color should have the card
    assert len(new_state.discard_piles[card.color]) > 0


def test_ascending_order():
    """Test that ascending order constraint is enforced."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    # Manually set up a state where player has specific cards
    from lost_cities.game.state import LCCard
    state.hands[0] = sorted([
        LCCard(0, 5), LCCard(0, 3), LCCard(1, 0),
        LCCard(1, 4), LCCard(2, 7), LCCard(3, 2),
        LCCard(4, 8), LCCard(4, 9),
    ])
    # Play a 5 to red expedition
    state.expeditions[0][0] = [LCCard(0, 5)]

    valid = game.get_valid_moves(state)

    # Find play actions for red (color 0)
    for hand_pos in range(len(state.hands[0])):
        card = state.hands[0][hand_pos]
        if card.color == 0:
            # Play to expedition: dest=0
            for draw_src in range(6):
                action = hand_pos * 12 + 0 * 6 + draw_src
                if card.value <= 5:
                    # 3 can't be played after 5 (not ascending)
                    assert valid[action] == 0 or draw_src == 0 and len(state.deck) == 0
                # value > 5 would be valid (but we don't have one of color 0 > 5 in this hand)


def test_cant_draw_from_own_discard():
    """Test that you can't draw from the pile you just discarded to."""
    game = LostCitiesGame()
    state = game.get_initial_state()

    valid = game.get_valid_moves(state)

    for action in range(96):
        if valid[action]:
            hand_pos = action // 12
            dest = (action % 12) // 6
            draw_src = action % 6
            if dest == 1:  # Discard
                card = state.hands[0][hand_pos]
                # draw_src should NOT be card.color + 1
                assert draw_src != card.color + 1


def test_scoring():
    """Test score calculation."""
    from lost_cities.game.state import LCCard
    state = LostCitiesState()

    # Empty expedition = 0 points
    assert state.compute_score(0) == 0

    # Expedition with wager + cards: (3+5+7 - 20) * 2 = -10
    state.expeditions[0][0] = [LCCard(0, 0), LCCard(0, 3), LCCard(0, 5), LCCard(0, 7)]
    assert state.compute_score(0) == (3 + 5 + 7 - 20) * 2

    # High-value expedition: (2+3+4+5+6+7+8+9+10 - 20) * 1 + 20 bonus = 34 + 20 = 54
    state.expeditions[0][1] = [LCCard(1, v) for v in range(2, 11)]
    green_score = (sum(range(2, 11)) - 20) * 1 + 20  # 9 cards = 8+ bonus
    expected = (3 + 5 + 7 - 20) * 2 + green_score
    assert state.compute_score(0) == expected


def test_full_game():
    """Test that a full game can be played without errors."""
    game = LostCitiesGame(seed=42)
    state = game.get_initial_state()
    moves = 0

    while not game.is_terminal(state):
        valid = game.get_valid_moves(state)
        assert np.sum(valid) > 0, f"No valid moves at turn {moves}"
        actions = np.where(valid > 0)[0]
        action = np.random.choice(actions)
        state = game.get_next_state(state, action)
        moves += 1

    # Game should end with scores
    score0 = state.compute_score(0)
    score1 = state.compute_score(1)
    reward = game.get_reward(state, 0)
    assert reward in [-1.0, 0.0, 1.0]
    print(f"Game finished in {moves} moves. Score: {score0} vs {score1}, reward: {reward}")


if __name__ == '__main__':
    test_initial_state()
    test_valid_moves()
    test_state_copy()
    test_canonical_form()
    test_to_tensor()
    test_play_to_expedition()
    test_discard_action()
    test_ascending_order()
    test_cant_draw_from_own_discard()
    test_scoring()
    test_full_game()
    print("All tests passed!")
