"""Basic tests for Mandala game engine."""
import numpy as np
from mandala_rl.game.engine import MandalaGame
from mandala_rl.game.state import GameState


def test_initial_state():
    """Test initial game state setup."""
    game = MandalaGame()
    state = game.get_initial_state()

    # Check that cards are dealt
    assert len(state.player_hands[0]) == 6
    assert len(state.player_hands[1]) == 6
    assert state.current_player == 0
    assert not state.game_over


def test_valid_moves():
    """Test move generation."""
    game = MandalaGame()
    state = game.get_initial_state()

    valid_moves = game.get_valid_moves(state)
    assert len(valid_moves) > 0
    assert np.sum(valid_moves) > 0  # At least some valid moves


def test_state_copy():
    """Test state copying."""
    game = MandalaGame()
    state = game.get_initial_state()
    state_copy = state.copy()

    assert state is not state_copy
    assert state.current_player == state_copy.current_player
    assert len(state.player_hands[0]) == len(state_copy.player_hands[0])


def test_canonical_form():
    """Test canonical state representation."""
    game = MandalaGame()
    state = game.get_initial_state()

    # Player 0's perspective
    canonical = state.get_canonical_form()
    assert canonical.current_player == 0

    # Player 1's perspective
    state.current_player = 1
    canonical = state.get_canonical_form()
    assert canonical.current_player == 0


if __name__ == '__main__':
    test_initial_state()
    test_valid_moves()
    test_state_copy()
    test_canonical_form()
    print("All tests passed!")
