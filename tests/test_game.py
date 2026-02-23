"""Basic tests for Mandala game engine."""
import numpy as np
from mandala_rl.game.engine import MandalaGame, NUM_ACTIONS
from mandala_rl.game.state import GameState


def test_initial_state():
    """Test initial game state setup."""
    game = MandalaGame()
    state = game.get_initial_state()

    assert len(state.hands[0]) == 6
    assert len(state.hands[1]) == 6
    assert state.current_player == 0
    assert not state.game_over
    assert state.phase == 0


def test_valid_moves():
    """Test move generation."""
    game = MandalaGame()
    state = game.get_initial_state()

    valid_moves = game.get_valid_moves(state)
    assert len(valid_moves) == NUM_ACTIONS
    assert np.sum(valid_moves) > 0


def test_state_copy():
    """Test state copying."""
    game = MandalaGame()
    state = game.get_initial_state()
    state_copy = state.copy()

    assert state is not state_copy
    assert state.current_player == state_copy.current_player
    assert len(state.hands[0]) == len(state_copy.hands[0])
    assert state.phase == state_copy.phase


def test_canonical_form():
    """Test canonical state representation."""
    game = MandalaGame()
    state = game.get_initial_state()

    canonical = state.get_canonical_form()
    assert canonical.current_player == 0

    state.current_player = 1
    canonical = state.get_canonical_form()
    assert canonical.current_player == 0


def test_tensor_shape():
    """Verify tensor has 137 channels."""
    game = MandalaGame()
    state = game.get_initial_state()
    tensor = state.to_tensor()
    assert tensor.shape == (137, 8, 8)


if __name__ == '__main__':
    test_initial_state()
    test_valid_moves()
    test_state_copy()
    test_canonical_form()
    test_tensor_shape()
    print("All tests passed!")
