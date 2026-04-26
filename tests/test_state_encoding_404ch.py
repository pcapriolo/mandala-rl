"""Tests for DEVLOG #176: state-encoding upgrade 280 → 404 channels.

Adds 4 new 31-channel ranges that expose legal info the network was missing:
  Ch 280-310: my deck-only counts (face-down deck)
  Ch 311-341: my discard-only counts
  Ch 342-372: opp in-play counts (public)
  Ch 373-403: trash composition (public)

Without these, two strategically distinct states (e.g. "3 Golds on top of my
deck" vs "3 Golds in my discard") collapsed to the same input tensor —
state aliasing that prevented the network from learning nuanced action-card
play decisions (DEVLOG #175 follow-up).

Tests exercise the new encoding via the BatchedMCTS surface: construct a
manager, run one step, inspect the state tensor channels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

mcts_cpp = pytest.importorskip("mcts_cpp")


def _create_mgr_and_get_root_tensors(seed: int = 1, num_games: int = 1):
    """Spin up a 1-game BatchedMCTS, run begin_move, return root tensors."""
    import numpy as np
    mgr = mcts_cpp.BatchedMCTS(
        game_type="dominion",
        seed=seed,
        num_simulations=2,
        province_supply=8,
        max_action_cards=1,
        forced_kingdom_cards=[21],  # Smithy
        disabled_basic_supply=[0, 3, 6, 16],
        early_terminate_decided=False,
    )
    mgr.init_games(num_games)
    root_tensors = mgr.begin_move()
    return mgr, root_tensors


def test_tensor_has_404_channels():
    """The state tensor produced by C++ has the new channel count.
    Pre-fix: 280 channels. Post-fix: 404 channels."""
    mgr, root_tensors = _create_mgr_and_get_root_tensors()
    assert len(root_tensors) >= 1
    t = root_tensors[0]
    # Tensor shape is [channels, 8, 8] per DOM_TENSOR_SHAPE
    assert t.shape[0] == 404, f"expected 404 channels, got {t.shape[0]}"
    assert t.shape[1] == 8 and t.shape[2] == 8


def test_new_channels_are_within_expected_value_ranges():
    """All new-channel values (280-403) must be in [0, 1] like the rest of the
    encoding. Negative or >1 values would indicate a divisor bug or sign flip."""
    import numpy as np
    mgr, root_tensors = _create_mgr_and_get_root_tensors()
    t = np.asarray(root_tensors[0])
    new_block = t[280:404, :, :]
    assert new_block.min() >= 0.0, f"negative value found: {new_block.min()}"
    assert new_block.max() <= 1.0, f"value > 1.0 found: {new_block.max()}"


def test_starting_deck_encoded_in_my_deck_only_channel():
    """Starting deck = 7 Copper + 3 Estate. After draw of 5, hand has some,
    deck has the rest. My deck-only (Ch 280-310) should have non-zero values
    for whatever's in face-down deck. Combined with hand (Ch 31-61), they
    should sum to ≈ full deck (Ch 0-30)."""
    import numpy as np
    mgr, root_tensors = _create_mgr_and_get_root_tensors(seed=42)
    t = np.asarray(root_tensors[0])
    # Channel value at any of the 64 spatial slots (set_channel fills uniformly)
    full_deck = t[0:31, 0, 0]      # full deck composition / 12
    my_hand = t[31:62, 0, 0]       # hand only / 10
    my_in_play = t[62:93, 0, 0]    # in-play only / 5
    my_deck = t[280:311, 0, 0]     # NEW: deck only / 12
    my_discard = t[311:342, 0, 0]  # NEW: discard only / 12
    # Reconstruct counts (each channel was divided by its divisor)
    full_counts = (full_deck * 12).round()
    hand_counts = (my_hand * 10).round()
    in_play_counts = (my_in_play * 5).round()
    deck_counts = (my_deck * 12).round()
    discard_counts = (my_discard * 12).round()
    # Invariant: full = hand + in_play + deck + discard for every card type
    reconstructed = hand_counts + in_play_counts + deck_counts + discard_counts
    np.testing.assert_array_equal(
        full_counts, reconstructed,
        err_msg="my full-deck channel does not equal sum of hand+in_play+deck+discard"
    )


def test_starting_state_no_in_play_or_discard_or_trash():
    """At game start, in-play and discard are empty for both players, and trash
    is empty. Channels for those should be all zero."""
    import numpy as np
    mgr, root_tensors = _create_mgr_and_get_root_tensors(seed=7)
    t = np.asarray(root_tensors[0])
    # My in-play (existing Ch 62-92) — empty at start
    assert t[62:93, :, :].sum() == 0, "my in-play should be empty at start"
    # My discard-only (NEW Ch 311-341) — empty at start
    assert t[311:342, :, :].sum() == 0, "my discard should be empty at start"
    # Opp in-play (NEW Ch 342-372) — empty at start
    assert t[342:373, :, :].sum() == 0, "opp in-play should be empty at start"
    # Trash (NEW Ch 373-403) — empty at start
    assert t[373:404, :, :].sum() == 0, "trash should be empty at start"


def test_my_deck_only_starting_state_correct():
    """Starting deck: 7 Copper + 3 Estate, then draw 5 cards. Whatever's NOT in
    hand stays in the face-down deck. Total hand+deck = 10. After draw, hand
    is some subset and deck-only Ch 280-310 reflects the rest."""
    import numpy as np
    mgr, root_tensors = _create_mgr_and_get_root_tensors(seed=123)
    t = np.asarray(root_tensors[0])
    hand_counts = (t[31:62, 0, 0] * 10).round()
    deck_counts = (t[280:311, 0, 0] * 12).round()
    # Total cards in hand + deck = 10 (starting deck size)
    assert hand_counts.sum() + deck_counts.sum() == 10, (
        f"hand+deck should equal starting deck of 10 cards, got "
        f"hand_total={hand_counts.sum()} deck_total={deck_counts.sum()}"
    )
    # All cards are Copper (id=0) or Estate (id=3)
    nonzero_indices = set(np.flatnonzero(hand_counts + deck_counts).tolist())
    assert nonzero_indices.issubset({0, 3}), (
        f"only Copper/Estate expected at start, got cards at indices {nonzero_indices}"
    )


def test_config_schema_default_is_404():
    """Schema default for input_channels matches DOM_TENSOR_CHANNELS (404)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_schema",
        REPO_ROOT / "mandala_rl" / "training" / "config_schema.py",
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules["config_schema"] = cs
    spec.loader.exec_module(cs)
    fd = next(
        (f for f in cs.fields(cs.DominionConfig) if f.name == "input_channels"),
        None,
    )
    assert fd is not None
    assert fd.default == 404, (
        f"schema default {fd.default} ≠ DOM_TENSOR_CHANNELS=404"
    )


def test_yaml_input_channels_is_404():
    """Pod-deployed YAML must specify the new channel count."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_schema",
        REPO_ROOT / "mandala_rl" / "training" / "config_schema.py",
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules["config_schema"] = cs
    spec.loader.exec_module(cs)
    cfg = cs.DominionConfig.load(REPO_ROOT / "configs" / "dominion.yaml")
    assert cfg.input_channels == 404, (
        f"configs/dominion.yaml has input_channels={cfg.input_channels}, expected 404"
    )
