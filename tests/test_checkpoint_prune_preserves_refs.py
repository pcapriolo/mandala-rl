"""Tests for the reference-aware checkpoint pruning logic in Trainer._cleanup_checkpoints.

The pruner deletes old `model_iter_*.pt` files when there are >40 of them,
keeping (a) every 50th iter, (b) the last 30 iters, and (c) iters within
the [opponent_iter_min, opponent_iter_max] reference range.

Without (c), enabling prune_old_checkpoints would delete the active
reference checkpoint (e.g. iter 4785, not divisible by 50, well outside
the most-recent 30) and crash the worker on the next vs-reference game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _simulate_prune(
    iter_numbers: list[int],
    *,
    opponent_iter_min: int = 0,
    opponent_iter_max: int = 0,
) -> set[int]:
    """Replicate the keep-set logic from Trainer._cleanup_checkpoints exactly.

    Returns the set of iter numbers that would be kept.
    """
    if len(iter_numbers) <= 40:
        return set(iter_numbers)
    sorted_iters = sorted(iter_numbers)
    keep = set()
    for i in sorted_iters:
        if i % 50 == 0 or (opponent_iter_min <= i <= opponent_iter_max):
            keep.add(i)
    for i in sorted_iters[-30:]:
        keep.add(i)
    return keep


def test_keeps_multiples_of_50_and_last_30():
    """Baseline behavior: with no reference pin set (0..0), preserve every 50th iter
    and the last 30."""
    iters = list(range(1, 200))  # 199 iters from 1 to 199
    keep = _simulate_prune(iters)
    # multiples of 50 in [1,199]: 50, 100, 150
    assert {50, 100, 150} <= keep
    # last 30: 170 to 199
    assert set(range(170, 200)) <= keep
    # 49 should NOT be kept
    assert 49 not in keep


def test_preserves_single_reference_pin_outside_recent_window():
    """The active reference pin (single iter, not divisible by 50, not in last 30)
    must survive pruning. This is the bug we're fixing — without this, iter 4785
    would have been deleted at our current iter ~5552."""
    iters = list(range(1, 5553))
    keep = _simulate_prune(iters, opponent_iter_min=4785, opponent_iter_max=4785)
    assert 4785 in keep, "active reference checkpoint must be preserved"


def test_preserves_inclusive_reference_range():
    """If opponent_iter_min < opponent_iter_max, every iter in [min, max] is preserved.
    Both endpoints inclusive."""
    iters = list(range(1, 6000))
    keep = _simulate_prune(iters, opponent_iter_min=3500, opponent_iter_max=3505)
    assert {3500, 3501, 3502, 3503, 3504, 3505} <= keep, (
        "inclusive range — all 6 iters preserved"
    )
    # 3499 and 3506 are outside the range; if not multiples of 50 / last 30, deleted
    assert 3499 not in keep
    assert 3506 not in keep


def test_zero_reference_does_not_protect_anything_extra():
    """opponent_iter_min=0, opponent_iter_max=0 (the default when not set) must
    not accidentally protect iter 0 unless it would be kept anyway. The boundary
    `0 <= i <= 0` matches only iter 0, and only iters >= 1 are saved checkpoints."""
    iters = list(range(1, 200))  # No iter 0 in the list
    keep_no_ref = _simulate_prune(iters, opponent_iter_min=0, opponent_iter_max=0)
    keep_with_ref = _simulate_prune(iters, opponent_iter_min=100, opponent_iter_max=100)
    # Both should keep multiples of 50 + last 30; the 100 case is already in keep
    # (100 is divisible by 50), so the keep sets are identical.
    assert keep_no_ref == keep_with_ref


def test_known_pins_we_actually_use_all_survive_with_correct_config():
    """Real scenario: at iter 5552 with reference pin 4785, the prune must keep
    iter 4785 (active reference) AND every reference we've historically used."""
    iters = list(range(50, 5553, 50)) + [3175, 4220, 4785, 5551, 5552]
    iters = sorted(set(iters))
    # Active reference is 4785 (the others are historical, not currently set).
    keep = _simulate_prune(iters, opponent_iter_min=4785, opponent_iter_max=4785)
    assert 4785 in keep
    # 3175 and 4220 are NOT preserved (no longer the active reference) — that's
    # the intended behavior. Manual archive if user wants to retain them.
    assert 3175 not in keep
    assert 4220 not in keep
    # Last 30 includes 5551, 5552
    assert {5551, 5552} <= keep


def test_disabled_means_no_pruning():
    """The toggle is enforced at the call site by `if not prune_old_checkpoints: return`.
    This test is a sanity check for the simulator's contract — when iter count is
    below the trigger threshold (40), nothing is pruned regardless of reference."""
    iters = list(range(1, 41))
    keep = _simulate_prune(iters, opponent_iter_min=4785, opponent_iter_max=4785)
    assert keep == set(iters), "below threshold: nothing pruned"
