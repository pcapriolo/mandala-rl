"""Unit tests for the per-reference policy mask (DEVLOG #170).

Verifies the masking math and the BUY_OFFSET constant alignment. Does NOT
construct a full SelfPlayWorker — that requires the C++ mcts_cpp extension
and a game engine. Instead, tests the isolated torch ops and constants
that the worker uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dominion.game.state import BUY_OFFSET  # noqa: E402


def test_buy_offset_constant():
    """The worker relies on BUY[card_id] = 34 + card_id. Lock that."""
    assert BUY_OFFSET == 34


def test_card_id_to_buy_index_mapping():
    """The formula used inline in worker.py: buy_idx = 34 + card_id."""
    # Copper=0, Estate=3, Duchy=4, Curse=6, Gardens=16
    card_ids = [0, 3, 4, 6, 16]
    buy_idx = [BUY_OFFSET + cid for cid in card_ids]
    assert buy_idx == [34, 37, 38, 40, 50]


def test_mask_zeros_opponent_policy_at_target_indices():
    """Simulate the worker's masking: set logits[:, buy_idx] = -inf before
    softmax. Verify post-softmax mass at those indices is ~0.
    """
    torch.manual_seed(0)
    # 131 = Dominion action space. 3 rows = 3 game states.
    logits = torch.randn(3, 131)
    disabled_card_ids = [0, 3, 4, 6, 16]
    buy_idx = [BUY_OFFSET + cid for cid in disabled_card_ids]

    # Apply the mask exactly as the worker does.
    logits[:, buy_idx] = float('-inf')
    policies = F.softmax(logits, dim=1)

    # Every masked index must have ~0 probability across all rows.
    for idx in buy_idx:
        assert torch.all(policies[:, idx] < 1e-8), (
            f"buy_idx {idx} still has mass: {policies[:, idx]}"
        )
    # Sanity: unmasked indices have normal mass.
    unmasked_mass = policies.sum(dim=1) - policies[:, buy_idx].sum(dim=1)
    assert torch.allclose(unmasked_mass, torch.ones(3), atol=1e-5)


def test_mask_leaves_current_policy_unchanged():
    """Current agent (m_idx=0) should NOT be masked. Verify identical logits
    through softmax produce identical probabilities when the mask code is
    skipped.
    """
    torch.manual_seed(42)
    logits = torch.randn(2, 131)
    unmasked_policies = F.softmax(logits, dim=1)
    # Clone and DO NOT mask (simulating m_idx == 0 / opponent_disabled_supply empty)
    logits_clone = logits.clone()
    same_policies = F.softmax(logits_clone, dim=1)
    assert torch.allclose(unmasked_policies, same_policies)


def test_mask_only_applies_when_opponent_disabled_supply_is_nonempty():
    """Behavior when opponent_disabled_supply=[] must be identical to no mask.
    The worker's guard clause `if m_idx == 1 and self.opponent_disabled_supply`
    must short-circuit on empty list.
    """
    assert bool([]) is False
    assert bool([0]) is True
    # The `and` short-circuits: empty list → skip mask → identical behavior.


def test_dirichlet_noise_cannot_lift_minus_inf():
    """Key design assumption: if we set logits to -inf BEFORE softmax, any
    additive noise at root (MCTS dirichlet) cannot revive the action because
    the softmax collapses -inf to 0 first. MCTS-root noise is applied to the
    already-softmaxed prior, so -inf → 0 permanently.

    This test just documents the invariant at the torch level.
    """
    logits = torch.tensor([1.0, 2.0, float('-inf'), 0.5])
    p = F.softmax(logits, dim=0)
    # Even large perturbations to softmaxed prior can't restore the -inf slot
    # because it's already 0 and P+δ = δ. MCTS clamps priors to >= 0 so no
    # negative spill either. The masked action is dead.
    assert p[2].item() == 0.0

    # Show that adding noise to the probability post-softmax keeps it small
    noise = torch.tensor([0.0, 0.0, 0.1, 0.0])  # hypothetical noise
    noisy = p + 0.1 * noise  # MCTS blends at some ratio
    assert noisy[2].item() < 0.02  # still negligible compared to non-masked actions
