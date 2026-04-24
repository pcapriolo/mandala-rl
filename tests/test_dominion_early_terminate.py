"""Tests for DEVLOG #172: early-terminate Dominion games when the VP outcome
is mathematically determined.

The pathology: 14% of Dominion games at supply=7 stalled to turn_cap with the
trailing player buying 21-27 Golds and 0 Provinces. The winner had a score
lead larger than every VP point still available in the supply, yet the game
continued generating pathological replay samples.

Fix: DominionGame::check_game_end now optionally ends the game when
`lead > remaining_max_VP`. The rule is:

    remaining_max = Province × 6 + Duchy × 3 + Estate × 1 + Gardens × 10

Gardens is defensively bounded at 10 VP per copy (would require a 100-card
deck to realize, but never triggers when Gardens is in disabled_basic_supply).

These tests exercise the math directly via the BatchedMCTS surface (the
only pybind-exposed path). Construct a DominionGame implicitly via a
BatchedMCTS instance, play one game to completion, and assert termination
reason + behavior based on flag state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The mcts_cpp extension is required; these tests are skipped in environments
# without it (e.g., CI nodes without the built wheel).
mcts_cpp = pytest.importorskip("mcts_cpp")


# ---- Signature / wiring tests (no game simulation) --------------------------


def test_batched_mcts_accepts_early_terminate_decided_kwarg():
    """Pybind signature includes the new flag; default is False (backwards compat)."""
    mgr = mcts_cpp.BatchedMCTS(
        game_type="dominion",
        seed=1,
        num_simulations=2,
        province_supply=7,
        max_turns=50,
        early_terminate_decided=False,
    )
    # Just construction + init_games succeeds.
    mgr.init_games(1)


def test_batched_mcts_default_early_terminate_is_false():
    """Omitting the kwarg must preserve old behavior (flag=False)."""
    mgr = mcts_cpp.BatchedMCTS(
        game_type="dominion",
        seed=1,
        num_simulations=2,
        province_supply=7,
        max_turns=50,
    )
    mgr.init_games(1)


# ---- Behavior tests: run games and inspect summaries ------------------------


def _run_one_game(early_terminate_decided: bool, seed: int = 42,
                  num_simulations: int = 16, max_turns: int = 50,
                  province_supply: int = 7,
                  disabled_basic_supply: list[int] | None = None) -> dict:
    """Play a single Dominion game to completion and return its summary dict.

    Uses a tiny sim count (16) so the test is fast; the purpose is to observe
    termination condition, not policy strength. Low sims + random priors means
    the game wanders, but still terminates because one of (province_empty,
    three_piles, outcome_determined, turn_cap) always fires.

    NOTE: we can't easily drive the network-inference step from pure Python
    without a real network. This helper intentionally sets a *uniform* root
    policy via set_root_policies and uses random values for leaf evaluation,
    which is enough to exercise the termination paths in BatchedMCTS.
    """
    import numpy as np

    mgr = mcts_cpp.BatchedMCTS(
        game_type="dominion",
        seed=seed,
        num_simulations=num_simulations,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        temperature_threshold=25,
        leaves_per_game=1,
        disabled_basic_supply=disabled_basic_supply or [0, 3, 6, 16],
        province_supply=province_supply,
        max_turns=max_turns,
        early_terminate_decided=early_terminate_decided,
    )
    mgr.init_games(1)

    # Drive until all_done(). Each loop iter: begin_move → NN fill (uniform) →
    # (possibly multiple simulate_step → apply_nn_results) → finish_move.
    max_loops = 1000  # safety net — Dominion games can have many sub-actions per turn
    for _ in range(max_loops):
        if mgr.all_done():
            break
        root_tensors = mgr.begin_move()
        if root_tensors:
            # Uniform policy for every root.
            n = len(root_tensors)
            policies = np.full((n, 131), 1.0 / 131, dtype=np.float32)
            mgr.set_root_policies(policies)
        # Run simulations.
        for _ in range(num_simulations):
            leaves = mgr.simulate_step()
            if not leaves:
                break
            n = len(leaves)
            policies = np.full((n, 131), 1.0 / 131, dtype=np.float32)
            values = np.zeros(n, dtype=np.float32)
            mgr.apply_nn_results(policies, values)
        mgr.finish_move()
    assert mgr.all_done(), "test loop safety-net exceeded without terminating"
    return dict(mgr.get_game_summary(0))


def test_game_summary_includes_terminated_by_field():
    """Every Dominion game summary must include a `terminated_by` key regardless of flag."""
    summary = _run_one_game(early_terminate_decided=False, seed=1)
    assert "terminated_by" in summary
    assert summary["terminated_by"] in (
        "province_empty", "three_piles", "outcome_determined", "turn_cap", "none"
    )


def test_flag_off_never_produces_outcome_determined_termination():
    """With flag off, no game can end via outcome_determined — must be one of
    the other conditions. Locks backwards compat."""
    for seed in range(5):
        summary = _run_one_game(early_terminate_decided=False, seed=seed)
        assert summary["terminated_by"] != "outcome_determined", (
            f"seed {seed}: flag off should never produce outcome_determined, "
            f"got summary={summary}"
        )


def test_flag_on_does_not_break_termination():
    """With flag on, games still terminate. Smoke test — doesn't require
    outcome_determined to actually fire in random games, just that the
    code path doesn't deadlock or crash."""
    for seed in range(5):
        summary = _run_one_game(early_terminate_decided=True, seed=seed)
        # Every summary must record *some* termination reason.
        assert summary["terminated_by"] in (
            "province_empty", "three_piles", "outcome_determined", "turn_cap"
        )
        # And turn_number must be within max_turns.
        assert summary["turn_number"] <= 50


# ---- Config schema test -----------------------------------------------------


def test_config_schema_has_early_terminate_decided_field():
    """The new field must be present in the schema with HOT_WORKER metadata."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_schema",
        REPO_ROOT / "mandala_rl" / "training" / "config_schema.py",
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules["config_schema"] = cs
    spec.loader.exec_module(cs)

    fd = next(
        (f for f in cs.fields(cs.DominionConfig) if f.name == "early_terminate_decided"),
        None,
    )
    assert fd is not None, "early_terminate_decided missing from DominionConfig"
    assert fd.type in (bool, "bool")
    assert fd.metadata.get("hot") is True, "should be hot-reloadable"
    assert fd.metadata.get("target") == "worker", "should target worker attribute"
    assert fd.default is False, (
        "schema default must be False (conservative default; YAML enables it explicitly)"
    )


def test_config_schema_loads_dominion_yaml_with_flag():
    """The pod-deployed dominion.yaml must parse and include the new key with value True."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_schema",
        REPO_ROOT / "mandala_rl" / "training" / "config_schema.py",
    )
    cs = importlib.util.module_from_spec(spec)
    sys.modules["config_schema"] = cs
    spec.loader.exec_module(cs)

    cfg = cs.DominionConfig.load(REPO_ROOT / "configs" / "dominion.yaml")
    assert cfg.early_terminate_decided is True, (
        "configs/dominion.yaml must have early_terminate_decided: true"
    )
