"""Tests for mandala_rl.training.config_schema.

13 cases covering: golden-diff identity, strict load validation, hot-reload
propagation to both worker-attr and config-dict targets, and explicit
regressions for the three silent-skip bugs hit during the 2026-04-22 session
(`checkpoint_frequency`, `entropy_weight`, `mcts_leaf_eval_source`).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import yaml

# Import the schema module directly to avoid triggering the mandala_rl
# package __init__.py, which imports the C++ mcts extension.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "config_schema", REPO_ROOT / "mandala_rl" / "training" / "config_schema.py"
)
cs = importlib.util.module_from_spec(spec)
sys.modules["config_schema"] = cs
spec.loader.exec_module(cs)

DominionConfig = cs.DominionConfig
fields = cs.fields

DOMINION_YAML = REPO_ROOT / "configs" / "dominion.yaml"


# --- fixtures -----------------------------------------------------------------


class MockWorker:
    """Stand-in for SelfPlayWorker that accepts arbitrary attribute writes."""

    pass


def _write_yaml(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Copy the current dominion.yaml into a tmp path, optionally mutating keys."""
    raw = yaml.safe_load(DOMINION_YAML.read_text()) or {}
    if overrides:
        raw.update(overrides)
    out = tmp_path / "dominion.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


def _make_worker_from_schema(schema: "DominionConfig") -> MockWorker:
    """Build a MockWorker with every hot-worker attribute set to the schema's value."""
    w = MockWorker()
    for fd in fields(DominionConfig):
        if fd.metadata.get("hot") and fd.metadata.get("target") == "worker":
            setattr(w, fd.name, getattr(schema, fd.name))
    return w


# --- core correctness ---------------------------------------------------------


def test_01_golden_flat_dict_identity():
    """Loading the current YAML via schema gives a dict whose keys and values
    survive a round-trip via YAML (the authoritative form).
    """
    schema = DominionConfig.load(DOMINION_YAML)
    flat = schema.to_flat_dict()
    raw = yaml.safe_load(DOMINION_YAML.read_text())
    # Every raw YAML key appears with identical value.
    for k, v in raw.items():
        assert k in flat, f"YAML key {k} missing from flat dict"
        assert flat[k] == v, f"value mismatch on {k}: YAML={v!r} flat={flat[k]!r}"
    # Flat dict has no extras beyond the schema.
    assert set(flat) == set(raw), (
        f"flat-dict keys diverge from YAML keys: "
        f"extra={set(flat) - set(raw)} missing={set(raw) - set(flat)}"
    )


def test_02_no_renames_yaml_key_equals_schema_field():
    """Schema field names equal YAML key names exactly. Verifies there are no
    surprise renames — the flat dict is transparent.
    """
    raw = yaml.safe_load(DOMINION_YAML.read_text())
    schema_field_names = {fd.name for fd in fields(DominionConfig)}
    assert set(raw) == schema_field_names, (
        f"YAML keys diverge from schema fields: "
        f"YAML_only={set(raw) - schema_field_names} "
        f"schema_only={schema_field_names - set(raw)}"
    )


# --- load-time validation -----------------------------------------------------


def test_03_unknown_key_raises(tmp_path):
    """A YAML with an extra top-level key raises ValueError at load()."""
    path = _write_yaml(tmp_path, overrides={"bogus_key_xyz": 42})
    with pytest.raises(ValueError, match="Unknown keys"):
        DominionConfig.load(path)


def test_04_missing_key_raises_strict_mode(tmp_path):
    """Strict mode: every schema field must be in YAML. Dataclass defaults
    are docs-only. This is the bug class the old system silently hit.
    """
    raw = yaml.safe_load(DOMINION_YAML.read_text())
    del raw["entropy_weight"]  # any field works
    path = tmp_path / "dominion.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Missing keys"):
        DominionConfig.load(path)


def test_05_literal_validation_via_post_init(tmp_path):
    """mcts_leaf_eval_source must be 'score' or 'value' at load time."""
    path = _write_yaml(tmp_path, overrides={"mcts_leaf_eval_source": "potato"})
    with pytest.raises(ValueError, match="mcts_leaf_eval_source"):
        DominionConfig.load(path)


# --- hot-reload propagation ---------------------------------------------------


def test_06_reload_worker_target_mutates_worker(tmp_path):
    """A worker-target field that changes in YAML mutates the worker attribute."""
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    assert worker.dirichlet_epsilon == 0.15

    # Edit YAML, call reload_into.
    new_path = _write_yaml(tmp_path, overrides={"dirichlet_epsilon": 0.25})
    config = schema.to_flat_dict()
    changes = schema.reload_into(new_path, config, worker)

    assert worker.dirichlet_epsilon == 0.25
    assert any("dirichlet_epsilon" in c for c in changes)


def test_07_reload_config_target_mutates_config(tmp_path):
    """A config-target field that changes in YAML mutates the config dict."""
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    baseline = config["entropy_weight"]
    new_val = baseline + 0.01

    new_path = _write_yaml(tmp_path, overrides={"entropy_weight": new_val})
    changes = schema.reload_into(new_path, config, worker)

    assert config["entropy_weight"] == new_val
    assert any("entropy_weight" in c for c in changes)


def test_08_reload_static_field_does_not_propagate(tmp_path):
    """Static (restart-only) fields are not written to worker or config,
    even if they differ in the YAML. Silent no-op for those — correct,
    since the value can't take effect at runtime.
    """
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    original_channels = config["channels"]

    new_path = _write_yaml(tmp_path, overrides={"channels": 9999})
    changes = schema.reload_into(new_path, config, worker)

    assert config["channels"] == original_channels  # unchanged
    assert not any("channels" in c for c in changes)


def test_09_missing_hot_field_at_reload_raises(tmp_path):
    """If a hot-reloadable field is missing from YAML at reload time, raise
    loudly — the live process shouldn't silently continue with stale values.
    """
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()

    raw = yaml.safe_load(DOMINION_YAML.read_text())
    del raw["dirichlet_epsilon"]  # any hot field works
    bad_path = tmp_path / "dominion.yaml"
    bad_path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="dirichlet_epsilon"):
        schema.reload_into(bad_path, config, worker)


def test_10_no_changes_returns_empty_list(tmp_path):
    """Reloading an identical YAML produces zero changes."""
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()

    changes = schema.reload_into(DOMINION_YAML, config, worker)
    assert changes == []


# --- regression tests: the three silent-skip bugs from 2026-04-22 -------------


def test_11_regression_checkpoint_frequency_hot_reloads(tmp_path):
    """REGRESSION: DEVLOG #165 — `checkpoint_frequency` was in _CONFIG_TOP_KEYS
    expecting top-level, but lived nested under `training:` in the old YAML.
    Silent skip. This test locks the fix: changing it now fires a reload.
    """
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    assert config["checkpoint_frequency"] == 5

    new_path = _write_yaml(tmp_path, overrides={"checkpoint_frequency": 10})
    changes = schema.reload_into(new_path, config, worker)

    assert config["checkpoint_frequency"] == 10
    assert any("checkpoint_frequency" in c for c in changes)


def test_12_regression_entropy_weight_hot_reloads(tmp_path):
    """REGRESSION: DEVLOG #168 — `entropy_weight` had the same nested-key bug.
    """
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    baseline = config["entropy_weight"]
    new_val = baseline + 0.01

    new_path = _write_yaml(tmp_path, overrides={"entropy_weight": new_val})
    changes = schema.reload_into(new_path, config, worker)

    assert config["entropy_weight"] == new_val
    assert any("entropy_weight" in c for c in changes)


def test_14_opponent_disabled_supply_field_hot_reloads(tmp_path):
    """DEVLOG #170: new field `opponent_disabled_supply` (list[int]) must load,
    default to [], and hot-reload onto the worker when changed.
    """
    # Drift-tolerant: live YAML may carry deployed values (Phase 5+ has Smithy in
    # the mask). Verify the field loads as list[int] and hot-reloads to a new value
    # regardless of baseline.
    schema = DominionConfig.load(DOMINION_YAML)
    assert isinstance(schema.opponent_disabled_supply, list)

    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    assert isinstance(worker.opponent_disabled_supply, list)
    baseline = list(worker.opponent_disabled_supply)
    new_value = [0] if baseline != [0] else [0, 3]

    new_path = _write_yaml(tmp_path, overrides={"opponent_disabled_supply": new_value})
    changes = schema.reload_into(new_path, config, worker)

    assert worker.opponent_disabled_supply == new_value
    assert any("opponent_disabled_supply" in c for c in changes)


def test_13_regression_leaf_eval_source_hot_reloads(tmp_path):
    """REGRESSION: `mcts_leaf_eval_source` was missing from every whitelist
    entirely — unreachable as a hot-reload target. This test locks that it
    now propagates to the worker attribute.
    """
    schema = DominionConfig.load(DOMINION_YAML)
    worker = _make_worker_from_schema(schema)
    config = schema.to_flat_dict()
    assert worker.mcts_leaf_eval_source == "score"

    new_path = _write_yaml(tmp_path, overrides={"mcts_leaf_eval_source": "value"})
    changes = schema.reload_into(new_path, config, worker)

    assert worker.mcts_leaf_eval_source == "value"
    assert any("mcts_leaf_eval_source" in c for c in changes)
