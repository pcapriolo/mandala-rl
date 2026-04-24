"""Declarative config schema for Dominion training.

One dataclass is the single source of truth for every config key: its type,
its current-value default, and whether it's hot-reloadable (and where — worker
attribute or self.config dict).

Replaces the four whitelists (_TUNABLE_KEYS / _WORKER_TOP_KEYS / _CONFIG_TOP_KEYS
/ _CONFIG_NESTED_KEYS) that lived in trainer.py. Those whitelists had 10
silent-skip bugs: keys listed as top-level that actually lived nested under
`training:` or `evaluation:` in YAML. Here, every field declares exactly where
it goes — no duplication, no drift possible.

YAML is flat (no nested sections). YAML key name == schema field name ==
flat-config-dict key == worker-attribute name. No renames.

Strict loading: every schema field must be present in YAML. Dataclass defaults
are documentation; YAML is the authoritative live-state source.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union

import yaml


# Metadata shorthands. Each field's metadata declares hot-reloadability and
# propagation target. STATIC means restart-only (architecture, constructor
# args, optimizer state). HOT_WORKER means the value is re-read onto the
# SelfPlayWorker's attribute each iter. HOT_CONFIG means the value lives in
# self.config and is read via .get() each use.
STATIC: dict[str, Any] = {"hot": False}
HOT_WORKER: dict[str, Any] = {"hot": True, "target": "worker"}
HOT_CONFIG: dict[str, Any] = {"hot": True, "target": "config"}


@dataclass
class DominionConfig:
    # --- network (restart-only: architecture) ---
    input_channels: int = field(default=280, metadata=STATIC)
    num_actions: int = field(default=131, metadata=STATIC)
    num_res_blocks: int = field(default=10, metadata=STATIC)
    channels: int = field(default=128, metadata=STATIC)
    belief_size: int = field(default=31, metadata=STATIC)
    phase_aware_policy: bool = field(default=False, metadata=STATIC)
    factored_policy: bool = field(default=False, metadata=STATIC)

    # --- mcts ---
    mcts_simulations: int = field(default=800, metadata=STATIC)
    c_puct: float = field(default=1.5, metadata=HOT_WORKER)
    dirichlet_alpha: float = field(default=0.15, metadata=HOT_WORKER)
    dirichlet_epsilon: float = field(default=0.15, metadata=HOT_WORKER)
    action_explore_boost: float = field(default=0.0, metadata=HOT_WORKER)
    action_buy_force_rate: float = field(default=0.0, metadata=HOT_WORKER)
    action_play_force_rate: float = field(default=0.0, metadata=HOT_WORKER)
    mcts_leaf_eval_source: str = field(default="score", metadata=HOT_WORKER)

    # --- self-play ---
    games_per_iteration: int = field(default=100, metadata=STATIC)
    parallel_games: int = field(default=256, metadata=STATIC)
    leaves_per_game: int = field(default=8, metadata=STATIC)
    temperature: float = field(default=1.0, metadata=HOT_WORKER)
    temperature_threshold: int = field(default=25, metadata=HOT_WORKER)
    explore_epsilon: float = field(default=0.0, metadata=HOT_WORKER)
    opponent_diversity_ratio: float = field(default=0.2, metadata=HOT_CONFIG)
    opponent_iter_min: int = field(default=3600, metadata=HOT_CONFIG)
    opponent_iter_max: int = field(default=3600, metadata=HOT_CONFIG)

    # --- training ---
    num_iterations: int = field(default=3000, metadata=STATIC)
    batch_size: int = field(default=256, metadata=HOT_CONFIG)
    epochs_per_iteration: int = field(default=1, metadata=HOT_CONFIG)
    learning_rate: float = field(default=2.7e-5, metadata=STATIC)
    weight_decay: float = field(default=0.0001, metadata=STATIC)
    lr_milestones: list[int] = field(default_factory=list, metadata=STATIC)
    lr_gamma: float = field(default=0.3, metadata=STATIC)
    replay_buffer_size: int = field(default=100000, metadata=STATIC)
    min_buffer_for_training: int = field(default=10000, metadata=HOT_CONFIG)
    checkpoint_frequency: int = field(default=5, metadata=HOT_CONFIG)
    checkpoint_every_n_games: int = field(default=0, metadata=HOT_CONFIG)
    entropy_weight: float = field(default=0.01, metadata=HOT_CONFIG)
    policy_weight: float = field(default=1.0, metadata=HOT_CONFIG)
    max_discard_rate: float = field(default=0.0, metadata=HOT_CONFIG)
    seed_reinject_frequency: int = field(default=0, metadata=HOT_CONFIG)
    min_play_rate: float = field(default=0.0, metadata=STATIC)
    prune_old_checkpoints: bool = field(default=False, metadata=STATIC)

    # --- evaluation ---
    num_games: int = field(default=100, metadata=STATIC)
    win_threshold: float = field(default=0.55, metadata=STATIC)
    eval_frequency: int = field(default=5, metadata=HOT_CONFIG)
    eval_num_games: int = field(default=20, metadata=STATIC)
    eval_mcts_simulations: int = field(default=400, metadata=STATIC)
    eval_start_iteration: int = field(default=1, metadata=STATIC)

    # --- paths ---
    checkpoint_dir: str = field(
        default="/workspace/dominion_data/checkpoints", metadata=STATIC
    )
    replay_dir: str = field(
        default="/workspace/dominion_data/replays", metadata=STATIC
    )
    log_dir: str = field(default="/workspace/dominion_data/logs", metadata=STATIC)
    elo_file: str = field(
        default="/workspace/dominion_data/elo_ratings.json", metadata=STATIC
    )

    # --- curriculum (top-level in current YAML, all hot on the worker) ---
    disabled_basic_supply: list[int] = field(
        default_factory=lambda: [6, 16], metadata=HOT_WORKER
    )
    # Card IDs the reference opponent's policy must treat as illegal during
    # inference (buy-action logits forced to -inf before softmax). Unlike
    # disabled_basic_supply (game-level, both players), this is per-policy:
    # supply stays globally available; only the reference's selection is
    # constrained. DEVLOG #170. Default [] = behavior identical to before.
    opponent_disabled_supply: list[int] = field(
        default_factory=list, metadata=HOT_WORKER
    )
    province_supply: int = field(default=7, metadata=HOT_WORKER)
    max_action_cards: int = field(default=0, metadata=HOT_WORKER)
    forced_kingdom_cards: list[int] = field(
        default_factory=list, metadata=HOT_WORKER
    )
    big_money_force_rate: float = field(default=0.0, metadata=HOT_WORKER)
    draw_penalty: float = field(default=0.0, metadata=HOT_WORKER)
    drop_draws: bool = field(default=True, metadata=HOT_WORKER)
    max_turns: int = field(default=50, metadata=HOT_WORKER)
    # DEVLOG #172: end Dominion games early when the VP outcome is mathematically
    # determined (winner's VP lead > max remaining supply VP). Cuts off the 14%
    # of games stalling to turn_cap with a trailing player infinite-Gold-buying.
    # Hot-reloads onto the worker attribute; takes effect on next new games
    # (BatchedMCTS reads it at construction — games already in progress keep
    # their original flag value until this iter's self-play batch completes).
    early_terminate_decided: bool = field(default=False, metadata=HOT_WORKER)

    # --- misc ---
    device: str = field(default="auto", metadata=STATIC)
    seed: int = field(default=42, metadata=STATIC)
    save_replay_frequency: int = field(default=1, metadata=HOT_CONFIG)
    deploy_frequency: int = field(default=25, metadata=HOT_CONFIG)

    def __post_init__(self):
        if self.mcts_leaf_eval_source not in ("score", "value"):
            raise ValueError(
                f"mcts_leaf_eval_source must be 'score' or 'value', "
                f"got {self.mcts_leaf_eval_source!r}"
            )

    @classmethod
    def load(cls, path: str | Path) -> "DominionConfig":
        """Parse YAML and construct the dataclass. Strict: rejects unknown keys
        and requires every declared field to be present. Defaults are docs-only.
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top-level must be a mapping, got {type(raw)}")
        known = {fd.name for fd in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        missing = known - set(raw)
        if missing:
            raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
        return cls(**raw)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return every field as a flat dict. Keys match field names exactly
        (no renames, since YAML is already flat with matching names).
        """
        return {fd.name: getattr(self, fd.name) for fd in fields(self)}

    def reload_into(
        self, path: str | Path, config_dict: dict, worker: Any
    ) -> list[str]:
        """Re-read YAML and apply changes to hot-reloadable fields only.

        For `hot=True` fields: compares the YAML value to the current in-memory
        value (worker attribute or config dict entry per `target`). On mismatch,
        updates it and appends a human-readable change string.

        Raises ValueError if a hot field is missing from YAML (corruption), a
        required key is missing (schema/YAML drift), or an unknown key appears.

        Returns the list of change strings for the caller to log.
        """
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        changes: list[str] = []
        for fd in fields(self):
            if not fd.metadata.get("hot"):
                continue
            if fd.name not in raw:
                raise ValueError(
                    f"Hot field {fd.name!r} missing from {path} "
                    f"(YAML drifted from schema)"
                )
            new_val = raw[fd.name]
            target = fd.metadata["target"]
            if target == "worker":
                old_val = getattr(worker, fd.name)
                if old_val != new_val:
                    setattr(worker, fd.name, new_val)
                    changes.append(f"{fd.name} {old_val} → {new_val}")
            elif target == "config":
                old_val = config_dict.get(fd.name)
                if old_val != new_val:
                    config_dict[fd.name] = new_val
                    changes.append(f"config.{fd.name} {old_val} → {new_val}")
            else:
                raise ValueError(
                    f"Field {fd.name!r} has invalid target {target!r} "
                    f"(must be 'worker' or 'config')"
                )
        return changes
