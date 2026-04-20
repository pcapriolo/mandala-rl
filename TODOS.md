# TODOs

Deferred work captured during code reviews. Each entry: what, why, pros/cons, context, dependencies.

---

## 1. Trainer unit tests

**What:** Add a test file `tests/test_trainer.py` that exercises the core Trainer paths.

**Why:** `mandala_rl/training/trainer.py` is ~1200 lines with zero test coverage today. Recent changes added hot-reload logic for curriculum keys and a warmup gate for the replay buffer (DEVLOG #160) — both landed with manual verification only. The next bug in this area will be silent.

**Pros:**
- Catches regressions in hot-reload whitelist drift (e.g. new YAML key added but forgotten in `_WORKER_TOP_KEYS` / `_CONFIG_TOP_KEYS`).
- Catches warmup-gate math errors (`max(config_min, warmup_target)`).
- Catches checkpoint roundtrip bugs for `_warmup_target`.

**Cons:**
- Needs ~200 lines of mock infrastructure (mock `SelfPlayWorker`, `ReplayBuffer`, file I/O).
- Trainer has many integration points; tests will need careful mocking boundaries.

**Context:** Drive 2-3 iterations against mocks and assert: (a) hot-reload updates worker attrs on YAML change, (b) warmup gate skips `_train_network` until `len(buffer) >= warmup_target`, (c) save/load roundtrip preserves `_warmup_target`. Match existing test style at `tests/test_dominion.py`. Estimated ~15 minutes of Claude Code time.

**Depends on / blocked by:** None. Can land any time after Phase 4 ships.

---

## 2. YAML schema validation in hot-reload

**What:** Warn on unknown top-level YAML keys during `_hot_reload_config` so typos fail loud.

**Why:** Today, a typo like `batch_sze: 1024` in `configs/dominion.yaml` is silently ignored — `raw.get('batch_sze')` returns the typo value but `_CONFIG_TOP_KEYS` doesn't reference it, so nothing happens. The user thinks the setting took effect. This pre-existed the hot-reload expansion; DEVLOG #160 widened the attack surface by adding more keys.

**Pros:**
- Typos become immediate loud warnings instead of silent no-ops.
- Cheap: a simple allowlist check (`set(raw.keys()) - known_keys`) during hot-reload.

**Cons:**
- Must be careful not to flag YAML keys consumed elsewhere (e.g., `network:`, `mcts:`, `selfplay:`, `training:` subsections that live under nested dicts and aren't in the flat top-level whitelist).

**Context:** Build the known-keys set from: all section dicts in YAML (e.g. `mcts`, `selfplay`, `training`, `paths`, `evaluation`, `network`) plus `_WORKER_TOP_KEYS` + `_CONFIG_TOP_KEYS` + the architecture/path keys that are read at init. Log unknown keys as a WARNING once per iteration (or only when they appear/change).

**Depends on / blocked by:** None.

---

## 3. Hot-reload weight_decay

**What:** Make `weight_decay` changes in YAML take effect without a restart.

**Why:** Today, `weight_decay` is baked into `AdamW(...)` at trainer init. YAML edits silently no-op until the process restarts. This was explicitly excluded from the DEVLOG #160 expansion to keep scope tight — but it means a weight-decay tweak forces a full restart.

**Pros:**
- Consistent with the "all training tunable config is hot-reloadable" goal.
- One extra line per iteration: `for pg in self.optimizer.param_groups: pg['weight_decay'] = new_val`.

**Cons:**
- Weight decay is a knob that rarely changes at runtime. Low practical value.
- Adds one more key to maintain in the hot-reload whitelist.

**Context:** When `training.weight_decay` changes in YAML, compare against current `self.optimizer.param_groups[0]['weight_decay']`. If different, update every param_group and log the change. Add `'weight_decay'` handling to `_hot_reload_config` alongside the existing `_CONFIG_TOP_KEYS` loop.

**Depends on / blocked by:** None.
