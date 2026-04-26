# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 5 — Silver/Gold/Province + Duchy + Smithy, Copper/Estate masked, supply=8 |
| **Province Supply** | 8 |
| **Kingdom** | `[21]` (Smithy only, forced via `forced_kingdom_cards`) |
| **Disabled** | Copper(0), Estate(3), Curse(6), Gardens(16) |
| **Reference** | `iter_4785` pin or `iter_6020` pin (check live YAML); per-reference mask `[0, 3, 4, 6, 16, 21]` — reference refuses Smithy |
| **Force rates** | `action_play_force_rate: 0.2` bootstrap per DEVLOG #175. `action_buy_force_rate: 0.0`. |
| **Strength signal** | Metric-based gates only; new metric: `action_plays > 0` (network learning to play Smithy) |
| **Last Updated** | 2026-04-25 |

### Recent changes
- 2026-04-25: **`action_play_force_rate: 0.0 → 0.2` (DEVLOG #175).** Bootstrap Smithy plays. Network was trained on Phase 0-4 with no action cards, so policy prior on PLAY[Smithy] is ≈0 by construction — Smithy gets bought (1.5% via dirichlet) but never played. Force-play hot-reload kicks in at ~20% of action-phase decisions where Smithy is in hand. Reversible, no code change.
- 2026-04-25 (earlier): **Phase 5 transition.** `max_action_cards: 0 → 1`, `forced_kingdom_cards: [] → [21]`, `opponent_disabled_supply` extended with Smithy id. Reference pin moved to a Phase-5-aware checkpoint. supply=8 retained.
- 2026-04-25: **`province_supply: 7 → 8` (DEVLOG #173).** Calibration test. After 116 iters of DEVLOG #172's `early_terminate_decided=true`, mean prov fell 3.36 → 3.27 (cleaner training signal but metric tradeoff: outcome_determined cuts off games before winner buys their last prov). Theoretical max under supply=7 is 3.5; we were at 96% of max. Bumping to supply=8 raises max to 4.0 and tests whether 3.36 was a *policy* limit or a *supply* limit. **If prov rises to 3.65-3.85**, policy is competent and we advance to action cards (Smithy / Phase 5). **If prov stays near 3.30-3.45**, policy is the limit and the loser-blind-spot (DEVLOG #172) is the bind regardless of supply size.
- 2026-04-25: **Pruned 370 of 484 stale checkpoints; freed 18.5 GB.** Disk hit 100% mid-deploy. Kept every 50th iter + 5 known reference pins (2200, 3175, 3600, 4220, 4785) + `model_latest.pt`. `prune_old_checkpoints: false` left in YAML for now — periodic manual prunes preferred over automatic. Watchdog restarted training cleanly from `model_latest.pt` (iter 5551).
- 2026-04-24: **DEVLOG #172 — Early-terminate Dominion games when VP outcome is determined.** Game-rule-level termination: end games when `|vp_lead| > supply_VP_remaining`. Targets the 14% stuck-at-turn-50 pathology (loser refuses Province, buys 21-27 Golds with $11 hands). C++ change requiring full restart and ~33-iter buffer rebuild. Termination distribution post-deploy: 71% province_empty / 19% outcome_determined / 10% turn_cap / 0% three_piles. Value loss trended down (0.25 → 0.22), value-head pred std improved (0.77 → 0.78), but mean prov fell from 3.36 → 3.27 because outcome_determined ends games before winner buys their last prov.
- 2026-04-24: **DEVLOG #171 — Dirichlet ε 0.15 → 0.30** to force current-agent exploration of newly-unmasked Duchy. 146-iter window: ~0.5-0.9% of games have Duchy buys, buyer winrate ~17% (vs 50% baseline → asymmetric signal flowing). Prov flat at 3.36, no change to ceiling — confirmed Duchy isn't the prov bottleneck.
- 2026-04-24: **DEVLOG #170 — Per-reference policy masking.** Stage 1 (full Phase-3 mask `[0, 3, 4, 6, 16]`) → Stage 2 (Duchy unmasked globally, reference still refuses it via per-reference policy mask). Asymmetric signal mechanism for breaking dead-card contamination in symmetric self-play.
- 2026-04-22: **Reference ladder step 3 — pin iter 3175 → 3600.** 524 iters post step-2, live agent matched 3175's window prov 3.09 (no improvement). Iter 3600 single-iter 3.24 prov / 67.0 argmax / 30.9 turns / 0.12 est — beats 3175 on prov (+0.06), argmax (+5.5), turns (-2.4); est marginal regression (+0.02). 5-iter window confirms 3598/3599/3600 = 3.20/3.30/3.24 prov sustained peak. See DEVLOG #166.
- 2026-04-21: **Reference ladder step 2 — pin iter 3075 → 3175.** 130 iters post step-1 swap, policy plateaued at avg_provinces 3.08 (gate 3.45). Selection criterion corrected from windowed mean to single-iter: the reference IS one checkpoint's weights, so its strength is its single-iter play. Among saved checkpoints, iter 3175 dominates single-iter (prov 3.18, argmax 61.5, est 0.10 — at Phase 4 gate), window 3.09 confirms stable peak. Pin retained (min==max). See DEVLOG #164.
- 2026-04-20: **Introduced Rule #7 — reference-play every phase.** Diagnostics confirmed that self-play alone converges to a fixed point below the prior phase's actual strength (Phase 4 stuck at ~45% mcts_province_argmax_pct vs Phase 3 peak of 58%, with avg_provinces 2.35 vs Phase 3 peak 3.48). More MCTS sims (800→1600) didn't help; swapping MCTS leaf eval from score head to value head didn't help. The bottleneck is genuinely that self-play training outcomes don't separate Province from Gold at mid-game $8+ — both "win sometimes" against similarly-confused self. Fix: every phase plays 20% of games vs the prior phase's peak checkpoint. Asymmetric outcomes teach the value head the Q-gap it was missing. Not a crutch (no forced actions). Retroactively applied to current Phase 4 against iter 2200 baseline. See DEVLOG #163.
- 2026-04-19: **Phase 3 → Phase 4 transition (enable Copper/Estate/Duchy in supply, single-variable).** Phase 3 gates held for 80 consecutive iters (iter 2067–2146): avg_provinces 3.46–3.50 (mechanical max 3.5), draw_rate 0.00, avg_turns 26–28. Re-enabling the three cards with real learning signal (Copper, Estate, Duchy) while holding Curse and Gardens disabled — neither has a natural buyer without attack or engine cards, so enabling them adds policy noise without signal. `disabled_basic_supply: [0, 3, 4, 6, 16] → [6, 16]`. `max_turns: 50` retained (Phase 4 expected turns 28–38; will bump to 70 via hot-reload only if clipping emerges). All other hyperparameters unchanged. Gates tightened to mastery criteria (avg_prov ≥ 3.45, avg_turns < 30). Starting deck still 7 Copper + 3 Estate regardless of supply config (`cpp/dominion_game.cpp:594-602`). See DEVLOG #159.
- 2026-04-19: **Phase 2 → Phase 3 transition (supply 5 → 7, single-variable).** Phase 2 gates saturated at iter 1754–1783: provinces/p = 2.5 (mechanical max), draw_rate = 0.0, avg_turns 18–20 (same mechanical-saturation pattern that retired Phase 1's turns-floor gate). Taking a strict single-variable supply step to restore Rule #2 compliance — card set, `max_turns: 50`, `draw_penalty`, `drop_draws` all unchanged. The old Phase 3 (bundled supply+VP-clutter) is now Phase 4. Smithy → Phase 5. Full Dominion → Phase 6. See DEVLOG #158.
- 2026-04-18: **Phase 1 → Phase 2 transition (supply 3 → 5).** Phase 1 gates held on MCTS % and coins, but `avg_turns < 13` was mechanically unreachable at supply=3 (games consistently 15–17 turns). Graduating with a smaller step than original plan (3→5, not 3→7) to reduce adaptation shock. Plan restructured: **Phase 2 now supply=5, Phase 3 now supply=7** (swapped from earlier 7/5 layout). Gates re-derived for both. `max_turns` 30 → 50 to accommodate longer supply=5 games. See DEVLOG #157.
- 2026-04-18: **Inserted new Phase 1 (supply=3, same cards as Phase 0).** VP-card enablement pushed to Phase 3. Smithy → Phase 4. Full Dominion → Phase 5. Motivation: isolate supply changes from card enablement per Rule #2. Phase 0 coins-wasted threshold relaxed to <3.0.
- 2026-04-17: **Collapsed Phase 0 to single stage (supply=1).** Removed stepped subphases (0a/0b/0c) and auto-graduation through supply=1→2→3. Dropped win-rate gate at every phase — symmetric self-play makes p0/p1 win rate a noisy ~50% signal that can't distinguish "learning" from "converging on the same equilibrium more confidently." Graduation between phases is now a human decision triggered when gate metrics hold. See DEVLOG #154.
- 2026-04-16: Fixed yaml.dump destroying config comments (DEVLOG #153).
- 2026-04-16: Config-driven curriculum graduation landed (DEVLOG #152) — then retired 2026-04-17 with the simplification above.
- 2026-04-16: Reverted temp_threshold and max_turns, kept draw_penalty=0 (DEVLOG #149).
- 2026-04-16: Fixed config passthrough bug in train.py — draw_penalty and max_turns were silently dropped for 117 iterations (DEVLOG #147).
- 2026-04-16: Removed 50-sim fast games, entropy 0.15→0.03, fixed policy_weight=1.0 (DEVLOG #145).
- 2026-04-15: Switched to pure self-play, diversity=0.0 (DEVLOG #144).
- 2026-04-11: Fresh start — Gold/Silver/1 Province, pure self-play (DEVLOG #141).

---

## Why no win-rate gate

In symmetric 2-player self-play, p0 vs p1 win rate is a function of the draw rate and symmetry, not strength:

- With zero draws and perfect symmetry, p0 win rate tends to 50% regardless of skill.
- With high draws, the "loser" (by some tiebreak) is whoever goes second — p0 win rate can show 14–17% despite no strength gap (observed iter 693).
- A model that prefers Gold over Province consistently will draw against itself forever at 50/50 win rate, yet clearly hasn't learned the phase.

What actually signals strength in Phase 0:
1. **MCTS province buy % when affordable** — does search *prefer* Province? If yes, the network has learned the terminal value.
2. **Coins wasted per game** — is economy being used to buy VP?
3. **Avg turns** — shorter games mean commitment to a winning line.
4. **Draw rate** — high draws mean neither side is executing a winning strategy. Tracked but not a gate (graduation from Phase 0 is defined by buying behavior, not draws).

**Future (out of scope here):** once the eval daemon is running, add Elo vs. a frozen reference checkpoint as the true strength signal. Self-play metrics tell us *what* the model does; Elo tells us *how well* it plays. Both are needed.

---

## Phase 0: Gold/Silver/Province (CURRENT)

**Supply:** Gold, Silver, Province — `province_supply: 1`
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards
**Goal:** Learn Province > Gold > Silver in buy priority when affordable.

**Config:**
```yaml
province_supply: 1
disabled_basic_supply: [0, 3, 4, 6, 16]
max_action_cards: 0
draw_penalty: 0.0
max_turns: 30
temperature_threshold: 25
opponent_diversity_ratio: 0.0  # Hard rule: pure self-play in Phase 0
entropy_weight: 0.03
policy_weight: 1.0
# 800 MCTS sims on all games
```

**Phase 0 graduation criteria (all must hold for 20 consecutive iterations):**
- MCTS province buy % > 90% (when Province is affordable, search prefers it)
- Avg coins wasted < 3.0 per game
- Avg turns < 17

**Mechanism:** metrics are computed and logged each iteration by the trainer. A human reviews them (dashboard / `losses.jsonl`) and bumps `province_supply` in `configs/dominion.yaml` when all gates hold — no auto-graduation.

---

## Phase 1: Bump Province Supply to 3

**Supply:** Gold, Silver, Province — `province_supply: 3`
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards
**Goal:** Transfer Phase 0's Province > Gold > Silver priority to a larger terminal state (3 provinces to accumulate).

**Config changes from Phase 0:**
```yaml
province_supply: 3
```

**Graduation criteria (all must hold for 20 consecutive iterations):**
- MCTS province buy % > 90% (when Province is affordable, search prefers it)
- Avg coins wasted < 3.0 per game
- Avg turns < 13

(Gates mirror Phase 0 — same behavioral signals apply with larger terminal state.)

---

## Phase 2: Supply=5

**Supply:** Gold, Silver, Province — `province_supply: 5`
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards (same as Phase 1)
**Goal:** Transfer Phase 1's Province > Gold > Silver priority to a larger terminal state (5 Provinces) with longer game horizon. Intermediate rung before introducing VP clutter.

**Config changes from Phase 1:**
```yaml
province_supply: 5
max_turns: 50
```

**Graduation criteria (all must hold for 20 consecutive iterations):**
- Province/player > 2.0  *(max possible 2.5 at supply=5; ≥2.0 means both sides buying decisively)*
- Avg game length 20–35 turns
- Draw rate < 5%

---

## Phase 3: Supply=7

**Supply:** Gold, Silver, Province — `province_supply: 7`
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards (same as Phase 2)
**Goal:** Transfer Phase 2's Province > Gold > Silver priority to a 7-Province terminal with longer horizon. Final pre-VP-clutter rung; isolates the supply scaling from the card-enablement change to come in Phase 4 (strict Rule #2).

**Config changes from Phase 2:**
```yaml
province_supply: 7
```

`max_turns` stays at 50 — avg_turns scaling has been ~4 turns per 2-supply bump (supply 3 → 11, supply 5 → 19), so supply=7 projects to ~22–26, well under the cap.

**Graduation criteria (all must hold for 20 consecutive iterations):**
- Province/player > 3.0  *(max possible 3.5 at supply=7; ≥3.0 means both sides buying decisively)*
- Draw rate < 5%
- Avg turns < 40  *(upper bound only — lower bound is mechanically unreachable when terminal saturates, same pattern as Phase 1/2 per DEVLOG #157)*

**Outcome:** Gates held for 80 consecutive iters (2067–2146). Saturated: avg_provinces 3.46–3.50, avg_turns 26–28, draw_rate 0.0. Graduated to Phase 4 on 2026-04-19 (DEVLOG #159).

---

## Phase 4: Re-enable Copper/Estate/Duchy (CURRENT)

**Supply (as of 2026-04-25):** Gold, Silver, Duchy, Province — `province_supply: 8`. Copper and Estate are masked via `disabled_basic_supply: [0, 3, 6, 16]` (Copper, Estate, Curse, Gardens). Duchy is supply-available; the reference (`iter_4785`) refuses it per-policy via DEVLOG #170. The province bump 7→8 is a calibration test (DEVLOG #173) and may be reverted depending on whether mean prov tracks the new 4.0 ceiling.
**Disabled:** Copper(0), Estate(3), Curse(6), Gardens(16), all action cards
**Goal:** Policy must preserve Phase 3's Province-buying mastery while correctly treating the re-enabled supply cards: ignore Copper-in-supply (anti-economy), ignore Estate-in-supply (dead 1-VP buy), treat Duchy as a potential marginal endgame VP grab. Starting deck stays 7 Copper + 3 Estate — the network has always seen those in hand; this transition is purely a supply-availability change.

**Why Curse and Gardens held:** neither has a natural buyer in this supply set. Curse has cost 0 and value -1 — only bought if forced by an attack card (none in this phase). Gardens (1 VP per 10 cards) rewards big-deck engines, which don't exist without action cards. Enabling them would add argmax noise in the policy head without any learning signal. They graduate with Phase 5 (Smithy) when engine-style play starts to matter.

**Config changes from Phase 3:**
```yaml
disabled_basic_supply: [6, 16]   # was [0, 3, 4, 6, 16]
# max_turns: 50 unchanged — bump to 70 via hot-reload only if >5% of games clip
# 2026-04-20 retroactive Rule #7 addition:
opponent_diversity_ratio: 0.2    # was 0.0 — reference-play vs iter 2200 (Phase 3 peak)
opponent_iter_min: 2200
opponent_iter_max: 2200
```

All other hyperparameters unchanged (temperature_threshold, dirichlet_epsilon, entropy_weight, policy_weight, drop_draws, num_simulations, draw_penalty).

**Phase 4 status (2026-04-20):** stuck in a self-play bad equilibrium at iter 2650. `avg_provinces` 2.35 (gate 3.45), `avg_estates` 0.60, `avg_copper` 0.85 (policy buying dead cards). 275+ iters since Phase 4 started; recovery never completed. Applying Rule #7 retroactively as the intervention — iter 2200 (Phase 3 peak) is the reference. Expected trajectory: ~30-50 iters of vs-reference losses teach the value head the Q-gap; `avg_provinces` recovers toward 3.45; `avg_estates` / `avg_copper` fall toward 0. If no recovery within 50 iters, fall back to Phase 3 mask revert (`disabled_basic_supply: [0, 3, 4, 6, 16]`) to regenerate economy before re-trying.

**Graduation criteria (all must hold for 20 consecutive iterations):**
- `avg_provinces/player ≥ 3.45`  *(no regression — Phase 3 median was 3.48; mechanical max 3.5)*
- `avg_turns < 30`  *(deck pollution from Copper/Estate must not slow the deck — Phase 3 baseline was 26–28)*
- `draw_rate < 0.05`
- `avg_estates/player < 0.1`  *(policy must ignore Estate-in-supply)*
- `avg_copper/player < 0.1`  *(policy must ignore Copper-in-supply)*

**Tracked but not blocking:** `avg_duchies/player`. Unclear whether optimal play includes Duchy (endgame grab) or pure Province racing wins. Log it; don't gate on it.

**Plan reshuffle:** Phase 5 now Smithy (unchanged). Phase 6 now full basic supply with Curse + Gardens re-enabled alongside engine cards. Full Dominion → Phase 7.

---

## Phase 5: Smithy

**Supply:** Full basic cards + Smithy, Province (`province_supply: 7`)
**Goal:** Learn draw engine basics. Smithy (+3 cards) is the simplest engine card — teaches that action cards can accelerate economy.

**Config changes from Phase 4:**
```yaml
max_action_cards: 1
forced_kingdom_cards: []  # TBD — may force Smithy
# Rule #7: reference-play against Phase 4 peak
opponent_diversity_ratio: 0.2
opponent_iter_min: <Phase 4 peak iter, set when Phase 4 graduates>
opponent_iter_max: <same>
```

**Graduation criteria:** TBD based on Phase 4 results.

---

## Phase 6: Re-enable Curse + Gardens

**Supply:** Full basic supply + Smithy, Province (`province_supply: 7` or `8` TBD). Re-enables Curse(6) and Gardens(16) which were held back through Phases 4–5 because they have no natural buyer without attack/engine cards.
**Goal:** Policy learns that Curse is always bad (even available in supply) and that Gardens becomes attractive once deck-growth via Smithy is in play.
**Config changes from Phase 5:**
```yaml
disabled_basic_supply: []
```
**Graduation criteria:** TBD based on Phase 5 results.

---

## Phase 7: Full Dominion

**Supply:** Standard 10-card kingdom, Province (`province_supply: 8`)
**Goal:** Competitive play across varied kingdoms.

**Config changes:** Standard Dominion rules. TBD.

---

## Rules

1. **Never skip phases.** Each phase builds on learned representations from the previous one.
2. **Never change province_supply and disabled_basic_supply simultaneously.** One variable at a time.
3. **Checkpoint before every phase transition.** Back up model + config.
4. **No weight surgery.** Bias nudges only. Seed data injection is the approved intervention for stuck priors (DEVLOG #137).
5. **Phase advancement is a human decision.** Gates are observable criteria in `losses.jsonl` / dashboard; the human edits `province_supply` in the YAML and the trainer picks it up via hot-reload on the next iteration.
6. **Monitor overtraining ratio.** Each iteration ~3,000 examples. Buffer 100K, 1 epoch = each example seen ~1.3x. Safe.
7. **Every phase runs reference-play against the prior phase's peak checkpoint** (see "Reference-play" section below). Self-play alone converges to a fixed point that can be below the prior phase's actual strength — vs-reference games provide an asymmetric, external "strength floor" independent of the self-play equilibrium. Not a crutch (agent isn't forced to take any action); an environmental pressure. Part of the standard phase config, not a separate variable under Rule #2.

---

## Reference-play (Rule #7)

### Motivation

Self-play training converges to a fixed point where the policy is trained on visit distributions produced by itself. When MCTS visits split (e.g., 50/25/25 across Province/Gold/End-buy at $8+), the policy learns that split as its prior, which then produces similar splits next iter — a stable equilibrium even when that equilibrium is game-theoretically wrong (e.g., Province strictly dominates at $8+ with no +Buy, but self-play lands at ~58% mcts_province_argmax_pct at Phase 3 peak, not the ~95% first principles would predict).

Diagnostics (2026-04-20):
- Supply=1 → 100% argmax. Supply=7 → 40-65% argmax. Clean monotonic drop. Structural, not a bug.
- More MCTS sims (800→1600) did NOT improve argmax% — search isn't the bottleneck.
- Swapping MCTS leaf eval from score head to value head did NOT improve argmax% — network's Q(Province) and Q(Gold) at mid-game $8+ are genuinely close in the weights.
- Conclusion: in self-play, Province-buying and Gold-stockpiling both "win sometimes" against a similarly-confused opponent. The training distribution doesn't push Q(Province) past Q(Gold) because the outcome data doesn't separate them cleanly.

Playing a fraction of games against a stronger frozen reference breaks this symmetry. The reference picks Province at $8+ consistently; the current agent that buys Estate at turn 1 loses those games clearly. Clean asymmetric outcomes flow into the training buffer. Value head learns Province-state > Gold-state when the opponent is disciplined.

### Selection of the reference checkpoint

The reference is the **peak** checkpoint from phase N-1, defined as:
- An iteration where the phase-graduation gates held continuously, AND
- Within that stable window, the iteration with the highest mastery metric (`avg_provinces` for supply-scaling phases, TBD for engine phases)
- For Phase N=4: iter **2200** (Phase 3 gate-holding window 2067-2146, avg_provinces 3.48 peak)

The reference is a real checkpoint file on disk. We do not hand-code Big Money bots or any other oracle — the reference is whatever our training previously produced at best.

### Config

Already in `configs/dominion.yaml` (lines 32-34):
```yaml
selfplay:
  opponent_diversity_ratio: 0.2         # 20% of games vs frozen reference
  opponent_iter_min: 2200               # pinned to phase N-1 peak
  opponent_iter_max: 2200               # single-checkpoint reference (not a band)
```

**Not hot-reloadable** (as of 2026-04-20): although these keys are listed in `trainer.py:_CONFIG_TOP_KEYS`, the hot-reload path at `trainer.py:267-274` checks `raw[cfg_key]` at YAML top level, but `opponent_diversity_ratio` / `opponent_iter_min` / `opponent_iter_max` live **nested under `selfplay:`** in the YAML. The check silently skips them. A config edit to these keys will NOT take effect until pod restart. If hot-reload for these is wanted later, the fix is trainer.py nested-key handling — not a config restructure. Retroactive application to the current Phase 4 run therefore requires a training restart with snapshot first.

### When to turn on

**Default for every phase transition from Phase 4 onward:** `opponent_diversity_ratio: 0.2` lit up concurrently with the phase config change. It's part of the standard phase entry configuration, not a separate decision.

**For in-flight regressions (like current Phase 4):** turn on retroactively against the prior phase peak. The 2026-04-20 intervention for Phase 4 is: set `opponent_diversity_ratio: 0.2`, `opponent_iter_min/max: 2200/2200`. No other config change.

### When to turn off

`opponent_diversity_ratio: 0.0` **only when** the agent demonstrably matches or exceeds the reference in current-phase metrics:
- `avg_provinces` matches or exceeds the reference's phase-N-1 peak
- `mcts_province_argmax_pct` matches or exceeds the reference's phase-N-1 baseline

Turn off via YAML edit → hot-reload. Asymmetric signal has done its job; agent can now stabilize on self-play.

### What NOT to do

- **Don't use a "stronger than we've ever trained" hand-coded baseline.** The reference must be from our own training lineage. If a hand-coded Big Money bot wins every game against a weakened agent, the buffer fills with trivial losses and value head learns nothing except "you're worse than a script." A reference-checkpoint-of-the-same-architecture provides gradient-useful signal.
- **Don't bump ratio above 0.3** without explicit reasoning. Too much vs-reference play starves self-play of strategic diversity. Agent starts optimizing specifically against reference's weaknesses rather than learning the phase.
- **Don't use the same reference forever.** When advancing to phase N+1, the reference updates to phase N peak. Each phase graduates its own successor baseline.

### Scaling across phases

The same mechanism works for every phase because the reference is drawn from our own training, not hand-coded. Phase 5's reference is Phase 4 peak. Phase 6's reference is Phase 5 peak. Phases 1-3 didn't use it historically (no regression pressure); new Phase 4+ rule applies forward.

### Cost

~20% of self-play compute goes to vs-reference games. These games still generate training examples (both positions recorded). Net cost: small; signal-to-noise improves.

---

## Phase transition runbook

Use this for every supply/card-set graduation. Hot-reload handles `province_supply`, `max_turns`, `draw_penalty`, `big_money_force_rate`, `disabled_basic_supply` (see `trainer.py:210-213`); anything else requires a restart. **Note (2026-04-20):** `opponent_diversity_ratio`, `opponent_iter_min`, `opponent_iter_max` are NOT hot-reloadable despite appearing in `_CONFIG_TOP_KEYS` — the hot-reload path doesn't traverse `selfplay:` nesting. Any change to Rule #7 reference-play config requires a restart.

### Pre-flight

1. Confirm gates hold for 20+ consecutive iters via `scripts/check_phase_gates.py` (or by eyeballing `data/dominion/losses.jsonl` — the sync of the live pod file into the repo runs on the training commit cadence; for freshest data SSH directly).
2. Pull pod connection details from memory or ask the user — RunPod SSH port rotates on pod restart. Current: `ssh root@38.147.83.30 -p 26242 -i ~/.ssh/id_ed25519`.
3. Locate the **live** config on the pod — `python3 scripts/train.py --config configs/dominion.yaml` is relative to CWD. Find it via:
   ```
   ssh ... "readlink -f /proc/$(pgrep -f 'train.py.*dominion')/cwd"
   ```
   As of 2026-04-19 the live CWD is `/root/mandala-dom`, **not** `/workspace/mandala-rl`. Treat the workspace copy as stale; it is out of sync with the live training config.

### Deploy

1. **Pin the reference checkpoint** (Rule #7). Identify the peak iter of phase N-1 (highest `avg_provinces` within the gates-holding window). Set in YAML:
   ```yaml
   opponent_diversity_ratio: 0.2
   opponent_iter_min: <peak_iter>
   opponent_iter_max: <peak_iter>
   ```
   Verify the checkpoint file exists on the pod: `ls /workspace/dominion_data/checkpoints/model_iter_<peak_iter>.pt`. If pruned, pick the nearest surviving checkpoint in the gates-holding window and use that.
2. Edit `configs/dominion.yaml` in the repo — change curriculum key(s) + reference-play config.
3. Back up the pod file: `ssh ... "cp <live>/configs/dominion.yaml <live>/configs/dominion.yaml.bak_phase<N>_<YYYYMMDD>"`.
4. `scp` the local file to the same live path on the pod.
5. Verify the pod file with `grep -nE 'province_supply|disabled_basic_supply|max_turns|opponent_diversity|opponent_iter'` — the values should match the local edit.
6. Tail training stdout at `/root/train_dom.log` (found via `/proc/<pid>/fd/1`); watch for `Config reload: <attr> <old> → <new>` at the next iter boundary. If the log says nothing, the change didn't land — re-check file path and hot-reload whitelist.

### Update project artifacts

Before or after the deploy (either order is fine since these don't affect the running trainer):
- Update top-of-file status table + recent-changes bullet in this plan doc.
- Append a new numbered entry to `DEVLOG.md` with evidence, rationale, steps, expectation, rollback.
- Rename the working branch to something concrete (e.g., `pcapriolo/dominion-supply-7`).

### Rollback

Revert the YAML value in both the local repo and the pod's live config — hot-reload swaps back on the next iter. Or restore the `.bak_phase<N>_<date>` backup on the pod. No checkpoint or buffer surgery. Mixed-supply trajectories self-flush from the 100K buffer within ~33 iterations at 100 games × ~19–25 turns.

---

## Monitoring plan (post-transition)

### What to watch

| Signal | Source | Healthy band | Alarm |
|---|---|---|---|
| `Config reload:` stdout line | `/root/train_dom.log` | Exactly once, at the iter boundary after SCP | Missing — config didn't land |
| `avg_provinces` | `losses.jsonl` | Phase 4: transient dip post-transition; recover to ≥3.45 within ~40–80 iters | Stuck <3.0 past iter +60 — policy not re-mastering Province buying with clutter |
| `avg_turns` | `losses.jsonl` | Phase 4: 28–32 expected during adaptation, then settle <30 | ≥45 sustained → risk of clipping at `max_turns: 50`; bump to 70 if >5% clip |
| `avg_estates` / `avg_copper` | `losses.jsonl` | <0.1/player once policy converges | Sustained >0.3 → policy bought dead cards; value head needs more training on clutter-aware states |
| `avg_duchies` | `losses.jsonl` | Any value — not a gate, just track |  |
| `draw_rate` | `losses.jsonl` | <5% (Phase 4 gate) | ≥10% for 3+ iters → regression; consider rollback |
| `std_len` vs `avg_len` | `losses.jsonl` | `std_len` << `max_turns - avg_len` | `avg_len + std_len` approaches 50 → clipping |
| `mcts_province_argmax_pct` | `losses.jsonl` | Phase 3+ at supply≥5: typically 50–75%. **Not a regression below 90%** — mechanical drop with supply size (DEVLOG #159). | Sharp sustained drop (>20pp below recent baseline) → real regression |
| `value_loss` | `losses.jsonl` | Short-term spike expected (new terminal), then re-converge | Monotonic climb 10+ iters → overtraining or curriculum shock |
| Heartbeat age | `dominion_monitor.sh` existing alert (>1800s stale) | — | Existing `scripts/dominion_monitor.sh` handles this |

### Where to watch

- **Live stdout:** `ssh ... "tail -f /root/train_dom.log"`. Best for first 1–2 iters after deploy to confirm hot-reload and first post-transition game quality line.
- **Metrics file:** `/workspace/dominion_data/losses.jsonl` on the pod; the repo copy at `data/dominion/losses.jsonl` syncs via the training commits (iter ~1783 at time of writing; will catch up as the bot commits post-transition data).
- **Dashboard:** `http://<pod>:5000` via the `start_observer.py` process already running (PID 2217583). Useful for eyeballing curves.
- **Existing watchdog:** `scripts/dominion_monitor.sh` runs every 10 min via launchd — handles process-alive, heartbeat freshness, disk, GPU. Don't duplicate that.

### Gate check

Use `scripts/check_phase_gates.py --phase <N>` to count consecutive iterations passing the gates for a given phase against the target of 20. It SSH-tails the pod `losses.jsonl`, so it always sees the freshest data — not the repo mirror. Run ad-hoc before considering the next phase.

### Gate thresholds for Phase 3 (supply=7, Silver/Gold/Province only) — HISTORICAL, already graduated

All must hold for 20 consecutive iters:
- `avg_provinces > 3.0`  *(max 3.5 at supply=7)*
- `draw_rate < 0.05`
- `avg_turns < 40`  *(upper-bound only)*

### Gate thresholds for Phase 4 (supply=7, +Copper/Estate/Duchy) — ACTIVE

All must hold for 20 consecutive iters:
- `avg_provinces/player ≥ 3.45`  *(Phase 3 median was 3.48; no regression)*
- `avg_turns < 30`  *(Phase 3 baseline 26–28; clutter must not slow the deck)*
- `draw_rate < 0.05`
- `avg_estates/player < 0.1`  *(policy ignores Estate-in-supply)*
- `avg_copper/player < 0.1`  *(policy ignores Copper-in-supply)*

**Tracked but not gated:** `avg_duchies/player` — optional strategic behavior.

### Watchdog cadence

- **First 3 iters after deploy:** actively watch stdout. Confirm `Config reload:` line, confirm `avg_provinces` begins climbing above 2.5.
- **Iters 4–20:** run `check_phase_gates.py` every ~30–60 min (3–6 iters per check). Any "fail" iter resets the consecutive-count.
- **Iters 20–40:** check once per ~2 hours; if 20 consecutive passes, flag readiness for Phase 4.
- **Rollback trigger:** any single iter with `draw_rate ≥ 0.10`, or 3 consecutive iters with `avg_turns ≥ 45`, or 10 consecutive iters with `avg_provinces ≤ 2.5` past the transition iter.
