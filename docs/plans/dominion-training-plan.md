# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 4 — Silver/Gold/Province + Copper/Estate/Duchy re-enabled, supply=7 |
| **Province Supply** | 7 |
| **Disabled** | Curse(6), Gardens(16), all action cards |
| **Strength signal** | Metric-based gates only (no win rate — self-play makes it meaningless) |
| **Last Updated** | 2026-04-19 |

### Recent changes
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

**Supply:** Gold, Silver, Copper, Estate, Duchy, Province — `province_supply: 7`
**Disabled:** Curse(6), Gardens(16), all action cards
**Goal:** Policy must preserve Phase 3's Province-buying mastery while correctly treating the re-enabled supply cards: ignore Copper-in-supply (anti-economy), ignore Estate-in-supply (dead 1-VP buy), treat Duchy as a potential marginal endgame VP grab. Starting deck stays 7 Copper + 3 Estate — the network has always seen those in hand; this transition is purely a supply-availability change.

**Why Curse and Gardens held:** neither has a natural buyer in this supply set. Curse has cost 0 and value -1 — only bought if forced by an attack card (none in this phase). Gardens (1 VP per 10 cards) rewards big-deck engines, which don't exist without action cards. Enabling them would add argmax noise in the policy head without any learning signal. They graduate with Phase 5 (Smithy) when engine-style play starts to matter.

**Config changes from Phase 3:**
```yaml
disabled_basic_supply: [6, 16]   # was [0, 3, 4, 6, 16]
# max_turns: 50 unchanged — bump to 70 via hot-reload only if >5% of games clip
```

All other hyperparameters unchanged (temperature_threshold, dirichlet_epsilon, entropy_weight, policy_weight, drop_draws, num_simulations, draw_penalty, opponent_diversity_ratio).

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

---

## Phase transition runbook

Use this for every supply/card-set graduation. Hot-reload handles `province_supply`, `max_turns`, `draw_penalty`, `big_money_force_rate` (see `trainer.py:210-213`); anything else requires a restart.

### Pre-flight

1. Confirm gates hold for 20+ consecutive iters via `scripts/check_phase_gates.py` (or by eyeballing `data/dominion/losses.jsonl` — the sync of the live pod file into the repo runs on the training commit cadence; for freshest data SSH directly).
2. Pull pod connection details from memory or ask the user — RunPod SSH port rotates on pod restart. Current: `ssh root@38.147.83.30 -p 26242 -i ~/.ssh/id_ed25519`.
3. Locate the **live** config on the pod — `python3 scripts/train.py --config configs/dominion.yaml` is relative to CWD. Find it via:
   ```
   ssh ... "readlink -f /proc/$(pgrep -f 'train.py.*dominion')/cwd"
   ```
   As of 2026-04-19 the live CWD is `/root/mandala-dom`, **not** `/workspace/mandala-rl`. Treat the workspace copy as stale; it is out of sync with the live training config.

### Deploy

1. Edit `configs/dominion.yaml` in the repo — change only the targeted keys.
2. Back up the pod file: `ssh ... "cp <live>/configs/dominion.yaml <live>/configs/dominion.yaml.bak_phase<N>_<YYYYMMDD>"`.
3. `scp` the local file to the same live path on the pod.
4. Verify the pod file with `grep -nE 'province_supply|disabled_basic_supply|max_turns'` — the values should match the local edit.
5. Tail training stdout at `/root/train_dom.log` (found via `/proc/<pid>/fd/1`); watch for `Config reload: <attr> <old> → <new>` at the next iter boundary. If the log says nothing, the change didn't land — re-check file path and hot-reload whitelist.

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
