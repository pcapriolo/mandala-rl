# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 2 — Gold/Silver/Province, supply=5 |
| **Province Supply** | 5 |
| **Strength signal** | Metric-based gates only (no win rate — self-play makes it meaningless) |
| **Last Updated** | 2026-04-18 |

### Recent changes
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

## Phase 2: Supply=5 (CURRENT)

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

## Phase 3: Full VP Cards, Supply=7

**Supply:** Gold, Silver, Copper, Estate, Duchy, Province (`province_supply: 7`)
**Disabled:** Action cards only
**Goal:** Learn Duchy as secondary VP source. Learn Copper/Estate are dead cards (green/copper pollution), now with a larger horizon so the engine can build Gold first before committing to VP buying.

**Config changes from Phase 2:**
```yaml
disabled_basic_supply: []
province_supply: 7
max_turns: 70
```

Note: This transition changes both `province_supply` (5→7) and `disabled_basic_supply` simultaneously — a knowing exception to Rule #2. Raising supply while enabling VP clutter gives the model enough horizon to absorb the new cards (Copper/Estate/Duchy) without panic-buying VP on a tight terminal.

**Graduation criteria:**
- Province/player > 2.5
- Duchy buying present (> 0.3/player)
- Copper buying < 0.1/player
- Draw rate < 5%

---

## Phase 4: Smithy

**Supply:** Full basic cards + Smithy, Province (`province_supply: 7`)
**Goal:** Learn draw engine basics. Smithy (+3 cards) is the simplest engine card — teaches that action cards can accelerate economy.

**Config changes from Phase 3:**
```yaml
province_supply: 7
max_action_cards: 1
forced_kingdom_cards: []  # TBD — may force Smithy
```

**Graduation criteria:** TBD based on Phase 3 results.

---

## Phase 5: Full Dominion

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
