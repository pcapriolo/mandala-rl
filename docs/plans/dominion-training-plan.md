# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 0 — Gold/Silver/Province |
| **Iteration** | Running (config drift — see DEVLOG #154) |
| **Province Supply** | Config: 1 (stepped curriculum). RunPod: still 3 (never synced). |
| **Key Metrics** | avg_turns 52-56, draw% ~49%, province >1/player — confirms supply=3 active on RunPod |
| **Last Updated** | 2026-04-17 |

### Notes
- 2026-04-17: **Config drift discovered.** RunPod never received DEVLOG #152 config changes (province_supply: 3→1, max_turns: 70→30). Sync script was never run. Also fixed sync script bug: province_supply was incorrectly listed as requiring restart (it's hot-reloadable). See DEVLOG #154.
- 2026-04-16: **Fixed yaml.dump destroying config comments.** Graduation write-back now uses regex line replacement instead of yaml.dump. Preserves all inline comments and formatting. See DEVLOG #153.
- 2026-04-16: **Config-driven curriculum graduation.** Graduation now writes province_supply and max_turns back to YAML config on disk. Config file is the single source of truth — hot-reload reads correct values with no exclusions or workarounds. See DEVLOG #152.
- 2026-04-16: **Switched to stepped auto-graduation curriculum.** supply=3 was stuck at Province%=47%, draw%=49% after 245 iters. Jump from supply=1→3 was too large. New approach: auto-graduate 1→2→3 with criteria checks in trainer. No seeding, no bias nudges — pure self-play only.
- 2026-04-16: **Reverted temp_threshold and max_turns, kept draw_penalty=0.** Lowering temp_threshold to 15 amplified the bad policy (province% crashed 45→7%). Restored temp_threshold=25 and max_turns=70. Only net change: draw_penalty=0.0.
- 2026-04-16: Fixed config passthrough bug in train.py — draw_penalty and max_turns were silently dropped for 117 iterations. See DEVLOG #147.
- 2026-04-16: Removed 50-sim fast games, reduced entropy 0.15→0.03, fixed policy_weight to 1.0. See DEVLOG #145.
- 2026-04-15: Switched to pure self-play (diversity=0.0). Opponent pool (iters 779-796) were Phase 1 models — wrong game. Metrics plateaued for 600+ iters. See DEVLOG #144.
- 2026-04-14: Phase 0 training healthy. Win rate trending up. Graduation target: waste <2 coins + MCTS province buy % >90%.

---

## Phase 0: Gold/Silver/Province (CURRENT)

**Supply:** Gold, Silver, Province — **stepped curriculum** (supply 1 → 2 → 3)
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards
**Goal:** Learn that Province > Gold > Silver in buy priority when affordable.

### Stepped Curriculum (auto-graduation)

The model auto-graduates through supply levels. Each step builds Province-buying
priors that transfer to the next. No seeding, no bias nudges — pure self-play only.

Graduation writes `province_supply` and `max_turns` back to `configs/dominion.yaml`
on disk. The config file is the single source of truth — hot-reload reads it naturally.

```
supply=1 (max_turns=30) → supply=2 (max_turns=50) → supply=3 (max_turns=70)
```

| Step | Supply | Why | Graduation Criteria | Est. Iters |
|------|--------|-----|---------------------|------------|
| 0a | 1 | Province = game-ender. Trivial to learn. | prov/player >= 0.9, draw% < 5%, 10 consec | ~20-30 |
| 0b | 2 | Must buy Province repeatedly. Reinforces habit. | prov/player >= 1.5, draw% < 10%, 15 consec | ~50-100 |
| 0c | 3 | Full Phase 0 target. Model arrives with strong priors. | Phase 0 graduation criteria below | ~100-200 |

**Config:**
```yaml
province_supply: 1             # Auto-graduates 1→2→3 via curriculum_steps
disabled_basic_supply: [0, 3, 4, 6, 16]
max_action_cards: 0
draw_penalty: 0.0
max_turns: 30                  # Tight for supply=1 (auto-adjusts on graduation)
temperature_threshold: 25
opponent_diversity_ratio: 0.0  # Pure self-play — no opponent diversity in Phase 0
entropy_weight: 0.03
policy_weight: 1.0
# All games at full MCTS sims (800)
```

**Phase 0 rule:** `opponent_diversity_ratio` must be 0.0. Pure self-play is correct for Phase 0.

**Phase 0 graduation criteria (supply=3, all must hold for 20 consecutive iterations):**
- Waste < 2 coins per game (efficient buying)
- MCTS province buy % > 90% (when Province is affordable, bot buys it)
- Draw rate < 5%
- Win rate > 52%

---

## Phase 1: Full VP Cards

**Supply:** Gold, Silver, Copper, Estate, Duchy, Province (supply=5)
**Disabled:** Action cards only
**Goal:** Learn Duchy as secondary VP source. Learn Copper/Estate are dead cards (green/copper pollution).

**Config changes from Phase 0:**
```yaml
disabled_basic_supply: []  # All basic cards enabled
province_supply: 5
```

**Graduation criteria:**
- Province/player > 1.5
- Duchy buying present (> 0.3/player)
- Copper buying < 0.1/player (learned to avoid)
- Draw rate < 15%
- Win rate > 52%

---

## Phase 2: More Provinces

**Supply:** Full basic cards, Province (supply=7)
**Disabled:** Action cards only
**Goal:** Deeper economy/VP tradeoff decisions. With 7 Provinces, games are longer and economy pacing matters more — build Gold engine first, then pivot to Province buying at the right moment.

**Config changes from Phase 1:**
```yaml
province_supply: 7
```

**Graduation criteria:**
- Province/player > 2.5
- Avg game length 25-45 turns
- Draw rate < 15%
- Win rate > 52%

---

## Phase 3: Smithy

**Supply:** Full basic cards + Smithy, Province (supply=7)
**Goal:** Learn draw engine basics. Smithy (+3 cards) is the simplest engine card — teaches the bot that action cards can accelerate economy.

**Config changes from Phase 2:**
```yaml
max_action_cards: 1
forced_kingdom_cards: []  # TBD — may force Smithy
```

**Graduation criteria:** TBD based on Phase 2 results.

---

## Phase 4: Full Dominion

**Supply:** Standard 10-card kingdom, Province (supply=8)
**Goal:** Competitive play across varied kingdoms.

**Config changes:** Standard Dominion rules. TBD.

---

## Rules

1. **Never skip phases.** Each phase builds on learned representations from the previous one.
2. **Never change province_supply and disabled_basic_supply simultaneously.** One variable at a time.
3. **Checkpoint before every phase transition.** Back up model + config.
4. **No weight surgery.** Bias nudges only. Seed data injection is the approved intervention for stuck priors (DEVLOG #137).
5. **Monitor overtraining ratio.** Each iteration ~3,000 examples. Buffer 100K, 1 epoch = each example seen ~1.3x. Safe.
