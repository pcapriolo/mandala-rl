# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 0 — Gold/Silver/Province |
| **Iteration** | ~204 (run 6) |
| **Province Supply** | 3 |
| **Key Metrics** | Recovering from collapse — Province%=~47%, draw%=~49%, waste=0 |
| **Last Updated** | 2026-04-16 |

### Notes
- 2026-04-16: **Reverted temp_threshold and max_turns, kept draw_penalty=0.** Lowering temp_threshold to 15 amplified the bad policy (province% crashed 45→7%). Restored temp_threshold=25 and max_turns=70. Only change from original: draw_penalty=0.0 (symmetric penalty confirmed harmful). Buffer needs ~30 iters to recover. See DEVLOG #149.
- 2026-04-16: Removed draw_penalty and max_turns, lowered temp_threshold. See DEVLOG #148. **Reverted — see above.**
- 2026-04-16: Fixed config passthrough bug in train.py — draw_penalty and max_turns were silently dropped for 117 iterations. See DEVLOG #147.
- 2026-04-16: Added mcts_province_argmax_pct metric. See DEVLOG #146.
- 2026-04-16: Removed 50-sim fast games, reduced entropy 0.15→0.03, fixed policy_weight to 1.0. See DEVLOG #145.
- 2026-04-15: Switched to pure self-play (diversity=0.0). Opponent pool (iters 779-796) were Phase 1 models — wrong game. Metrics plateaued for 600+ iters. See DEVLOG #144.
- 2026-04-14: Phase 0 training healthy. Win rate trending up. Graduation target: waste <2 coins + MCTS province buy % >90%.

---

## Phase 0: Gold/Silver/Province (CURRENT)

**Supply:** Gold, Silver, Province (supply=3)
**Disabled:** Copper(0), Estate(3), Duchy(4), Curse(6), Gardens(16), all action cards
**Goal:** Learn that Province > Gold > Silver in buy priority when affordable. With 3 Provinces, both players can buy — winner is whoever accumulates more.

**Config:**
```yaml
province_supply: 3
disabled_basic_supply: [0, 3, 4, 6, 16]
max_action_cards: 0
draw_penalty: 0.0              # Symmetric penalty destabilizes training — removed (DEVLOG #148)
max_turns: 70                  # Safety net restored — bad policy needs cap (DEVLOG #149)
temperature_threshold: 25      # Exploration needed while policy recovers (DEVLOG #149)
opponent_diversity_ratio: 0.0  # Pure self-play — no opponent diversity in Phase 0
entropy_weight: 0.03           # Low — let policy sharpen toward Province
policy_weight: 1.0             # Fixed, no decay
# All games at full MCTS sims (800) — no fast/full split
```

**Phase 0 rule:** `opponent_diversity_ratio` must be 0.0. The opponent pool (iters 779-796) are Phase 1 models trained on a different game (Estate/Duchy enabled). Playing against them produces out-of-distribution noise, not useful signal. Pure self-play is correct for a 3-card game. Re-evaluate diversity when entering Phase 1+.

**Graduation criteria (all must hold for 20 consecutive iterations):**
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
