# Dominion Training Plan — Phased Curriculum

## Current Status

| Field | Value |
|-------|-------|
| **Phase** | 0 — Gold/Silver/Province |
| **Iteration** | ~1019 |
| **Province Supply** | 3 |
| **Key Metrics** | Plateaued — MCTS prov% ~36%, waste ~3.7, draw rate ~8% |
| **Last Updated** | 2026-04-14 |

### Notes
- 2026-04-14 (iter 1019): Training plateaued. Metrics flat for 200+ iterations. Config changes applied: entropy_weight 0.15→0.05, temperature_threshold 25→15. Goal: sharpen policy on Province buying to break plateau.
- 2026-04-14 (iter 869): Phase 0 training healthy. Win rate trending up. Graduation target: waste <2 coins + MCTS province buy % >90%.

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
draw_penalty: 0.2
max_turns: 70
```

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
