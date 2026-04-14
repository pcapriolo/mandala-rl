# Dominion Training Plan — 2-Phase Curriculum

**Last updated:** 2026-04-14 (iter 869, Phase 0a)

## Overview

Train a Dominion bot from scratch using curriculum learning. Start with the simplest decisive game (1 Province, Gold/Silver only) and progressively add complexity. Each phase graduates only when metrics prove mastery.

## Current Status

**Phase 0a** — Single Province, Gold/Silver only. Iteration 869. Win rate trending up (47.6% -> 50.0% -> 54.7%). Training healthy.

---

## Phase 0a: Learn to Buy Province (CURRENT)

**Supply:** Gold, Silver, Province (supply=1)
**Disabled:** Copper, Estate, Duchy, Curse, Gardens, all action cards
**Goal:** Learn that Province > Gold > Silver in buy priority when affordable.

With only 1 Province in the supply, the first player to buy it wins decisively (~9 VP vs ~3 VP). No draws possible. Clean gradient signal.

**Config:**
```yaml
province_supply: 1
disabled_basic_supply: [0, 3, 4, 6, 16]
max_action_cards: 0
opponent_diversity_ratio: 0.0
draw_penalty: 0.2
```

**Graduation criteria (all must hold for 20 consecutive iterations):**
- Win rate > 52% (non-random play)
- Province/player ~ 0.5 (one player gets it per game)
- Avg game length < 30 turns
- Draw rate < 5%
- Policy loss < 0.10

**Estimated graduation:** ~iter 1000-1200

---

## Phase 0b: Multi-Province Strategy

**Supply:** Gold, Silver, Province (supply=3)
**Goal:** Learn multi-Province accumulation and supply depletion timing.

With 3 Provinces, both players can buy Provinces. The winner is whoever accumulates more. This teaches economy pacing — build Gold engine first, then pivot to Province buying at the right moment.

**Config changes from 0a:**
```yaml
province_supply: 3
```

**Graduation criteria (all must hold for 30 consecutive iterations):**
- Province/player > 1.2
- Draw rate < 20%
- Avg game length 25-45 turns
- Win rate > 52%

**Risk: HIGH.** Draw trap (both players buy exactly 1 Province each). This exact failure mode caused the Phase 0 plateau at 82% draws (DEVLOG #141). The key difference: the bot now enters this phase already knowing Province is decisive, so it should naturally race for more. Mitigations (in order):
1. Monitor draw rate every 10 iterations. If > 40% for 20 iterations, increase `draw_penalty` from 0.2 to 0.5.
2. If still > 40% after 50 more iterations, re-enable `opponent_diversity_ratio: 0.3` against Phase 0a checkpoints (which play Province-first).
3. If still stuck after 100 iterations at supply=3: revert to supply=2 as intermediate step.

---

## Phase 1: Full VP Cards

**Supply:** Gold, Silver, Copper, Estate, Duchy, Province (supply=4)
**Goal:** Learn Duchy as secondary VP source. Learn Copper/Estate are dead cards (green/copper pollution).

Re-enable all basic treasure and VP cards. Province supply raised to 4 (closer to standard 8 but still fast games). The bot must learn:
- Province > Duchy > Estate in VP value
- Copper pollutes the deck (dilutes draw quality)
- Estate is only useful near game end

**Config changes from 0b:**
```yaml
disabled_basic_supply: []  # All basic cards enabled
province_supply: 4
```

**Graduation criteria:**
- Province/player > 1.5
- Duchy buying present (> 0.3/player)
- Copper buying < 0.1/player (learned to avoid)
- Draw rate < 15%
- Win rate > 52%

---

## Phase 2: Action Cards

**Supply:** Full basic + 2-3 simple action cards
**Goal:** Learn action card synergies (Village + Smithy, etc.)

Start with `max_action_cards: 2`, restricted to simple non-attack cards. Expand gradually.

**Config changes from Phase 1:**
```yaml
max_action_cards: 2
province_supply: 6
```

**Graduation criteria:** TBD based on Phase 1 results.

---

## Phase 3: Full Dominion

**Supply:** Standard 10-card kingdom, province_supply=8
**Goal:** Competitive play across varied kingdoms.

**Config changes:** Standard Dominion rules. TBD.

---

## Rules

1. **Never skip phases.** Each phase builds on learned representations from the previous one.
2. **Never change province_supply and disabled_basic_supply simultaneously.** One variable at a time.
3. **Checkpoint before every phase transition.** Back up model + config to `/workspace/dominion_data/backup_phase{N}/`.
4. **No weight surgery.** Bias nudges only. Seed data injection is the approved intervention for stuck priors (DEVLOG #137).
5. **Monitor overtraining ratio.** Each iteration ~3,000 examples. Buffer 100K, 1 epoch = each example seen ~1.3x. Safe.
