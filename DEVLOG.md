# Development Log

Technical changelog for the Mandala RL project. Each entry captures a significant architecture or process change: what changed, why, and key details.

---

## DEVLOG #68 — 2026-03-03: Smart ActionBigMoney seed + opponent diversity + buffer persistence

**Problem:** After rollback to iter 190, Big Money recovered to 3.7 provinces/game (iter 198) but the bot still ignores all 24 kingdom action cards (action_rate=0%). Pure self-play can't escape the Big Money local optimum — both players play identically, no action card games exist in training data, no gradient signal. Previous forcing approaches (DEVLOG #58-65) all failed by disrupting the economy.

**Root cause (re-confirmed):** The value head has zero calibration for action card positions. It's never seen a game where buying Smithy → playing Smithy → drawing 3 extra cards → more coins → more provinces led to a win. The information is in the tensor (channels 151-155 encode action card bonuses), but without training examples the network can't connect inputs to outcomes.

**Fix 1: Smart ActionBigMoney seed.** `scripts/seed_dominion_smart_abm.py` — a general heuristic that evaluates available kingdom cards by scoring function: `plus_cards*3 + plus_actions*2 + plus_coins*2 + plus_buys + special_bonus`. Buys top 1-2 action cards, plays in chain order (villages first for +actions, then draw cards), falls back to BigMoney. Handles all pending decisions (Chapel trashing, Moat reactions, Vassal/Library choices, etc.). Generated 300 games of ABM vs pure BM: 75K examples, 11.7 action plays/game, ABM wins 39%. Top cards bought: Chapel (195), Smithy (106), Village (104), Moat (54), Market (21). Seed provides diverse action card training signal across many different kingdom setups.

**Fix 2: Re-enabled opponent diversity.** `opponent_diversity_ratio: 0.2` — 20% of full-sim games play against older (weaker) checkpoints. Checkpoint pool from iter 191-206 is clean BigMoney. Against weaker opponents, marginal strategies (Smithy+BM) become visibly winning, creating gradient signal for action card learning.

**Fix 3: Buffer persistence.** Modified `trainer.py` to save `buffer_latest.pkl` alongside model checkpoints, and auto-load on `--resume`. Prevents the iter 190-210 degradation where rollback preserved weights but lost the buffer context that supported them. Buffer is saved as a separate file (not embedded in checkpoint) to avoid the OOM from 3x memory spikes (DEVLOG #35).

**Deployed:** Seed injected on top of existing 128K BigMoney buffer (no flush). Resumed from iter 206 (model_latest.pt). All forcing still disabled — seed provides calibration, diversity provides exploration, no economy disruption.

**Expected:** Value head should learn action card positions have nonzero value within 5-10 iterations. Policy should start voluntarily buying action cards (action_buys > 1.0) within 15-20 iterations. Province count should remain stable at 3.0+ (no economy disruption from seed-only approach). If action_buys don't emerge by iter 225, generate more diverse seed games with 3-4 action cards per deck.

**Files changed:** `scripts/seed_dominion_smart_abm.py` (new), `mandala_rl/training/trainer.py` (buffer save/load), `configs/dominion.yaml` (opponent_diversity_ratio: 0.2).

## DEVLOG #67 — 2026-03-03: Buffer loss caused iter 190-210 degradation (root cause + fix)

**Problem:** After DEVLOG #66 rollback to iter 190, provinces never recovered to the 3.5 baseline. Instead they declined from 2.9 → 1.4 over 20 iterations (iter 191-210), with copper buys rising and purchasing power stuck at $4.8.

**Root cause:** The rollback preserved network weights but NOT the replay buffer. The original iter 190 had 100K+ examples of strong BigMoney play accumulated over iters 77-190 (bootstrapped by the DEVLOG #57 BigMoney seed). After restart, the buffer rebuilt from scratch with only self-play from the rolled-back model. Early games were noisy (small buffer = high variance training), policy degraded slightly, degraded policy produced worse games, buffer filled with mediocre data — a vicious cycle. By iter 210 the buffer was 100K mediocre games reinforcing mediocre play.

**Fix (applied):** Re-rolled back to iter 190 with BigMoney seed re-injected (padded to 156 channels). Immediate recovery: iter 191 hit 2.9 provinces, iter 194 hit 3.7 — matching the original post-seed trajectory from DEVLOG #57.

**Permanent fix:** Added buffer persistence to `trainer.py`. Buffer saved as `buffer_latest.pkl` (separate from checkpoint to avoid OOM). Auto-loaded on `--resume`. Future restarts preserve buffer context.

---

## DEVLOG #66 — 2026-03-03: Rollback to iter_190 — DEVLOG #65 restart never applied (PID crash + ghost config)

**Problem (root cause discovery):** DEVLOG #65 (09:22) claimed to disable `action_play_force_rate: 0.15→0.0` and restart training. But between 09:22 and 10:22, the training process crashed (PID 1907007 → 1916223, unexplained restart with no logged intervention). The crash caused an auto-restart using the **original config on disk**, which still had `action_play_force_rate: 0.15`. The Monk's 10:22 check saw the new PID and accepted it as the configured restart. Iters 277-287 were trained with play-forcing still active — `action_buys` remaining stuck at 2.27-2.39 (should have dropped toward 1.5) is the telltale. The DEVLOG #65 fix never actually ran.

**Compounded failure:** With play forcing still active and the training spiral already deep (avg_provinces 0.34-0.55 vs 3.5+ baseline), continuing was pointless. All three iter-285 targets were missed: action_buys at 2.27 (target <1.5), avg_coins_at_buy at 3.59 (target >4.0), avg_provinces at 0.55 (target >1.5). No forcing levers remain.

**Action (standing CEO authority, 10:22 commitment):** Rolled back to model_iter_190.pt — pre-diversity, pre-ActionBigMoney-seed Big Money baseline (3.5 provinces/game, 15-18 treasures/game). Three changes:
1. `action_play_force_rate: 0.15 → 0.0` (actually applied this time, confirmed in log)
2. `opponent_diversity_ratio: 0.2 → 0.0` (checkpoint pool iter_222-287 contains corrupted action-buying weights; re-enable after recovery above 3.0 provinces)
3. Resume from model_iter_190_rollback.pt (151 channels → migrated to 156 automatically)

**Confirmed running:** PID 1952160, Iteration 191, "Migrating input channels: 151→156", "Resuming from iteration 190, total games: 19000."

**Expected recovery:** avg_provinces >3.0 within 10 iters (by ~iter 200), avg_coins_at_buy >5.0 within 5 iters, avg_action_buys <0.5 (no forcing, no diversity contamination). After confirmed recovery, re-enable opponent_diversity_ratio: 0.2 to allow organic action card exploration against older Big Money opponents.

## DEVLOG #65 — 2026-03-03: Disable action_play_force_rate entirely (CEO approval)

**Problem:** Provinces stuck at 0.55-0.98 for 17+ iterations (iters 258-275) despite disabling buy_force_rate at DEVLOG #64. avg_coins_at_buy remained stuck at 3.47-3.82 — baseline copper-deck level — meaning zero economic development. The bot continues buying ~2.2 action cards per game from its learned ActionBigMoney prior (DEVLOG #62 seed). These buys consume the $3 budget that should go to Silver, creating persistent Silver starvation. action_play_force_rate=0.15 was the last active forcing mechanism.

**Fix:** Disabled action_play_force_rate 0.15→0.0 per CEO approval (received in CEO inbox before this check). Killed PID 1871112, restarted from model_latest.pt (iter 275) as PID 1907007. No buffer flush — the economic problem is in the policy, not the buffer quality.

**Standing authority note:** CEO granted explicit approval and also authorized the next escalation path: if action_buys do not drop below 1.5 within 10 iters post-restart (by ~iter 285), proceed with rollback to pre-diversity checkpoint (~iter 190).

**Expected:** Without any forcing, self-play will expose that 2.2 action buys per game at avg_coins_at_buy=3.5 loses to pure Silver. The value head (currently healthy at 0.25-0.37) will backpropagate negative outcomes from action-heavy games, suppressing action_buys below 1.5 naturally. Target: action_buys < 1.5 and avg_coins_at_buy > 4.0 by iter 285.

**Files changed:** `configs/dominion.yaml` (action_play_force_rate: 0.15 → 0.0).

---

## DEVLOG #64 — 2026-03-03: Disable action_buy_force_rate entirely (standing authority)

**Problem:** Three consecutive iterations (255: 0.95, 256: 0.76, 257: 0.94) below the 1.0 province alarm threshold triggered the standing authority set in DEVLOG #63. Reducing force_rate from 0.10→0.05 (#63) bought only 5 iters of marginal recovery before the same collapse resumed. Root cause: at avg_coins_at_buy=3.5-3.9, even a 5% forced action buy consumes coin budget needed for Silver (3 coins). Each forced kingdom card buy replaces ~1 Silver purchase. Over 10 iterations this compounds: fewer Silvers → lower purchasing power → provinces stay suppressed. The action_rate signal (5-7%) confirms the network HAS internalized action card play from the seeded training — forcing is no longer needed to generate signal. Value_loss also breached 0.45 at iter 258 (0.4911), likely from the compounding economic degradation reducing game quality.

**Fix:** Disabled `action_buy_force_rate: 0.05 → 0.0` in `configs/dominion.yaml`. `action_play_force_rate: 0.15` retained — play-side forcing generates action_rate signal without hurting economy (playing an action card you already bought doesn't consume coin budget). Killed PID 1792394, restarted from model_latest.pt (iter 258), new PID 1836977. No buffer flush — existing 100K buffer retains valid game experience.

**Expected:** With no forced action buying, Silver/Gold purchases normalize. avg_coins_at_buy should recover above 4.0 within 5 iterations. Provinces should recover above 1.5 within 10 iterations (by iter ~268) and above 2.0 within 15. Action_rate should remain at 4-6% — the network voluntarily plays action cards from learned value signal, not from forcing. Value_loss should stabilize below 0.45 as game quality improves.

## DEVLOG #63 — 2026-03-03: Reduce action_buy_force_rate 0.10→0.05 to ease economic crowding

**Problem:** DEVLOG #62 seeded run (iter 221+) successfully enabled action card play (action_rate 4-6%), but provinces collapsed from 3.0+ to 0.70 over 27 iterations. Root cause: `action_buy_force_rate=0.10` forces ~1 in 10 buy-phase decisions to purchase a kingdom action card. With avg_coins_at_buy=3.5-3.9 (barely enough for Silver at 3 coins), the forced action buys consume coin budget that would otherwise go to Silver/Gold, starving the economic pipeline needed for Province purchases (8 coins). Over 27 iterations the effect compounded: fewer Silvers/Golds → lower avg_coins_at_buy → fewer Provinces → weaker score signal → Province decline from 1.91 to 0.70 at iter 248 (first breach below 1.0).

**Fix:** Reduced `action_buy_force_rate: 0.10 → 0.05` in `configs/dominion.yaml`. Restarted from iter_248 checkpoint (model_latest.pt) without buffer flush or seed re-injection — the seeded action-play behavior is already baked into the weights and buffer examples. New training: PID 1792394, iter 249 self-play active.

**Expected:** With half the forced action buying, ~0.5 fewer kingdom card buys/game frees up buy slots for Silver. Avg_coins_at_buy should stabilize above 4.0 within 5 iters. Provinces should recover above 1.5 within 10 iterations (by iter ~258) and above 2.0 within 15 iterations. Action_rate should remain at 4-6% — the seeded behavior is reinforced by network weights, not by the force mechanism alone.

**Standing authority from CEO:** If provinces don't recover above 2.0 within 10 iterations (by iter ~258), reduce force_rate further to 0.02 or disable entirely. Bot has internalized action-card play via seeds — forcing is no longer needed as the primary driver.

## DEVLOG #62 — 2026-03-02: ActionBigMoney seed to break action card deadlock (root cause fix)

**Problem:** Three successive interventions (tensor channels #58, explore boost #59, buy-forcing #60, play-forcing #61) all failed to enable action card play. Training collapsed at iter 234-235: provinces=0.13, treasures=1.22, action_rate=0.047 (finally nonzero but at the cost of all BigMoney quality). Root cause was never addressed.

**Root cause (the disease, not symptoms):** The VALUE HEAD has zero calibration for action card positions. Self-play has never produced a game where buying Smithy → playing Smithy → drawing 3 cards → having more coins → buying Province led to a win. Every MCTS simulation of "play action card" continues into territory the value head rates as bad — correctly, because its entire training history says "action plays lose." Tensor channels tell the network "Smithy draws +3 cards" but the value head can't map that to game outcome without examples. Buy/play forcing injected random action plays; the outcomes were bad (disrupted BigMoney, no learned card-chain optimization), reinforcing the value head's bias. Each intervention treated the symptom (low exploration) not the disease (no gradient signal proving actions win).

**Fix: Demonstration seeding (same approach as DEVLOG #57's BigMoney seed).** The `scripts/seed_dominion_action_bm.py` script was already written and the `action_bm_seed.pkl` (200K examples) already existed on RunPod. ActionBigMoney heuristic: buys 1-2 kingdom action cards (Smithy, Market, etc.) early, plays them in priority order each turn, then falls back to BigMoney. The 200K games include thousands of "played Smithy → drew 3 → bought Province → won" examples. This gives the value head its first gradient signal for "action cards are good when played."

**Deployed:** Killed stalled training (iter 235, PID dead). Rolled back to iter_221 checkpoint (156-channel, earliest pre-collapse weights available). Restarted with `--flush-buffer --seed-buffer /workspace/dominion_data/action_bm_seed.pkl` (PID 1674598). `action_explore_boost: 1.0` (disabled). Buy-forcing (10%) and play-forcing (15%) remain active — with the seed providing correct value calibration, forcing now reinforces the right behavior instead of causing random degradation. The seed is what was missing: forcing without value signal = noise; forcing WITH value signal = structured exploration.

**Expected:** Within 5 iterations, value_loss should reflect the seed's varied action outcomes. Within 10-15 iterations, the policy should voluntarily buy and play Smithy against itself. Province recovery to 2.5+ by iter 230, 3.0+ by iter 235.

## DEVLOG #61 — 2026-03-02: Epsilon-greedy action card playing

**Problem:** Buy forcing (DEVLOG #60) successfully increased action buys from 0.3 to 5.8/player, but action play rate remained flat 0% across 14 iterations (219-232). The forced action buys pollute the deck — action cards are dead draws that dilute treasure density, causing provinces to drop from 2.5 to 1.1 and purchasing power from $5.20 to $4.00. Buy forcing without play forcing is poison.

**Solution:** `action_play_force_rate: 0.15` — during Dominion ACTION phase, 15% of the time, override MCTS and force-play a random playable action card from hand instead of END_ACTIONS. Implementation mirrors buy forcing: check `DOM_PHASE_ACTION` + `actions_remaining > 0`, scan valid moves in `[DOM_PLAY_OFFSET, DOM_BUY_OFFSET)` for playable actions, coin flip, override `action_probs`. Placed before buy forcing in `finish_move()`.

**Why this completes the loop:** Buy forcing seeds action cards into the deck. Play forcing makes the bot actually use them. The bot now experiences the full causal chain: buy Village → draw Village → play Village → get +1 card +2 actions → chain more actions → see game outcome. The value head can finally learn whether action-heavy decks win. 85% of ACTION decisions still use full MCTS.

**Files changed:** `cpp/batched_mcts.h` (+member), `cpp/batched_mcts.cpp` (+constructor, +play forcing in finish_move), `cpp/bindings.cpp` (+param), `mandala_rl/selfplay/worker.py` (+param), `mandala_rl/training/trainer.py` (+config), `scripts/train.py` (+config extraction), `configs/dominion.yaml` (+action_play_force_rate: 0.15).

**Deployed:** Resumed from iter 233 (model_latest.pt, 156-channel, 69K buffer preserved). Used `--game dominion` (fixed prior bug where game flag was missing).

**Expected:** Within 3-5 iterations, `action_play_rate` should become nonzero (5-15%). Action cards actually get played, generating training signal. If Smithy+BigMoney is genuinely strong (it is), the network should learn to buy and play Smithy voluntarily within 20-30 iterations as the value head discovers the payoff.

## DEVLOG #60 — 2026-03-02: Epsilon-greedy action card buying

**Problem:** Action play rate remains 0.0% through 13+ iterations since DEVLOG #58's tensor channels were added. The channels tell the network "you have a Smithy that draws +3 cards" but MCTS never plays it because the policy assigns near-zero probability to PLAY actions. The root-only boost (#58) failed because deeper tree nodes still evaluate actions poorly. The channels are informational dead weight — the network can't learn from what it never tries.

**Root cause:** The bot never *buys* action cards, so it never *has* them in hand, so the new tensor channels 152-155 never activate, so there's no gradient signal. The buy phase is upstream of the play phase. Must force buying action cards to seed the causal chain.

**Solution:** `action_buy_force_rate: 0.10` — during Dominion BUY phase, 10% of the time, override MCTS and force buying a random affordable kingdom action card. Implementation in `finish_move()`: check `DOM_PHASE_BUY`, collect kingdom action cards where `supply > 0 && cost <= coins`, coin flip at `action_buy_force_rate_`, override `action_probs` to uniform over buyable actions. The forced policy is recorded as the training target.

**Why this works where boost failed:** The boost nudged MCTS exploration but MCTS correctly rejected action plays (tree continuation was bad). Forcing bypasses MCTS entirely — the card gets bought, enters the deck, and eventually appears in hand. Then: (a) tensor channels 152-155 light up, (b) MCTS might discover playing is good on its own, (c) game outcome teaches value head whether that deck composition wins. 90% of buy decisions still use full MCTS — BigMoney quality preserved.

**Files changed:** `cpp/batched_mcts.h` (+member), `cpp/batched_mcts.cpp` (+constructor, +forcing logic in finish_move), `cpp/bindings.cpp` (+param), `mandala_rl/selfplay/worker.py` (+param passthrough), `mandala_rl/training/trainer.py` (+config), `scripts/train.py` (+config extraction), `configs/dominion.yaml` (+action_buy_force_rate: 0.10).

**Deployed:** Resumed from iter 237 (model_latest.pt, 156-channel), no buffer flush. Buffer rebuilds from scratch as usual.

**Expected:** Within 5-10 iterations, `action_buys` should rise from 0.3/game to ~2-3/game. Action cards start appearing in hands. Tensor channels activate. Within 15-20 iterations, `action_play_rate` should become nonzero as the network discovers which actions are worth playing. Smithy+BigMoney is a strong strategy — the network just needs to stumble into it.

## DEVLOG #59 — 2026-03-02: Remove action_explore_boost (regression at iter 225-233)

**Problem:** 9 consecutive iterations (225-233) with avg_provinces below 3.0 alarm threshold (prior baseline 3.3-3.7). Policy loss stuck 0.085-0.144 vs prior 0.02-0.05, failing the "must return below 0.06 by iter 230" gate. action_rate=0.0 throughout — the boost wasn't enabling action plays. avg_treasures declining to 12-15 range (prior 15-18).

**Root cause:** DEVLOG #58's `action_explore_boost: 3.0` applies at root only. MCTS explores "play Smithy" at the root, but deeper in the tree the unmodified policy still assigns near-zero probability to action plays — so action card simulations degrade quickly into random/BigMoney continuation. The value estimate for "play Smithy + degraded continuation" is often WORSE than "skip action + play BigMoney." MCTS correctly rejects action plays most of the time, but the wasted simulations reduce BigMoney move quality. Result: worse BigMoney play without enabling action card play.

**Fix:** Set `action_explore_boost: 1.0` (disabled). The new tensor channels (151-156, added in DEVLOG #58) remain — they passively teach the network about action card value. Flushed the replay buffer (`--flush-buffer`) to remove degraded training examples from the boost period. Opponent diversity (DEVLOG #57) continues running at 20%.

**Deployed:** Updated `configs/dominion.yaml`, SCP'd to RunPod, restarted from `model_latest.pt` (iter 233) with `--game dominion --flush-buffer`. Single clean process (PID 1529212).

**Expected:** Policy loss should return below 0.06 within 3-5 iterations. avg_provinces should recover to 3.0+ within 5 iterations as MCTS simulation budget is no longer wasted on action card exploration.

## DEVLOG #58 — 2026-03-02: Deploy Dominion RL model to production (iter 105)

**Context:** Training hit the key milestone: avg_provinces=3.77, avg_treasures=15.4, avg_score=25.4, avg_len=131 (games ending naturally). CEO requested deploying the RL model to replace the Big Money heuristic.

**ONNX export:** Exported `model_latest.pt` (iter 105, 10,500 games) from RunPod using `scripts/export_onnx.py`. Architecture: 151 input channels, 131 actions, 10 ResNet blocks, 128 channels. Output: `data/deploy/dominion/model.onnx` (15MB).

**serve.py changes:**
- Added `DominionModelServer` class: wraps `OnnxModelServer`, implements `get_action(state)` via `get_canonical_form() → to_tensor() → ONNX predict → masked greedy argmax`.
- Updated Dominion loading: tries `data/deploy/dominion/model.onnx` first, falls back to `DominionHeuristicServer` if not present.
- Updated `loaded_games['dominion']` to expose iteration + total_games (was hardcoded `'heuristic'`).

**Landing page:** Updated Dominion card to show "Self-taught AI — trained on X self-play games" instead of checkpoint filename. Matches Mandala/LC style.

**Training state at deploy:** value_loss plateau 0.31-0.41 for 10 iters (not yet converging to ~0.10), p0_wr noisy 0.11-0.25 (high draw rate 65-74% inflates P1 apparent advantage — not a bug, both players have converged to same Big Money strategy). Bot plays recognizable Dominion: buys Silver/Gold/Province, ends games naturally.

## DEVLOG #57 — 2026-03-01: Seed replay buffer with Big Money heuristic games

**Problem:** After 77 iterations with value_loss healthy (~0.05), the bot is still stuck in a copper+estate local optimum. avg_treasures=0.04 (Silver never bought), avg_provinces=0. The value head now has gradient signal (DEVLOG #56 reward amplification worked), but the policy hasn't discovered Silver/Gold/Province buying. Self-play can't escape the equilibrium because both players play the same degenerate strategy — there's no "good example" in the replay buffer showing that Silver leads to better outcomes.

**Root cause:** Bootstrap problem. The policy needs to see Silver/Gold/Province buying to learn it's better. But self-play only generates data from the current (degenerate) policy. Without external signal, the network can't self-improve out of this hole.

**Big Money strategy:** Province($8) > Gold($6) > Silver($3), always. Play all treasures each turn, then buy the best card you can afford. This is a well-known strong baseline for Dominion that wins most games against naive opponents. Games complete naturally in ~220 moves (vs infinite with degenerate policy).

**Fix:** Wrote `scripts/seed_dominion_bigmoney.py` that generates Big Money vs Big Money games and saves (canonical_state_tensor, one_hot_policy, outcome) examples as a replay buffer pkl. Generated 300 games = 69,487 examples in 13 seconds. Restarted training from iter 77 checkpoint with `--seed-buffer /workspace/dominion_data/bm_seed.pkl`. The trainer pre-filled the replay buffer with these demonstrations and cached them for re-injection. Self-play will gradually overwrite them over ~3-4 iterations (23K new examples/iter), by which point the policy should have learned the Silver/Gold/Province buying pattern.

**Expected:** avg_treasures > 0.1 within 2 iterations. avg_provinces > 0 within 5 iterations. avg_len < 200 (games ending naturally) within 3-5 iterations.

---

## #1 — Project Bootstrap
**Feb 2, 2026**

Built the full AlphaZero pipeline from scratch: MCTS tree search, ResNet policy/value network, self-play worker, circular replay buffer, and training loop. The initial network was 10 ResNet blocks with 128 channels (~3M parameters). Everything ran on Apple Silicon via PyTorch's MPS backend.

The architecture follows DeepMind's AlphaZero paper: self-play generates (state, policy, value) training examples using MCTS + neural network guidance, then the network trains on those examples in a loop. Value head predicts game outcome (tanh, [-1, 1]), policy head predicts move probabilities (softmax over action space).

---

## #2 — Complete Mandala Rules
**Feb 2, 2026**

Implemented the full Mandala card game: 108 cards (6 colors x 18), 2 Mandalas with Mountains and Fields, River/Cup scoring. The action space is 30 moves: 12 BUILD_MOUNTAIN (6 colors x 2 mandalas) + 12 GROW_FIELD (same) + 6 DISCARD (one per color).

State representation uses 59 input channels on an 8x8 grid. All states are converted to "canonical form" (current player's perspective) before feeding to the network, so the network always learns from "my" point of view. This is a standard AlphaZero trick that halves the effective problem — the network doesn't need to learn separate strategies for player 0 vs player 1.

---

## #3 — Training Observer
**Feb 2, 2026**

Added a Flask web server for live training monitoring. Every self-play game gets serialized to JSON, and the web UI lets you replay games move-by-move. Without this, training was a black box — you'd wait hours and only see a loss curve. Now you can watch the bot's actual play and spot obvious mistakes (e.g., discarding good cards, ignoring open mandalas).

Complements TensorBoard which shows loss curves and metrics.

---

## #4 — Automatic Elo Evaluation
**Feb 2, 2026**

Integrated Elo evaluation into the training loop. Every 10 iterations, the current model plays 20 games against the previous checkpoint. Both ratings update based on results (K=32, initial 1500). Results logged to TensorBoard and saved to disk.

Eval uses reduced MCTS (400 sims vs 800 in self-play) to avoid becoming a bottleneck. This was the first version — later replaced by tournament-style evaluation after discovering the adjacent-comparison approach was too noisy to detect small improvements.

---

## #5 — Critical Bug: Canonical Form Must Copy
**Feb 5, 2026**

One-line fix with massive impact. The canonical form function returned the original object when current_player == 0, instead of a copy. This violated MCTS's immutability invariant: tree nodes shared state references, so mutating one node's state corrupted others. Manifested as "Invalid action: no card in hand" errors during self-play.

The lesson: in MCTS, every state must be an independent copy. Any shared reference is a ticking time bomb. This bug was hard to reproduce locally (seed-dependent) but crashed immediately in distributed workers with different random seeds.

---

## #6 — Lost Cities + Batched Self-Play
**Feb 12, 2026**

Two big changes in one commit:

**Lost Cities**: Added a second game to validate the framework's generality. Lost Cities is a 2-player card game with 60 cards (5 colors x 12), expeditions with ascending-value constraints, and wager multipliers. Action space: 96 compound actions (8 hand positions x 2 destinations x 6 draw sources). Input: 66 tensor channels. A clean game interface made this straightforward — implement the rules, plug in, train.

**Batched Self-Play**: Instead of playing one game at a time (network called once per MCTS simulation), play 64 games simultaneously and batch all neural network calls. One forward pass evaluates leaves from all 64 games at once. GPU utilization went from ~5% to ~80%. Self-play speedup: **3-4x**.

**Architecture**: Network shrank from 10 blocks/128ch to 8 blocks/96ch (~2M params) — the larger network was overfitting on the small replay buffer.

---

## #7 — C++ MCTS Engine
**Feb 13, 2026**

Rewrote MCTS tree traversal in C++ with pybind11 bindings. Python handled the tree search logic fine but was the bottleneck — GIL contention and interpreter overhead made each simulation slow. C++ does the same tree operations 5-10x faster.

The API uses a "split-phase" design that keeps the Python/C++ boundary clean:
1. C++ returns game states as numpy arrays
2. Python sends NN policies back to C++
3. C++ traverses trees to leaves, returns leaf states
4. Python sends NN evaluations back
5. C++ selects actions, advances games

Also added **virtual loss**: collect 4 leaves per game per simulation step (instead of 1), which means fewer Python/C++ round-trips and larger NN batches. With 64 games x 4 leaves = 256 states per batch.

**Result**: 100 games/iteration dropped from ~12 min to ~4 min.

---

## #8 — CUDA + Mixed Precision
**Feb 13, 2026**

Added GPU acceleration for NVIDIA hardware. Auto-detect: cuda > mps > cpu. Key optimizations:

- **Mixed precision (AMP fp16)**: Half-precision forward pass + gradient scaling. ~2x speedup on CUDA, free accuracy (loss scaling handles underflow).
- **torch.compile**: JIT kernel fusion for the ResNet. ~20-40% additional speedup on CUDA.
- **Deployment**: One-command GPU instance setup on RunPod.

Training moved from MacBook (MPS, ~4 min/iter) to RunPod A100 (CUDA, ~35 sec/iter). That's a **7x speedup** from hardware + software combined.

---

## #9 — Disk Management
**Feb 13, 2026**

Long training runs were filling disk. The full checkpoint included the replay buffer (~100 MB), and saving it every 10 games within an iteration generated ~1 GB/iteration. Solution:

- Mid-iteration saves are lightweight (network only, ~20 MB)
- Full save (with replay buffer) only at iteration end
- Delete mid-iteration files after iteration completes
- Retain only the last 20 iteration checkpoints

Steady-state disk usage: ~500 MB (1 full checkpoint + 20 iteration checkpoints) instead of unbounded growth.

---

## #10 — Async Elo Evaluation
**Feb 13, 2026**

Elo evaluation was blocking training. Playing 20 games with Python MCTS took ~24 min — the GPU sat idle the entire time. Fix: spawn evaluation as a background subprocess on CPU.

```
Old: [Self-Play 4min] → [Train 2min] → [Eval 24min (GPU idle)] = 30 min/iter
New: [Self-Play 4min] → [Train 2min] → [Eval spawned in background] = 6 min/iter
```

The eval worker loads two checkpoints on CPU, plays the match, and writes results to disk + TensorBoard. If the previous eval is still running when a new one is due, it's skipped (no queue buildup).

**Tradeoff**: Eval uses Python MCTS (slower but simple). Self-play uses C++ MCTS (faster). Acceptable since eval is async.

---

## #11 — Replay Saving + RunPod Sync
**Feb 13, 2026**

Training runs on a remote GPU (RunPod), but monitoring happens locally. Built a curl-based sync pipeline that polls every 30 seconds and syncs heartbeats (overwrite), losses (overwrite), Elo ratings (merge — keep whichever has more entries), TensorBoard events (incremental), replays (incremental), and checkpoints (incremental, skip the huge full checkpoint).

Also added a heartbeat system: a JSON file updates every few seconds with current iteration, phase (self-play/training), and game count. The web dashboard reads this for live status.

---

## #12 — Eval Daemon + Deployment
**Feb 13, 2026**

Extracted evaluation into a standalone daemon that runs independently of training. This decouples evaluation cadence from training speed — the daemon watches for new checkpoints and evaluates them on its own schedule.

Also added:
- **Railway deployment**: A web server hosts both Mandala and Lost Cities UIs for public human play-testing
- **Post-game feedback**: Star ratings + comments after human vs AI games
- **Lightweight deploy checkpoints**: Network weights only (~7 MB vs 100 MB full)

---

## #13 — Network Downsizing (10/128 → 8/96)
**Feb 13, 2026**

Reduced the ResNet from 10 blocks / 128 channels (~3M params) to 8 blocks / 96 channels (~2M params). The larger network was overfitting: with only 100K examples in the replay buffer and 100 games/iteration, there wasn't enough data diversity to justify 3M parameters. The smaller network trains faster, generalizes better, and reduced per-iteration training time.

This is a common AlphaZero pattern — start smaller and scale up when you have enough data throughput to support it.

---

## #14 — MCTS Simulations: 800 → 1600
**Feb 13, 2026**

Doubled MCTS simulation count from 800 to 1600. The C++ engine made this affordable — 1600 sims in C++ is faster than 800 sims in Python was. More simulations = deeper search = stronger play = better training signal.

The quality/speed tradeoff shifted when we moved to C++. At 800 sims, the network was under-searching (many leaf nodes barely explored). At 1600, the search tree is deeper and move selection is more informed.

---

## #15 — Tournament Elo Evaluation
**Feb 14, 2026**

Replaced adjacent-iteration Elo (iter N vs N-1, 20 games) with tournament-style evaluation. The problem: one training iteration improves the network by a tiny amount. Playing 20 games between nearly-identical models is like measuring a hair's width with a ruler — the noise (±3.2 Elo) drowns the signal (+0.64 Elo for a 52% win rate).

Tournament approach: every 5 iterations, the new checkpoint plays 5 games each against ~20 opponents spread evenly across training history (100 games total). All participants' Elo ratings update from results. This gives stable, cumulative measurements. If iter 300 consistently beats early iters but barely edges recent ones, the Elo curve shows gradual improvement.

---

## #16 — Determinized MCTS (Information Set MCTS)
**Feb 14, 2026**

Fixed the oracle cheating problem. During MCTS self-play, the search could see the opponent's actual hand — information the network would never have at play time. The network was learning from a teacher with superhuman knowledge, creating a ceiling where training signal became useless.

Solution: **determinized MCTS** (a.k.a. Perfect Information Monte Carlo, PIMC). Before each simulation, randomize the hidden state:
1. Pool all unseen cards (opponent's hand + deck + hidden zones)
2. Shuffle the pool
3. Re-deal to original containers (preserving sizes)
4. Run the simulation against this randomized world

MCTS now explores "what if the opponent has cards X?" across many possible hands, averaging statistics in a shared tree.

**Bug #1**: Tree actions can become invalid in randomized worlds (e.g., "draw from red discard pile" when red cards ended up in the deck). Fixed by checking action validity during traversal and breaking to treat as leaf.

**Bug #2**: When traversal breaks at an internal node (has children), the code tried to expand it, overwriting children via unique_ptr reassignment — use-after-free crash. Fixed by checking if the node is a leaf before expansion; internal nodes get a neutral backup instead.

Results: +40 Elo step for Mandala (1500 → 1540 average). Lost Cities couldn't adapt — the noisier training signal pushed its loss from 1.7 to 2.6 and Elo declined.

---

## #17 — Belief Channels + Behavioral Inference (Fresh Start)
**Feb 15, 2026**

After 474 Mandala iterations and 357 Lost Cities iterations, both games plateaued or regressed. Determinized MCTS gave Mandala a +40 Elo step but no further gains. Lost Cities couldn't adapt to the noisier training signal at all (regressed from 1500 to 1453 Elo). Root cause: the network had zero information about the opponent's hand composition — it knew the hand *size* but nothing about *which* cards.

Added three types of new input channels:

**Belief channels** (6 for Mandala, 5 for LC): For each color, P(opponent has ≥1 card of this color), computed via hypergeometric probability from card counting. Uses all publicly visible cards to determine what's left in the unseen pool, then calculates the exact probability.

**Behavioral accumulators**: Track per-color action frequencies for the opponent throughout the game. Mandala tracks mountain plays, field plays, and discards per color. Lost Cities tracks expedition plays, discards, and draw-from-discard-pile per color. These are cumulative counters updated after each move, normalized by total opponent moves for the tensor.

**What they enable**: The network can now reason: "72% chance opponent holds red (math) + opponent has been investing in red mountains (behavior) = very likely they're holding red." Or: "opponent drew from the green discard pile twice = they want green cards."

Channel counts: Mandala 59→83 (+24), Lost Cities 66→86 (+20). Added LR schedule (decay at iters 200, 400, 600). Archived all v1 checkpoints and started fresh training on both games.

---

## #18 — Parallel Tournament Evaluation
**Feb 15, 2026**

Elo evaluation was falling behind training. At iteration 37, only iters 5/10/15 had Elo ratings. The eval daemon ran 20 sequential 5-game matches per tournament — loading one opponent at a time, playing 5 games, then loading the next. Even with C++ MCTS, this was too slow.

Fix: run all 100 tournament games (20 opponents x 5 games) in **one** `BatchedMCTS` session. Added `play_tournament()` to `FastArena` which accepts 1 current model + N opponents, initializes all games at once, and routes each game's NN inference to the correct model via `_eval_multi_model()`. The current model gets large batches (~50 tensors, GPU-efficient), opponents get small batches (~2-3 each, unavoidable but still faster than sequential).

Refactored `play_match()` to delegate to `play_tournament()` with a single opponent, eliminating code duplication. Updated `eval_daemon.py` to load all opponent models upfront and call one `play_tournament()` instead of 20 sequential `play_match()` calls.

Result: Mandala iter 20 tournament completed within minutes of deployment. Eval is catching up to training.

---

## #19 — Fix Overtraining: Buffer 100K→500K, Epochs 3→1
**Feb 16, 2026**

Both games regressing despite decreasing loss. Mandala Elo collapsed from ~1555 (iters 90-155) to ~1450 (iters 170-195). Lost Cities peaked at 1575 (iter 40) then declined to ~1515. Root cause: catastrophic overtraining on a too-small replay buffer.

**Diagnosis**: With 100K buffer, 256 batch size, and 3 epochs/iteration, the model performed 1,170 gradient steps per iteration but only added ~3,000 new examples (3% buffer refresh). Each example was seen ~38x before replacement. The value head memorized the buffer — 95.6% sign accuracy on buffer states, saturated ±1 outputs on 83% of inputs including random noise. The policy head collapsed to 94% max probability (4.5% of maximum entropy). Self-play quality degraded from over-confident, memorized play.

**Fix**: Three config changes. (1) `replay_buffer_size`: 100K→500K — each example seen ~2.6x instead of ~38x per buffer cycle. (2) `epochs_per_iteration`: 3→1 — gradient steps from 1,170 to 390. (3) `lr_milestones`: [200,400,600]→[50,150,300] — earlier LR decay post-resume. Combined overtraining ratio improvement: ~15x. Mandala resumed from iter 209 with empty buffer (model_latest.pt corrupted during stop, recovered from iter checkpoint). Lost Cities resumed from iter 90 with existing buffer.

---

## #20 — Training KPI Watchdog + Dashboard Integration
**Feb 16, 2026**

The overtraining regression (#19) was caught by manual Elo inspection, not by any automated system. Added a KPI watchdog that computes health indicators from existing synced data and displays them on the dashboard.

**KPIs (per game):** Elo Trend (20-point linear regression, green/yellow/red), Value Loss health (green if 0.15-0.40, red if near-zero or high), Buffer Diversity ratio (buffer_size / examples_per_iter — higher = less overtraining), Eval Lag (training iter minus eval iter), Peak Elo (all-time best with iteration).

**Milestones:** Recovery (Elo >1555 for 5 consecutive evals), Plateau Alert (Elo range <40 over 50 evals), Strong Play (Elo >1700). Status icons: checkmark (achieved), circle (pending), warning (triggered).

Implementation: `/api/kpis` endpoint in `server.py` computes everything server-side from existing `elo_ratings.json` and `losses.jsonl`. Frontend renders as a 2-column grid between the Elo chart and System Health sections. All thresholds defined as module constants for easy tuning. No new files or dependencies — pure computation on existing data, refreshes with the 30s dashboard cycle.

---

## #21 — Fix Mountain Placement Rule (Rule of Color)
**Feb 16, 2026**

Critical game engine bug discovered during CEO playtest: `can_play_to_mountain()` only checked if a color existed in the Fields, NOT the Mountain itself. This allowed duplicate colors on a single Mountain (e.g., W, Y, Y, P, W, P), violating standard Mandala rules where each color can appear at most once per Mandala (Mountain + both Fields combined).

**Root cause:** Both C++ (`mandala_game.cpp:259-264`) and Python (`engine.py:122-128`) implementations of `can_play_to_mountain()` were missing the Mountain check. The Field validation (`can_play_to_field`) correctly checked Mountain + opponent's Field, but the Mountain validation only checked Fields.

**Impact:** All 254 iterations of Mandala training learned a fundamentally different game. The AI's weak play (losing 28-60 to a trivial heuristic), card hoarding behavior, and stagnant Elo are likely downstream effects. The fix adds `for (int c : s.mountains[mandala]) if (c == color) return false;` to both implementations. Mandala training must restart from scratch; Lost Cities is unaffected. Current broken-rules bot stays live on Railway for Phase 1 validation while the corrected model trains in parallel.

---

## #22 — Atomic Checkpoint Saves & Buffer Tuning
**Feb 16, 2026**

`model_latest.pt` (which includes the full replay buffer for resume) was corrupting to 0 bytes every time the RunPod pod crashed or restarted — because `torch.save()` truncates the file before writing. Three occurrences in one session, each requiring manual recovery from `model_iter_*.pt` (which don't include the buffer). Losing the buffer means ~29 iterations of degraded training while it refills.

**Fix:** Atomic write — save to `model_latest.pt.tmp`, then `rename()` to `model_latest.pt`. On Linux, `rename()` is atomic within the same filesystem: if the pod dies mid-write, the old file survives intact. Two-line change in `trainer.py:358-360`.

**Buffer tuning:** Reduced `replay_buffer_size` from 500K to 100K in both configs. At 100 games/iter × ~35 positions, a 500K buffer takes 143 iterations to fill but allocates ~10 GB RAM immediately (the deque pre-allocates). Two training processes + eval daemon were consuming ~22 GB. 100K is sufficient for current training maturity and saves ~16 GB. Also reduced `parallel_games` from 64 to 32 to lower peak memory during MCTS.

---

## #23 — Per-Game Elo Scoring + Stratified Random Opponents
**Feb 16, 2026**

Two changes to the tournament Elo evaluation system.

**Per-game Elo updates:** Previously, a 3-2 tournament result was reduced to aggregate scores (0.6/0.4) and fed to a single `update_ratings()` call — meaning 5-0 and 3-2 results produced the same Elo delta direction but with incorrect magnitude. Now `record_match()` is called once per game, so Elo ratings update between each game (proper Elo dynamics). A 5-0 sweep produces a much larger rating change than a 3-2 edge, as expected. Also added per-iteration W/L/D stats to the Elo JSON and dashboard hover tooltips.

**Stratified random opponent selection:** The old `select_opponents()` used deterministic even spacing — always picking the exact same opponents for any given iteration count. Now uses stratified random sampling: divides prior checkpoints into N equal buckets and randomly picks one from each bucket. Earliest and most recent checkpoints are always anchored. This gives better coverage over time — if an iteration is re-evaluated, it faces a different mix of opponents, reducing systematic bias from always testing against the same subset.

---

## #24 — Benchmark Bots + Randomized Elo Order
**Feb 16, 2026**

Three changes to the evaluation system to improve measurement quality and add baseline benchmarks.

**Randomized Elo calculation order:** Since Elo updates are sequential (each game changes ratings before the next is processed), the order in which games are processed affects final ratings. Previously, games were processed opponent-by-opponent in iteration order — introducing systematic bias. Now all game results are collected, shuffled randomly, then Elo is applied in the shuffled order. This removes ordering bias.

**Benchmark bots:** Each evaluated iteration now plays 10 games against a RandomBot (uniform policy, zero values — effectively random valid moves) and 10 games against a StrategyBot (heuristic policy that decodes the state tensor and weights actions by basic strategy rules). Results are stored in `benchmark_stats` in the Elo JSON but do NOT affect Elo ratings. The bots implement the same `__call__(batch) -> (logits, values)` interface as neural network models, so they plug directly into the existing FastArena tournament infrastructure with zero C++ changes.

**Dashboard integration:** Elo chart tooltips now show "vs Random: 9W/1L/0D" and "vs Strategy: 7W/3L/0D" when hovering. A new "Benchmark Win Rate" chart sits below the Elo chart showing win rate % over iterations — Random line (solid) should climb toward 100%, Strategy line (dashed) provides a harder baseline. New files: `mandala_rl/evaluation/benchmark_bots.py`, `docs/basic_strategy.md`.

---

## #25 — SO-ISMCTS: Proper Imperfect Information Search
**Feb 17, 2026**

Lost Cities training failed catastrophically — policy loss regressed from 1.58 (iter 75) to 2.95 (iter 210), winning only 29% vs a basic strategy bot. Root cause: the determinized MCTS from entry #16 has a critical flaw for imperfect information games. When the selected tree action is invalid in a given determinization (e.g., playing a card the opponent doesn't hold in this random world), the simulation backs up 0.0 and wastes the entire traversal. This wastes simulations, biases visit counts toward always-available actions, and produces inconsistent training targets that destabilize learning.

**Fix: Single-Observer Information Set MCTS (SO-ISMCTS).** Three changes to the C++ MCTS engine:

1. **Availability tracking** (`mcts_node.h/cpp`): Each node tracks `availability_count` — how many determinizations found this action legal when the parent was visited. The PUCT exploration term uses `sqrt(availability_count)` instead of `sqrt(N_parent)`, properly boosting exploration for rarely-available actions.

2. **ISMCTS traversal** (`batched_mcts.cpp`): At each tree node during simulation, get the valid moves for the determinized state. Update availability counts for valid children. If any valid action has no child yet, create a placeholder with uniform prior (incremental expansion). Otherwise, select among valid children only — never waste a simulation on an invalid action.

3. **Additive expand** (`mcts_node.cpp`): `expand()` now only creates children for actions that don't already exist, so children from different determinizations coexist in the same tree. Previously it replaced all children on each call.

**Complementary config changes** for Lost Cities: `dirichlet_alpha` 0.3→0.15 (determinization provides exploration), `temperature_threshold` 30→15 (less early-game noise), `lr_milestones` [50,150,300]→[150,400,800] (more budget at high LR), `replay_buffer_size` 100K→50K (fresher data). Also wired `dirichlet_alpha`/`epsilon` from YAML to the worker — previously dead config that was ignored.

**Backward compatible**: For Mandala (where most tree actions are valid in all determinizations), `availability_count ≈ N_parent`, so behavior is nearly identical. Both games benefit at opponent-move nodes deeper in the tree where determinization shuffles hands.

## #26 — Score-Based Reward + Entropy Logging + Auto-Deploy
**Feb 17, 2026**

CEO playtesting revealed LC bot opens 3-5 thin expeditions and gets crushed by -20 base penalties. Root cause: `get_reward()` returned binary +1/-1 (win/loss). The value head couldn't distinguish "risky thin-expedition win" from "safe thick-expedition win" — both gave identical +1.0 training targets. Fix: score-margin reward normalized to [-1, 1] via `margin / 100.0f`. The value head now receives richer signal about position quality. Training resumed from existing checkpoint (iter 257) — value head recalibrates, policy head unaffected.

Also: production MCTS sims reduced 50→30 to cap LC response time under 5s. Policy entropy and max action probability now logged to tensorboard for monitoring network health. Raw network policy exposed in web UI alongside MCTS visit distribution (fixes misleading "100% confidence" — that was just temp=0 one-hot from MCTS, not collapsed entropy). Auto-export deploy checkpoints every 25 iterations via `deploy_frequency` config. Fixed stale test expecting 66 channels (should be 86).

---

## #27 — Fix Production Timeout: Auto-Reload Checkpoint Stripping
**Feb 18, 2026**

Both games broken on production — AI timing out after 60-80s. Root cause: `_auto_reload_worker()` in `serve.py` was loading raw training checkpoints (~20MB with optimizer state, scheduler, replay buffer metadata) directly into the model server. On Railway's limited CPU instance, `torch.load()` on these bloated files caused massive memory pressure and slow inference.

Fix: added `_strip_checkpoint()` function that auto-reload calls before hot-swapping. When a newer training iteration is found, it loads the full checkpoint, extracts only `model_state_dict` + iteration metadata, saves the stripped version (~7MB) to `data/deploy/`, then passes that lightweight file to the model server. This ensures the deploy directory always stays current and the server never touches the full training checkpoint. Fresh deploy checkpoints pushed: Mandala iter 491, LC iter 285.

---

## #28 — Fix LC Degenerate Self-Play Equilibrium
**Feb 18, 2026**

Score-margin reward (`margin/100`) caused a degenerate equilibrium: both players converged on "never start expeditions." Evidence from replay buffer analysis: 99.3% of actions were discards, 96.3% draws from discard piles, every game hit 150-turn cap, value targets mean=0.000. The bot learned that not playing guarantees score=0, which under pure margin reward is a "safe draw" (reward=0.0) — better than risking a -0.2 from a bad expedition.

Fix: hybrid reward — binary win/loss (±0.8-1.0) + small margin tiebreaker (±0.2). Drawing is now 0.0, clearly worse than winning (~0.9). `LC_MAX_TURNS` reduced 150→60 to kill infinite discard loops. Timeout penalty makes stalling suboptimal. Replay buffer flushed (50K poisoned examples). Added `--flush-buffer` flag to train.py for future reward changes. Also synced Python engine (production) to match C++ changes: MAX_TURNS=60, reward function, and added hard 5s time limit to LC web MCTS.

---

## #29 — Switch Production to ONNX Runtime (PyTorch Can't Run on Railway)
**Feb 18, 2026**

PyTorch forward passes hang indefinitely on Railway's resource-constrained containers. A `/debug/bench` endpoint confirmed: 121s timeout on a single forward pass (gunicorn 120s limit). Root cause: PyTorch's CPU runtime has ~400MB memory overhead that causes severe thrashing on Railway's limited containers (likely 512MB-1GB RAM shared with OS + other processes).

Fix: complete rewrite of `serve.py` to use **ONNX Runtime** instead of PyTorch. Exported both models to ONNX format via `scripts/export_onnx.py` (Mandala: 6.6MB, LC: 7.1MB). ONNX Runtime has ~50MB memory overhead vs PyTorch's ~400MB, and inference is 3.5ms per forward pass locally. No more auto-reload thread, quantization hacks, or MCTS — raw network policy only (0 MCTS sims). Requirements changed from `torch>=2.0.0+cpu` to `onnxruntime>=1.16.0` + `scipy>=1.10.0` (for softmax). Gunicorn timeout reduced from 120s to 30s. Session management classes inlined in serve.py to avoid importing PyTorch-dependent modules.

---

## #30 — Entropy Regularization to Break LC Degenerate Equilibrium
**Feb 18, 2026**

The hybrid reward fix (#28) failed to break the degenerate equilibrium after 137 iterations. Replay buffer analysis at iter 442 showed 98.2% discards, 1.8% expedition plays, entropy 0.452 — identical to pre-fix behavior. Root cause: the network's policy had collapsed to near-deterministic "always discard" and there was **no loss term to penalize this**. Even with correct reward signals, the policy gradient always reinforced the same 98% confident action.

Fix: added entropy regularization to `get_loss()` in model.py. Loss becomes `policy_loss + value_loss - entropy_weight * H(pi)`, where H(pi) is the policy entropy. This creates gradient pressure against over-confident policies. Set `entropy_weight: 0.02` for LC. Also increased exploration: Dirichlet alpha 0.15→0.3, epsilon 0.25→0.35, c_puct 1.0→1.5, temperature_threshold 15→30. Flushed replay buffer again. After 3 iterations, policy entropy nearly doubled from 0.452 to 0.870 — the regularization is working. Expedition plays should rise as the buffer fills with higher-entropy exploratory data.

---

## #32 — Fix Mandala Rules: Interactive Claiming, Partial Field Plays, Cup Visibility
**Feb 18, 2026**

A BGG playtester (tphilly5) immediately identified three rule violations in our Mandala engine. All ~700 training iterations were on wrong rules. This is a complete action space and game phase overhaul — training restarts from scratch.

**1. GROW_FIELD partial plays (was: auto-play ALL).** Players can now choose 1-N cards of a color to play to their field, instead of always playing all. This is a core strategic decision (commit everything vs. hold reserves). Action space expanded from 12 field actions to 84: `12 + mandala*42 + color*7 + (count-1)`.

**2. Interactive mountain claiming (was: auto-sorted by color).** When a mandala completes (all 6 colors), it enters a new CLAIM phase. The player with more field cards picks first, choosing one mountain color at a time. Players alternate until all colors are claimed. First card of each color → River (scoring order), rest → Cup. This is arguably THE most strategic decision in Mandala — the old engine removed it entirely.

**3. Cup color visibility.** Players can now inspect their own cup card colors (6 new tensor channels, Ch 90-95). The web UI shows colored cards for human player's cup, hidden cards for opponent's cup.

**Action space:** 30 → 108 (12 BUILD_MOUNTAIN + 84 GROW_FIELD + 6 DISCARD + 6 CLAIM_COLOR). **Tensor:** 83 → 96 channels. All Mandala checkpoints incompatible. Lost Cities training unaffected.

---

## #35 — Eliminate Checkpoint OOM: Stop Saving Replay Buffer
**Feb 18, 2026**

Pod kept hitting 100% container RAM (57.74 GB) and getting killed. Root cause: `_save_checkpoint()` serialized the entire replay buffer into `model_latest.pt` every iteration. This created **3 copies in RAM simultaneously**: (1) the original deque (~5 GB), (2) `list(self.buffer)` copy from `get_all_data()` (~5 GB), (3) pickle serialization during `torch.save()` (~5 GB). With 2 trainers potentially saving near-simultaneously, that's a +30 GB memory spike on top of baseline — blowing past 57 GB.

**Fix: Stop embedding replay buffer in checkpoints.** The buffer rebuilds naturally from self-play — after restart, it refills in ~33 iterations (100K / ~3K per iter). Model weights and optimizer state are preserved, so the agent doesn't lose skill. `model_latest.pt` drops from ~5 GB to ~22 MB. Also reduced both replay buffers from 200K to 100K as additional safety margin.

**Memory profile after fix:** 2 trainers (~3 GB RSS each at steady state) + 2 eval daemons (~1 GB each) + system ≈ 10 GB total. Checkpoint saves add ~100 MB peak instead of 15 GB. Well within the 57 GB container limit.

---

## #34 — Container RAM Limit Fix + Eval Daemon Game Detection Bug
**Feb 18, 2026**

RunPod pod kept OOMing at 100% memory. Root cause: `free -h` reports host RAM (503 GB) but the container is limited to **57.74 GB** by RunPod's cgroup. Two trainers with large replay buffers + two eval daemons exceeded this.

**Fix 1: Mandala replay buffer 500K → 200K.** At 200K with 3K examples/iter and 1 epoch, each example is seen ~1x before FIFO eviction — safe overtraining ratio. Buffer cycles every ~67 iterations. Combined with LC's 200K buffer, total memory fits within the 57 GB container: ~10 GB RSS per trainer + ~1 GB per eval daemon + system overhead ≈ 22 GB.

**Fix 2: eval_daemon.py game_type detection.** The game type check was `num_actions == 30` (an old action space size). Mandala has 108 actions, LC has 96 — so it **always** resolved to "lost_cities". The Mandala eval daemon was running LC games against a Mandala model (86 vs 96 input channels), crashing immediately. This is why Mandala Elo ratings were never populated. Fixed to `num_actions == 108`.

**Fix 3: Eval daemons on GPU.** With GPU at 5% utilization and eval daemons burning 1200% CPU, moved eval back to `--device cuda`. Uses <2 GB VRAM on a 49 GB A6000. GPU utilization jumped to 84%.

---

## #33 — Fix LC Overtraining + Disk Full Crash
**Feb 18, 2026**

Both trainers crashed simultaneously — `torch.save` failed with "PytorchStreamWriter failed writing file data.pkl" because the 20GB workspace was 100% full. Root causes: `data/archive/` held 8.1GB of old v1 training data, and LC had accumulated 140 iter checkpoints (should cap at 20).

Freed 11GB by deleting archive and trimming LC checkpoints. Also diagnosed LC Elo regression: Elo peaked at iter 421 (1746) then declined to 1569 by iter 433. The culprit was the 50K replay buffer — with ~3K examples/iter, the buffer turned over every ~17 iterations. Value loss was near zero (0.01), confirming memorization. Policy loss later doubled from 0.6 to 1.3 as the degenerate equilibrium destabilized. Increased `replay_buffer_size` from 50K to 200K (67-iteration turnover, matching the ratio that works for Mandala at 500K).

---

## #31 — Fix Mandala Overtraining: 500K Buffer, Absolute LR, GPU Eval
**Feb 18, 2026**

Mandala Elo was regressing — iter 500 (1373) and iter 550 (1396) scored below iter 95-107 (~1500-1550). Loss was decreasing but play quality declining: classic overtraining. Three root causes addressed:

**1. Replay buffer 100K→500K.** DEVLOG #19 identified 500K as necessary, but #22 reduced it to 100K due to memory constraints ("two trainers + eval = ~22 GB"). Those constraints are gone — current RunPod has 410 GiB free RAM. At 100K, each example was seen ~33x before FIFO eviction. At 500K, that drops to ~6.6x. Training phase time increases from ~35s to ~175s per iteration (1953 vs 391 gradient steps), but the diversity gain is worth it.

**2. Restart-resilient LR schedule.** `MultiStepLR` milestones fired relative to `scheduler.step()` call count, which reset to 0 on every pod restart (the scheduler state was deliberately not loaded from checkpoints). This meant the LR oscillated unpredictably between 0.001 and 0.000027 depending on restart history. Replaced with `_get_lr_for_iteration()` — a simple function that computes LR from the absolute iteration number, making it deterministic regardless of restarts. Updated Mandala milestones from [50, 150, 300] to [200, 500, 800] to give the model recovery headroom (LR = 9e-5 at iter 680 instead of 2.7e-5).

**3. Eval daemons CPU→GPU.** Both eval daemons were running `--device cpu --mcts-sims 100` while 47 GiB VRAM sat idle. Switched to `--device cuda --mcts-sims 200` in `start_training.sh`. Mandala had only 41/674 models evaluated; LC had 7/514. GPU eval should catch up within hours instead of days.

---

## #36 — Fix Field Cards: Discard Instead of Deck
**Feb 18, 2026**

After a mandala is completed and all mountain colors are claimed, field cards were incorrectly returned to the bottom of the draw deck. Per the official Lookout Games rulebook: "place all of the cards in both of the Mandala's Fields in the discard pile." This bug existed in both the Python engine (`engine.py:_finish_claiming()`) and the C++ engine (`mandala_game.cpp:finish_claiming()`).

**Impact:** Field cards recycling into the deck instead of the discard pile subtly distorts game dynamics — the deck stays larger than it should, games run longer, and the discard-to-deck reshuffle trigger occurs less frequently. The AI was trained on slightly wrong card flow. Fix is a one-line change in each engine: `state.discard.extend(field_cards)` instead of `state.deck = field_cards + state.deck`.

---

## #37 — Nuclear Timeout: Break LC Degenerate Equilibrium (Third Attempt)
**Feb 18, 2026**

Lost Cities degenerate equilibrium returned for the third time at iter 644. All 200 recent games: exactly 60 moves, P1 wins 75%, value loss 0.04 (memorization). Both players discard and draw from discard piles, recycling cards so the deck never depletes. The timeout penalty from DEVLOG #28 (+0.3 winner / -0.5 loser / -0.2 draw) and entropy regularization from DEVLOG #30 (0.05 weight) weren't enough — winning a stall-fest still pays +0.3.

**Fix: Nuclear timeout.** Changed `get_reward()` so timeout gives **-1.0 to BOTH players** regardless of score. Mutual stalling is now the worst possible outcome. Players must draw from the deck (depleting it, ending the game normally) to access the ±0.8-1.0 reward range. This breaks the Nash equilibrium: deviating from "always recycle" is now strictly better.

**Fresh start:** Deleted all LC checkpoints, replays, and Elo ratings. 644 iterations of degenerate weights are unrecoverable. Kept entropy_weight=0.05 and cranked exploration params (Dirichlet alpha=0.3, epsilon=0.35, c_puct=1.5, temp_threshold=30).

---

## #38 — End-of-Iteration Game Quality Monitor
**Feb 18, 2026**

Added automated game quality assessment to catch degenerate equilibria early instead of discovering them 600 iterations too late. Three components:

**C++ `get_game_summary()`:** New method on `BatchedMCTS` extracts behavioral stats from finished games via `dynamic_cast`. Mandala: mountain/field/discard play counts per player per color, scores. Lost Cities: expedition plays, color discards, discard-pile draws, active expedition counts, scores.

**Python `_log_game_quality()`:** Called after each self-play batch. Computes universal metrics (avg game length, std dev, P0 win rate, draw rate, avg score) plus game-specific assessments. LC tracks expeditions/player and play rate (expedition plays / total plays). Mandala tracks mountain/field/discard action percentages. Prints qualitative warnings: "DEGENERATE" for <0.5 expeditions, "ALL GAMES ~60 MOVES" for zero length variance, play rate alerts. All metrics written to Tensorboard and `losses.jsonl`.

**Dashboard charts:** Four new Chart.js canvases on `index.html` — game length per game (with ±1 std dev band and P0 win rate overlay) and play quality per game (Mandala: mountain/field/discard breakdown; LC: play rate + avg expeditions dual axis). Auto-refreshes every 30s alongside existing charts.

---

## #39 — Fix LC Timeout: Draw Instead of Nuclear (Fourth Attempt)
**Feb 19, 2026**

The nuclear timeout from DEVLOG #37 (-1.0 for both players) was **fundamentally incompatible with zero-sum AlphaZero training**. In a zero-sum framework, `get_reward()` returns from the current player's perspective, and the opponent's value target is the negation. So -1.0 from P0's perspective becomes +1.0 for P1. The "both players lose" intent is impossible — one side always "wins."

After 200 iterations of fresh training: all 100 recent games exactly 60 moves, P1 wins 100%, value loss ≈ 0. The network learned "P1 always wins timeout" (correct, given the reward) and had zero incentive to avoid it.

**Fix: timeout returns 0.0 (draw).** Both players get neutral reward (0.0 from both perspectives). Natural game endings give ±0.8-1.0, making them strictly preferable to timing out. The value head now has signal: positions leading to natural wins are worth more than positions leading to draws/timeouts. This is the only timeout reward that works in a zero-sum framework without creating perverse incentives.

**Fresh start again.** Deleted all LC checkpoints, replays, Elo, and losses from the 200 degenerate iterations.

---

## #40 — Break Degenerate Self-Play: Bot Seeding + KataGo Improvements
**Feb 19, 2026**

After 100+ iterations both games are stuck in degenerate self-play — Mandala has 80% discard rate (humans never discard), Lost Cities has 95%+ timeout/draw rate with value loss → 0. Four reward-shaping attempts failed for LC. Root cause: random initialization leads to degenerate attractors that no reward function can escape.

**Fix: four complementary techniques implemented together.**

1. **Buffer seeding (AlphaGo approach)**: Pre-fill replay buffer with strategy bot self-play. Both bots upgraded with expert heuristics — Mandala: river timing, hate-drafting, mountain blocking, discard suppression. Lost Cities: wager awareness, expedition profitability, score gap awareness. `seed_from_bots.py` generates 1500 bot-vs-bot games with MCTS (200 sims) as seed data. Bot data cycles out of circular buffer in ~15 iterations.

2. **Auxiliary score head (KataGo)**: Third network output predicting normalized score margin (unbounded, no activation). Loss weight 0.5 alongside value loss. Gives richer signal than binary win/loss — a 1-point win now looks different from a 50-point blowout. Required C++ `get_score()` export, replay buffer 4-tuple support, trainer score loss integration.

3. **Policy target pruning (KataGo)**: Zero out MCTS visit probabilities < 2%, renormalize. Removes noise from low-visit actions in the policy target.

4. **Playout cap randomization (KataGo)**: 75% of games run at 200 sims (for diversity, data discarded), 25% at full sims (recorded for training). ~3x faster iterations with same training data quality.

**Files changed**: 16 files across C++, Python model, training pipeline, and scripts. Fresh restart with seeding for both games.

---

## #41 — Fix LC Bot Draw Logic + Deterministic Seed Generation
**Feb 19, 2026**

LC seed data was all 60-move timeouts despite strategy bot upgrades. Root cause: two compounding issues.

1. **Bot drew from discard pile too aggressively**. Old logic: deck draw +0.5, discard draw +1.5 when "worth drawing" (collecting OR have 2+ cards). In LC, the deck has 44 cards — if both players draw from deck, game ends naturally at turn 44. Each discard draw extends the game by one turn. With 12.4 discard draws/player, games always hit the 60-turn timeout. Fix: deck draw +3.0, discard draw only 0.0 for exceptional cases (actively collecting with wagers AND card value 8+), -10.0 otherwise.

2. **Seed generation used stochastic play**. `seed_from_bots.py` had `temperature=1.0` and Dirichlet noise, which randomly selected discard draws despite the bot's preference for deck draws. Fix: `temperature=0.0`, no Dirichlet noise — seeds should show ideal play, not random exploration.

Result: seed data went from 60.0 avg length (all timeouts, 0.0 reward) to 44.4 avg length (natural deck depletion with actual winners). After 3 training iterations, LC games showed 36% decisive outcomes (vs 0% before), discard draws halved to 8.4/player.

---

## #42 — Claude-Reasoned Seed Data + Anti-Degeneration Fixes
**Feb 19-20, 2026**

Mandala training collapsed to 84% discard rate — a degenerate equilibrium where both players throw away cards instead of building mountains or growing fields. The heuristic bot seed data (43% discard) was overwhelmed within ~10 iterations by degenerate self-play.

**Root causes**: (1) No entropy regularization on Mandala (LC had `entropy_weight: 0.05` and avoided collapse). (2) Discard is always a "safe" valid action — with no exploration pressure, the policy collapsed to it. (3) Self-play generates 3,000+ examples/iteration that swamp any seed data.

**Three-pronged fix:**

1. **Claude-reasoned seed data**: 10 Claude Code sub-agents played 50 Mandala games + 120 LC games with genuine strategic reasoning (not heuristic bots). Each agent received full rules + strategy guide and deliberated on every move. Results: 3,297 Mandala examples (13.8% discard) and 5,283 LC examples. Far superior to heuristic bot seeds.

2. **Degenerate game filter** (`max_discard_rate: 0.10`): Self-play games with >10% discard are rejected before entering the replay buffer. At 84% discard, ALL games get filtered — the buffer stays clean with only Claude seed data. As entropy regularization pushes the policy toward constructive play, games will gradually pass the filter.

3. **Entropy regularization** (`entropy_weight: 0.05`): Added to Mandala config (was missing, LC already had it). Prevents policy from collapsing to a single dominant action.

4. **Periodic seed re-injection** (`seed_reinject_frequency: 10`): Every 10 iterations, Claude seed data is re-added to the buffer so it never fully dilutes.

**Files changed**: `configs/default.yaml` (3 new params), `mandala_rl/training/trainer.py` (filter + re-injection methods), `scripts/train.py` (config passthrough + set_seed_buffer call).

## #43 — Fresh Start: Supervised Pre-Training on Claude Seed Data
**Feb 19, 2026**

Previous approach (seeding a running network with Claude data) failed — degenerate self-play immediately diluted the seed examples. Both networks were beyond repair (Mandala iter 107: 84% discard, LC iter 314: 19% play rate).

**New approach**: AlphaGo-style supervised learning phase. Fresh random network → pre-train for 100 epochs on ONLY Claude seed data → then start self-play. The network enters self-play already knowing constructive play patterns.

Added `--pretrain-epochs N` flag to `scripts/train.py`. When used with `--seed-buffer`, it runs N epochs of supervised training on the seed buffer before entering the self-play loop. Reuses existing `_train_network()` — no trainer changes needed.

Also added LC quality filtering (`min_play_rate: 0.15` in `configs/lost_cities.yaml`): reject LC self-play games where expedition play rate is below 15%. Extends the existing Mandala discard filter to protect the LC buffer too.

**Files changed**: `scripts/train.py` (pretrain flag + LC config passthrough), `mandala_rl/training/trainer.py` (LC filter in `_filter_degenerate_games`), `configs/lost_cities.yaml` (min_play_rate).

## #44 — Belief Head + Determinization Fixes + Anti-Strategy-Fusion Overhaul
**Feb 21, 2026**

Comprehensive overhaul to fix the discard-heavy play problem, diagnosed as **strategy fusion** — PIMC-style MCTS averaging across too many possible opponent hands makes field plays look risky while discarding appears universally safe.

**Three bug fixes** in C++ `randomize_hidden()`: (1) discard pile was being shuffled into unseen pool but it's face-up public info, (2) current player's own cups shuffled despite being known, (3) belief `known[]` didn't count discard or own cups, underestimating known cards and inflating uncertainty.

**Belief Head** — 4th network output head: Conv2d(ch,32,1) → BN → ReLU → FC(32×64, 12) → Sigmoid. Predicts 12 binary labels: P(opp has ≥1 of color c in hand) for c=0..5 and P(opp has ≥1 of color c in cup) for c=0..5. Ground-truth labels extracted from full game state before canonical conversion. Loss: BCE, weight 0.5. Forces the network to learn opponent modeling explicitly.

**25 new tensor channels** (96→121): discard pile color counts, opponent cup colors, global visible card counts, dead-color flags (all 18 accounted for), game progress scalar.

**Color permutation augmentation**: 50% of training batches get a random permutation of [0..5] applied to all 19 color-indexed channel groups + policy action indices + belief labels. Free 720x effective data multiplier.

**Policy weight schedule**: 3.0 → 1.0 linear decay over first 256 iterations. Strong initial policy imitation from seed data, decaying to balanced multi-head loss.

**Temperature threshold**: 30 → 12. Previous value meant 75% of a short Mandala game was played with T=1.0 random exploration — now only ~30%.

**Total loss**: `L = policy_weight × policy_loss + value_loss + 0.5 × score_loss + 0.5 × belief_loss - 0.05 × entropy`

**Files changed**: `cpp/mandala_game.h` (121 channels), `cpp/mandala_game.cpp` (bug fixes + new channels), `cpp/batched_mcts.h` + `cpp/batched_mcts.cpp` (belief labels), `mandala_rl/network/model.py` (belief head + loss), `mandala_rl/game/state.py` (121-ch Python tensor), `mandala_rl/selfplay/worker.py` (7-tuple from C++, 5-tuple output), `mandala_rl/training/replay_buffer.py` (5-tuples + augmentation), `mandala_rl/training/trainer.py` (policy weight schedule + belief loss + augmentation), `configs/default.yaml`, `scripts/regenerate_seeds.py` (NEW), all play/eval scripts (4-tuple unpacking).

---

## #45 — Vectorized Augmentation + Policy Weight Config Fix
**Feb 21, 2026**

Two critical fixes found during the #44 deployment:

**Vectorized color augmentation**: The per-example Python loop in `augment_color_permutation()` was the #1 training bottleneck — 100% CPU, 16% GPU utilization. Rewrote as `augment_color_permutation_batch()` using numpy fancy indexing: build channel/policy/belief index maps once, then apply to entire batch via `states[:, channel_map]`. Same permutation per batch (720 possibilities across thousands of batches = sufficient diversity). Pre-training dropped from ~75 min to ~20 min.

**Policy weight config passthrough bug**: `config['training']['policy_weight']` was never extracted into the flat `training_config` dict passed to the Trainer, so `self.config.get('policy_weight', 1.0)` always returned the default 1.0. The entire 3.0→1.0 decay schedule was silently disabled. Fixed by adding `'policy_weight': config['training'].get('policy_weight', 1.0)` to train.py's config extraction.

**Initial results**: Discard rate immediately dropped to 38-43% (from ~80% baseline). Field play rose to 34-37%. The combination of belief head, determinization fixes, expert seed pre-training, and correct policy weighting appears to have broken the discard equilibrium.

---

## #46 — Kill Voluntary Discard: Action Mask + c_puct Fix
**Feb 21, 2026**

Root cause analysis revealed MCTS was tripling the discard rate: raw network policy had 14.6% discard (matching seed data), but 1600 MCTS sims boosted it to 40-46% via strategy fusion. Two fixes:

**Remove discard from action mask** when any mountain or field play exists (`cpp/mandala_game.cpp`, `mandala_rl/game/engine.py`). In Mandala, voluntary discard is pure tempo waste — expert play always finds a constructive use for cards. Discard is now only offered as a legal action when no BUILD_MOUNTAIN or GROW_FIELD is available.

**Lower c_puct from 1.0 to 0.5** (`configs/default.yaml`). Pre-trained policy was good (14% discard) but MCTS exploration was overriding it. Lower c_puct makes search trust the prior more.

**Result**: Mountain 21→32%, field 33→48%, discard 46→21%. The remaining 21% is forced discards from game mechanics (positions where all hand colors are already in both mandalas' mountains). Voluntary discard rate is now 0%.

---

## #47 — Fix Rule of Color: Allow Mountain Stacking
**Feb 21, 2026**

`can_play_to_mountain` incorrectly blocked playing a card if that color was already present in the mountain. Per the official Mandala rules: "you may always play into a specific area additional cards of a color that is already present there." The Rule of Color only prevents a color from crossing zones — you can't play to mountain if the color is in either field, and you can't play to your field if it's in the mountain or opponent's field. But stacking more of the same color within the same zone is always legal.

Fixed in both C++ (`cpp/mandala_game.cpp`) and Python (`mandala_rl/game/engine.py`) engines. Removed the mountain self-check from `can_play_to_mountain`, keeping both field checks. This dramatically expands valid BUILD_MOUNTAIN moves — colors already in the mountain are now playable targets instead of dead cards, which should significantly reduce forced discards (previously ~15/game, ~25% of all moves).

---

## #48 — Optional Discard with Variable Count
**Feb 21, 2026**

Two rule corrections to discard mechanics, per official Mandala rules:

**1. Discard is always available**, not just when forced. Players can strategically discard to cycle bad cards. Previously discard was only legal when no constructive play existed.

**2. Variable discard count**: players choose how many cards of one color to discard (1 to N), drawing that many replacements. Previously discarded ALL copies of a color. Action space expanded from 108 → 150: DISCARD now encodes `96 + color*8 + (count-1)` (48 actions), CLAIM_COLOR shifted to `144 + color`.

Changes across the entire codebase: C++ engine (mandala_game.h/cpp), Python engine (engine.py), color augmentation (replay_buffer.py), config (default.yaml), benchmark bot, eval daemon, all scripts with action decoding. Checkpoint migration automatically expands the policy head from 108 → 150 outputs, remapping old DISCARD weights to all count variants and CLAIM weights to the new offset (144-149).

---

## #49 — Revert Optional Discard + Field-Advantage Channels + Rollback
**Feb 21, 2026**

Optional variable-count discard (#48) caused a discard spiral. The network discovered discarding is zero-cost (discard N, draw N, hand stays same size) and exploited it. From iter 210→227: discard rate 10%→43%, avg score 36.8→13.2, draw rate 0%→23%. Game replays showed indiscriminate color cycling — not strategic discard. Both players adopted the same degenerate strategy, creating a Nash equilibrium where win rate stays ~50/50 and the value head gets no gradient signal to correct the policy.

**Three fixes:**

**1. Revert discard to forced-only.** Kept the 150-action space (no checkpoint migration needed) but discard actions only offered when no BUILD_MOUNTAIN or GROW_FIELD plays exist. Voluntary discard rate → 0%.

**2. Field-advantage tensor channels** (121→123). Added Ch 121/122: `my_field_total - opp_field_total` for each mandala, raw count. The tensor had per-color field data but no explicit field advantage signal. Field majority determines first pick from mountain during CLAIM — arguably the core strategic mechanic. Input channel migration zero-pads the new conv_input weights.

**3. Rollback to iter 210.** Deleted degenerate checkpoints (iter 213-227). Iter 210 was the last healthy iteration (57% mountain, 10% discard, 36.8 avg score). The poisoned replay buffer (~50% of 200K capacity) made self-correction impossible.

**First results post-rollback:** Iter 211: 0% discard, 65% mountain, 35% field, 41.2 avg score — excellent recovery. Training continues from here with forced-only discard. Optional discard will be re-introduced later once the network has learned the core mountain→field→claim loop.

---

## #50 — Pre-computed Scoring Channels (123 → 137)
**Feb 22, 2026**

Added 14 new tensor channels encoding pre-computed scores. The model previously had cup color counts (ch 90-95, 102-107) and river position values (ch 42-47, 48-53) as separate channels. Computing score requires element-wise multiplication (`cup_count × river_value`) — conv layers do weighted sums, not products, so this was wasting network capacity on a fundamentally simple computation.

**New channels:** Ch 123-124: total score (my / known opp). Ch 125-136: per-color score contribution for each player. Opponent score uses only **claimed** cup cards (`cups[2:]`), not the 2 hidden starting cards — respects information boundaries. My score uses all cups since I see my own starting cards. Normalization: total `/100`, per-color `/18`.

**Checkpoint migration** handled automatically by existing `conv_input.weight` zero-padding logic (123→137 channels). Old weights preserved, new 14 channels start at zero (no behavior change until training adapts). Color augmentation groups updated to include new per-color channels.

## #51 — LC Reward Signal: Binary → Continuous Score-Margin
**Feb 24, 2026**

Root-caused the LC degenerate equilibrium (14% play rate, 60% draws, avg score -38). The problem was NOT missing tensor channels — it was the reward signal. `get_reward()` returned nearly binary ±0.8, with a tiny ±0.2 score tiebreaker. When both players discard and score -40, margin=0 → reward=0.0 (draw). With 50-60% draws, the value head got zero gradient signal for most examples and learned "discarding is fine."

**Fix:** Replaced binary reward with continuous `clamp(margin / 100, -1, 1)`. Now winning 80-to-(-30) gives +1.0, a close 10-point win gives +0.1. Also boosted score loss weight from 0.5→1.0 to make the score head a first-class training signal. Removed `min_play_rate` filter that was starving the replay buffer by rejecting low-play-rate games (the network needs to see bad outcomes to learn from them). Fresh training restart with 111ch architecture.

**Files changed:** `cpp/lost_cities_game.cpp` (reward function), `mandala_rl/network/model.py` (score loss weight), `configs/lost_cities.yaml` (remove min_play_rate), `mandala_rl/training/trainer.py` (remove LC play-rate filter).

---

## #52 — Dominion Training Launch + Monk Hourly System
**Feb 28, 2026**

Two milestones today: Dominion self-play training started on RunPod, and the Monk autonomous monitoring system went live.

**Dominion Training (first 6 iterations):**
Started training at ~15:40. C++ engine with 131 actions, 151 tensor channels, ~4M param network (10 res blocks, 128 channels, 800 MCTS sims). Early results after 6 iterations (575 games):
- Policy loss: 1.11 → 0.28 (healthy drop, network learning action structure)
- Value loss: 0.0000 (expected — all games hitting 500-move cap, no decisive outcomes yet)
- Avg provinces: 0.00, avg game length: 500 (games not terminating naturally yet)
- GPU: ~20-40% utilization, 906 MiB / 49 GiB VRAM, disk at 51%

The zero-province/500-move pattern is expected at iteration 0-10. The network hasn't learned to buy provinces yet — it's still learning basic action structure. Province buying should emerge around iter 20-50 as policy loss drops further and the network discovers that provinces end the game and contribute to score.

**Monk Hourly System:**
Built `scripts/monk_hourly.sh` — an autonomous hourly wake-up that:
1. Reads latest metrics from `data/dominion/monitor.jsonl` (fed by the 10-min monitor)
2. Checks `GG_Monk_Inbox.md` for [NEW] CEO messages
3. Assesses training health (OK/WARNING/CRITICAL based on stall detection, disk usage, connectivity)
4. Posts a per-game stats table to `GG_CEO_Inbox.md` as a `[NEW]` message
5. Appends machine-readable JSON to `data/monk_hourly.jsonl`

Runs via launchd (`com.gg.monk-hourly.plist`, every 3600s). CEO checks `GG_CEO_Inbox.md` for formatted hourly reports with full metric tables. Complements the existing 10-min `dominion_monitor.sh` which handles restart logic and raw metric collection.

**Files added:** `scripts/monk_hourly.sh`, `~/Library/LaunchAgents/com.gg.monk-hourly.plist`

## DEVLOG #52 — 2026-03-01: Fix Dominion move-cap reward returning 0

**Problem:** Dominion training stuck in total degenerate equilibrium since iter 1. All games hit the 500-move cap. `get_reward()` guards on `s.game_over` (returns 0.0 if not terminal) — but move-cap games set `terminal=true` in MCTS without setting `game_over` in the game state. Result: 100% draw rate, value loss = 0.0, zero gradient to the value head for 46+ iterations.

**Root cause:** `batched_mcts.cpp` line 294 called `game_->get_reward()` for both natural terminals AND move-cap-forced terminals. The former works; the latter silently returns 0.0 because `game_over` was never set.

**Fix:** Track `move_cap_hit` boolean. When move cap fires, bypass `get_reward` and compute score margin directly from `get_score(p0) - get_score(p1)` scaled by /30. This matches the Dominion reward formula exactly, but without the `game_over` guard.

**Deployed:** SCP'd fix to RunPod, rebuilt via `pip install -e .`, restarted training from iter 46 checkpoint. Training restarted (PID 638027).

**Expected outcome:** Next iterations should show nonzero value loss as curse/VP imbalances create small nonzero margins at move cap. This bootstraps value head learning, which should eventually drive province buying.

## DEVLOG #53 — 2026-03-01: Monk incorrectly blocked Curse buying (reverted)

**What happened:** The autonomous Monk agent (hourly cron) diagnosed the copper+curse degenerate loop and added `if (i == CARD_CURSE) continue;` to `get_valid_moves()`, preventing voluntary Curse purchase. It believed Curse buying was a rule violation.

**Why it was wrong:** Buying Curse IS a legal Dominion action. Curse costs 0, so it's always affordable — but it's -1 VP. Additionally, the game ends when 3 supply piles are exhausted, so buying Curses to empty the pile is a legitimate endgame strategy. Removing legal moves changes the game rules.

**Reverted:** Restored Curse as a valid buy option. The move-cap reward fix (#52) already provides the gradient signal the value head needs — buying Curse gives -1 VP, which now shows up as a worse outcome. The network should learn "don't buy curses" through self-play, not by having the option removed.

**Lesson for the Monk:** Don't change game rules to fix degenerate behavior. The fix should always be in the training signal, not in the action space. Trust the learning loop — if the value head has gradient signal, the policy will adjust.

## DEVLOG #54 — 2026-03-01: Auto-play treasures in buy phase

**Problem:** Dominion training stuck in copper+curse degenerate loop. Root cause: buy phase required manually playing each treasure card one at a time before buying. With a random policy, probability of playing all 3 Coppers before accidentally buying Copper/Curse or hitting END_BUYS is ~1.7%. The bot converges on cost-0 buys (Copper, Curse) because they're always 1 action away, while Silver/Estate/action cards require a multi-step treasure-play sequence first.

**Fix:** Auto-play all treasures when transitioning from action phase to buy phase. Two edits to `cpp/dominion_game.cpp`:
1. **`get_next_state()` — END_ACTIONS handler:** After setting phase to BUY, loop through hand backwards, move all treasures to in_play, add coin values (including Merchant Silver bonus).
2. **`get_valid_moves()` — buy phase section:** Removed the `unique_hand` / `is_treasure()` / `DOM_PLAY_OFFSET` block. Buy phase valid moves are now just END_BUYS + affordable cards in supply.

Action space stays at 131 — PLAY actions simply won't be valid during buy phase. Tensor encoding unchanged (channel 126 for coins now reflects full treasure value immediately). No changes to action cards, pending states, or scoring.

**Deployed:** SCP'd to RunPod, rebuilt, restarted training from iter 55 checkpoint.

## DEVLOG #55 — 2026-03-01: Fix self-play stall: move cap, sim depth limit, gain-pending fallbacks

**Problem:** Self-play stalled at 75% completion every iteration. First batch of 64 games completed in ~3.5 min, but the remaining 25-36 games hung indefinitely with 99% CPU, 0% GPU. Training couldn't progress past iter 55.

**Root cause (primary):** MCTS simulation tree traversal had no depth limit. For degenerate games where both players cycle without buying (game never terminates naturally), the search tree grew unboundedly deep. Each of the 800 simulations per move traversed from root to the deepest leaf — hundreds of `get_next_state()` calls per simulation, all CPU-bound with zero GPU work. With 25 remaining games in small batches, each game took hours instead of seconds.

**Root cause (secondary):** Move cap was 500, set when each treasure play was a separate action. With DEVLOG #54's auto-play treasures, a Dominion turn is ~2-4 moves instead of ~5-10. The 500 cap was 5-10x too generous, allowing degenerate games to burn through more simulated moves than necessary.

**Root cause (tertiary):** Four gain-pending states (`DOM_PEND_GAIN`, `DOM_PEND_REMODEL_GAIN`, `DOM_PEND_MINE_GAIN`, `DOM_PEND_ARTISAN_GAIN`) in `get_valid_moves()` could return zero valid moves when qualifying supply piles were empty, causing MCTS to waste simulations. Added `DOM_DONE_SELECTING` fallback to each, with corresponding handlers in `resolve_pending()`.

**Three fixes:**
1. **Simulation depth limit = 100** (`batched_mcts.cpp`): MCTS traversal loop now breaks after 100 nodes, treating the position as a leaf for NN evaluation. Prevents O(N²) tree traversal degradation.
2. **Move cap 500 → 200** (`batched_mcts.cpp`): With auto-play treasures, 200 moves covers ~50-100 turns — well above any reasonable game length.
3. **Gain-pending fallbacks** (`dominion_game.cpp`): `DOM_DONE_SELECTING` added when no qualifying cards exist in supply.
4. **Zero-visit-count fallback** (`batched_mcts.cpp`): When MCTS produces zero visit counts (no children expanded), fall back to uniform random over valid moves instead of UB from `discrete_distribution`.

**Results (first 2 iters post-fix):**
- Iter 56: draw_rate 82%→61%, avg_estates 0.03→0.31, value_loss 0.0002→0.0013
- Iter 57: draw_rate→34%, avg_estates→0.60, avg_duchies→0.01, avg_buys→2.15
- Self-play completes in ~10 min/iter (3.5 min first batch + ~6 min tail batch) vs infinite stall before

**Files changed:** `cpp/batched_mcts.cpp` (sim depth, move cap, zero-visit fallback), `cpp/dominion_game.cpp` (gain-pending fallbacks + resolve_pending handlers).

## DEVLOG #56 — 2026-03-01: Amplify reward signal (margin/30 → margin/5) to break copper+estate equilibrium

**Problem:** Dominion training stuck at iter 66 in a copper+estate local optimum. Metrics flat for 14 iterations: avg_provinces=0, avg_len=200 (100% move-cap), avg_score=3.5, avg_treasures=0.03 (no Silver/Gold buying), draw_rate=34-45%. Value head predicting ±0.000 with target std=0.034.

**Root cause:** Reward denominator `/30` was calibrated for decisive Dominion games (10-20 VP margin), but current self-play produces 0-3 VP margins. A 2-VP edge gives reward 0.067 — indistinguishable from noise. The value head receives near-zero gradient and can't differentiate positions. Without value signal, policy has no reason to prefer Silver over Copper.

**Bootstrap problem:** Need Silver/Gold → to buy Provinces → to create decisive outcomes → to train value head → to prefer Silver/Gold. The reward amplification breaks the loop by making small VP differences (from estate-buying variance) into meaningful signal.

**Fix:** Changed `margin / 30.0f` to `margin / 5.0f` in both `dominion_game.cpp:get_reward()` and `batched_mcts.cpp` move-cap reward. A 2-VP margin now gives 0.4 reward instead of 0.067 — 6x amplification. A 5-VP margin saturates at 1.0, appropriate for current training stage. Can rescale back to /30 once province buying emerges and margins widen.

**Deployed:** SCP'd to RunPod, rebuilt C++, restarted training from iter 66 (PID 912063).

**Expected:** Value head targets should immediately show larger std (0.034 → ~0.20). Value loss should rise. Over 5-10 iters, policy should differentiate: estates→more VP→higher reward→buy more estates→eventually realize Silver→Gold→Province is even better.

## DEVLOG #57 — 2026-03-02: Opponent diversity + realized purchasing power

**Problem:** Dominion bot (iter 190, 19K games) plays solid Big Money but completely ignores action cards (`action_rate=0.0`). Both players use the same network — they converge to identical strategy and neither explores alternatives.

**Root cause:** Symmetric self-play trap (bootstrap deadlock). MCTS can only explore from the current policy. Both players play Big Money → no action card games exist in training data → no gradient to learn action cards → both players keep playing Big Money. Same mechanism as the earlier copper+estate equilibrium, but at a higher skill plateau.

**Solution: Opponent diversity.** Break the symmetry by playing ~20% of full-sim games against older (weaker) checkpoints. Against a weaker opponent, marginal strategies (like Smithy+Big Money) become clearly winning, creating gradient signal that reinforces action card play.

**Technical approach:**
- Reuse the proven FastArena two-model routing pattern in self-play. `play_games_vs_opponent()` in `worker.py` routes root expansions and leaf evaluations to the correct model based on `get_active_players()` and game index parity (even=current as P0, odd=current as P1).
- Training data collected from **both** players. Current player's MCTS targets are high-quality; opponent's policy targets reflect older play but value targets (game outcome) are always correct. Diverse policy examples aid exploration.
- Phase 3 added to `_generate_selfplay_games()` in trainer: after full-sim self-play games, load a random older checkpoint (from recent 50% of history), play `n_opponent` games, append to training set.
- Config-gated: `opponent_diversity_ratio: 0.2` in `dominion.yaml` (default 0 for other games). Only activates after iteration 20 (need checkpoint history).

**Purchasing power metric:** Added `total_coins_at_buy` and `buy_phase_entries` tracking to C++ DominionState. Records cumulative coins when entering buy phase. Logged to TensorBoard as `GameQuality/DOM_AvgCoinsAtBuy`. Expected ~3.5 for starting deck (7 coppers in 10 cards, draw 5), rising to ~5-6 with Silver/Gold buying.

**Files changed:** `cpp/dominion_game.h` (2 tracking fields), `cpp/dominion_game.cpp` (copy + record at buy entry), `cpp/batched_mcts.cpp` (export in summary), `mandala_rl/selfplay/worker.py` (play_games_vs_opponent + two-model routing), `mandala_rl/training/trainer.py` (Phase 3 + helpers + purchasing power logging), `configs/dominion.yaml` (opponent_diversity_ratio).

**Expected outcome:** Over 20-30 iterations, action card exploration should emerge — `action_buys` and `action_rate` should become nonzero as the current model discovers winning strategies against weaker opponents and incorporates them into self-play.

## DEVLOG #58 — 2026-03-02: Action card tensor channels + exploration boost

**Problem:** Dominion training at iter 220 (22K games). Bot plays solid Big Money (3.5 provinces/game, 15-18 silver/gold) but completely ignores all 24 kingdom action cards: `action_play_rate=0%`, `action_buys=0.3/game` (noise only). Opponent diversity (#57) hasn't broken the equilibrium yet — both sides of self-play ignore actions, creating a stable deadlock the bot can't escape. Two root causes: (1) the network can't see action card bonuses in the tensor, and (2) MCTS never explores playing actions because the policy assigns them near-zero probability.

**Solution A: New tensor channels (151 → 156).** Added 5 channels that surface action card value during the ACTION phase:

| Ch | Feature | Norm | Purpose |
|----|---------|------|---------|
| 151 | Treasure coins in hand | /12 | "You'll have at least X coins after auto-play" |
| 152 | Best +cards from playable action | /5 | "Playing this draws N more cards" |
| 153 | Best +actions from playable action | /5 | "You can chain more actions" |
| 154 | Best +coins from playable action | /5 | "Actions generate coins directly" |
| 155 | Action cards in hand | /5 | "You have N actions to consider" |

Channels 152-154 are conditional on `actions_remaining > 0` — if you can't play actions, they read 0. This lets the network distinguish "I have a Smithy but already used my action" from "I have a Smithy and can play it for +3 cards."

**Solution B: Action-phase exploration boost in MCTS.** In `set_root_policies`, when Dominion is in ACTION phase with `actions_remaining > 0`, multiply `policy[DOM_PLAY_OFFSET..DOM_BUY_OFFSET]` by `action_explore_boost` (default 3.0) before Dirichlet noise and expansion. The valid-move mask ensures only actually-playable action cards get boosted. `END_ACTIONS` retains its base probability but PLAY moves get 3x weight. This makes MCTS actually explore playing action cards instead of always choosing END_ACTIONS.

**Checkpoint migration:** The existing `load_checkpoint` logic (added during the 121→123 channel migration) already handles conv_input channel migration — it zero-inits new channels and copies old weights. No migration script needed; `--resume` from iter 220 just works.

**Files changed:** `cpp/dominion_game.h` (DOM_TENSOR_CHANNELS 151→156), `cpp/dominion_game.cpp` (5 new channels in to_tensor), `dominion/game/state.py` (Python mirror, tensor shape 156×8×8), `cpp/batched_mcts.h` (game_type_ + action_explore_boost_ members), `cpp/batched_mcts.cpp` (boost logic in set_root_policies), `cpp/bindings.cpp` (new parameter), `mandala_rl/selfplay/worker.py` (pass action_explore_boost), `mandala_rl/training/trainer.py` (pass config), `scripts/train.py` (extract from config), `configs/dominion.yaml` (input_channels: 156, action_explore_boost: 3.0).

**Risk:** Boost factor too high → MCTS wastes simulations on bad action plays → training quality drops. Boost too low → no effect. 3.0 is moderate: if network gives END_ACTIONS 90% and PLAY_SMITHY 1%, after boost Smithy gets ~3% — enough for MCTS to explore but not dominate. The boost is only at root (not leaf expansions), so it's pure exploration guidance. Can tune down to 2.0 or up to 5.0 based on initial results.

**Expected outcome:** Within 5-10 iterations of resumed training, `action_plays` should become nonzero as MCTS actually explores action cards. The new tensor channels give the network the information it needs to learn *which* actions are good. Combined with opponent diversity, this should break the Big Money equilibrium.
