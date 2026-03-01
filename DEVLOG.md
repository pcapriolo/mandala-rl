# Development Log

Technical changelog for the Mandala RL project. Each entry captures a significant architecture or process change: what changed, why, and key details.

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
