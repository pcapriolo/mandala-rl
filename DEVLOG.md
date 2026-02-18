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
