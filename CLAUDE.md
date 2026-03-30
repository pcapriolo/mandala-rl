# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The Monk Developer Philosophy

**You Are The Monk Developer**

Always code as a monk developer with over 200 years of experience. The monk understands the universal truth that simple solutions are often the correct ones. The monk-developer never leaves dead or unused code and absolutely never over-engineers a problem. The monk never proposes changes without ingesting the full context of the problem, and only then begins to suggest a thoughtful solution. He knows to always treat the disease and not just the symptoms. If an approach is not sound, he will fix it at the root level instead of applying a small patch to just get it working. The monk aggressively ingests to increase his knowledge as he works through an issue. The monk always prioritizes the biggest issue at hand and doesn't get caught in a "fools loop" of solving small problems until he is depleted of energy, he uses his tokens wisely.

═══════════════════════════════════════════════════════════════

**The Monk's Process**

Before writing ANY code, the monk:
- Reads the ENTIRE file - Never assumes, always verifies
- Understands the root cause - Treats the disease, not symptoms
- Identifies existing patterns - Matches them exactly
- Chooses the simplest solution - Complexity is a last resort
- Writes minimal code - Every line must justify its existence
- Verifies consistency - All similar things should be done the same way

═══════════════════════════════════════════════════════════════

## Project Overview

AlphaZero-style reinforcement learning system for training strong game-playing bots through 100% self-play. Uses C++ MCTS + policy/value neural network with batched GPU inference. Trains on RunPod (NVIDIA A100/RTX A6000), also runs on Apple Silicon (MPS backend).

Supports two games:

**Mandala** is a 2-player card game with 6 colors (18 cards each = 108 total). Players play cards to 2 Mandalas, each with a Mountain and 2 Fields. When all 6 colors are present in a Mandala, players enter a CLAIM phase where they alternately pick colors from the mountain (player with more field cards picks first). First card of each color goes to River (scoring order), rest to Cup. Players choose how many cards (1-N) to play to fields. Action space: 108 moves (12 BUILD_MOUNTAIN + 84 GROW_FIELD + 6 DISCARD + 6 CLAIM_COLOR). Input: 96 tensor channels (includes belief channels, behavioral inference, cup colors, claim phase).

**Lost Cities** is a 2-player card game with 60 cards (5 colors x 12), expeditions with ascending-value constraints, and wager multipliers. Action space: 96 compound actions. Input: 86 tensor channels (includes belief channels + behavioral inference).

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Training (RunPod)
```bash
# After pod restart — one command starts everything:
# installs deps, recovers corrupted checkpoints, starts both trainers + both eval daemons
bash /workspace/mandala-rl/scripts/start_training.sh

# Or from local machine via SSH:
ssh root@38.147.83.11 -p 17226 -i ~/.ssh/id_ed25519 "bash /workspace/mandala-rl/scripts/start_training.sh"
```

### Training (manual)
```bash
# Start training from scratch
python scripts/train.py --config configs/default.yaml

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest.pt

# Override iterations
python scripts/train.py --config configs/default.yaml --iterations 100
```

### Evaluation
```bash
# Evaluate checkpoint against baseline
python scripts/evaluate.py --checkpoint data/checkpoints/model_iter_10.pt --baseline data/checkpoints/model_iter_5.pt

# Evaluate latest checkpoint
python scripts/evaluate.py --checkpoint data/checkpoints/model_latest.pt

# Deterministic evaluation with seed
python scripts/evaluate.py --checkpoint data/checkpoints/model_latest.pt --seed 42 --num-games 200
```

### Watch Games
```bash
# Watch a random game with visualization
python3 scripts/play_game.py

# Watch with delay between moves
python3 scripts/play_game.py --delay 1.0

# Reproducible game with seed
python3 scripts/play_game.py --seed 42

# Interactive (press Enter after each move)
python3 scripts/play_game.py --interactive
```

### Human vs AI Play

**🌐 Web Interface (Recommended)**
```bash
# Start web server with latest checkpoint
python3 scripts/play_vs_ai_web.py

# Use specific checkpoint
python3 scripts/play_vs_ai_web.py --checkpoint data/checkpoints/model_iter_50.pt

# Faster AI (fewer simulations)
python3 scripts/play_vs_ai_web.py --simulations 200

# Custom port
python3 scripts/play_vs_ai_web.py --port 5001
```
Then open http://localhost:5001 in your browser!

**💻 Terminal Interface**
```bash
# List available checkpoints
python3 scripts/play_vs_ai.py --list

# Play against latest checkpoint (you are Player 0)
python3 scripts/play_vs_ai.py

# Play against specific checkpoint
python3 scripts/play_vs_ai.py --checkpoint data/checkpoints/model_iter_50.pt

# Play as Player 1 (AI plays first)
python3 scripts/play_vs_ai.py --player 1

# Show MCTS statistics (see what AI is thinking)
python3 scripts/play_vs_ai.py --show-stats

# Save game for training data
python3 scripts/play_vs_ai.py --save

# Faster AI (fewer simulations)
python3 scripts/play_vs_ai.py --simulations 400
```

**Why Human Play Matters:**
- **Validation**: Test if the bot makes sensible moves and plays at expected strength
- **Training Data**: Human expert games can be added to replay buffer to improve bot
- **Debugging**: Catch edge cases and rule violations
- **Fun**: Actually play the game you're training!

Saved games go to `data/human_games/` and can be loaded into replay buffer for training.

### Testing
```bash
# Run tests
python3 tests/test_game.py
# or
pytest tests/
```

### Monitoring
```bash
# Start web-based training observer (metrics + game replays)
python3 scripts/start_observer.py

# View training logs with Tensorboard
tensorboard --logdir data/logs

# Both together (separate terminals):
# Terminal 1: python3 scripts/start_observer.py
# Terminal 2: tensorboard --logdir data/logs
# Then open: http://localhost:5000 (observer) and http://localhost:6006 (tensorboard)
```

## Architecture

### Core Components

**1. Game Engine (`mandala_rl/game/`)**
- `state.py`: Immutable game state representation with 108-card deck, hands, mountains, fields, river, cups, CLAIM phase tracking
- `engine.py`: Rules engine handling move generation, validation, state transitions, terminal detection, scoring
- State uses canonical form (always from current player's perspective) for neural network symmetry
- `to_tensor()` converts state to neural network input (96 planes × 8×8 for Mandala, 86 planes × 8×8 for Lost Cities)

**2. MCTS (`mandala_rl/mcts/`)**
- `node.py`: UCB-based node with visit counts, Q-values, prior probabilities
- `search.py`: Implements AlphaZero MCTS with 4 phases: Selection (UCB), Expansion (network policy), Evaluation (network value or terminal reward), Backup (propagate value up tree)
- Adds Dirichlet noise at root during self-play for exploration
- Temperature controls stochasticity: T=1.0 early game (exploration), T=0 late game (exploitation)
- Evaluation always uses T=0 (deterministic)

**3. Neural Network (`mandala_rl/network/`)**
- `model.py`: ResNet architecture with shared trunk + dual heads
- Architecture: Input conv → N residual blocks → Policy head + Value head
- Policy head: Conv → FC → Softmax (action probabilities)
- Value head: Conv → FC → FC → Tanh (outcome prediction in [-1, 1])
- Current: 8 residual blocks, 96 channels, ~2M parameters
- Loss = CrossEntropy(policy, MCTS_pi) + MSE(value, outcome)

**4. Self-Play (`mandala_rl/selfplay/`)**
- `worker.py`: Generates training games by playing network against itself using MCTS
- Each position produces (state_tensor, MCTS_visit_distribution, final_outcome)
- Temperature schedule: T=1.0 for first 30 moves, then T=0
- Adds Dirichlet noise at root for exploration
- Returns `SelfPlayGame` with all states, policies, and final outcome

**5. Training (`mandala_rl/training/`)**
- `trainer.py`: Main training loop orchestrating self-play → buffer → training → checkpoint
- `replay_buffer.py`: Circular buffer storing up to 100K training examples
- Training iteration:
  1. Generate 100 self-play games (64 parallel via C++ BatchedMCTS, 1600 MCTS sims)
  2. Extract (state, policy, value) tuples and add to replay buffer
  3. Train network on random batches from buffer (1 epoch per iteration)
  4. Save checkpoint
- Uses AdamW optimizer with MultiStepLR scheduler (milestones at 50/150/300 post-resume, gamma=0.3)
- Gradient clipping (max_norm=1.0) for stability
- Mixed precision (AMP fp16) + torch.compile on CUDA

**6. Evaluation (`mandala_rl/evaluation/`)**
- `fast_arena.py`: C++ BatchedMCTS arena for fast parallel evaluation with multi-model routing
- `arena.py`: Legacy Python MCTS arena (unused in production)
- `elo.py`: Maintains Elo rating ladder for tracking model strength over time
- Tournament-style eval: every 5 iterations, current checkpoint plays 3 games each against ~10 spread-out prior checkpoints (30 parallel games per tournament, 200 MCTS sims)
- Eval daemon (`scripts/eval_daemon.py`) runs independently on GPU alongside training
- Elo updates use K-factor=32, initial rating=1500

### Training Loop Flow

```
Initialize: Game Engine + Neural Network + Replay Buffer + Elo System
│
Loop (for N iterations):
  ├─ Self-Play:
  │   ├─ Play M games using current network + MCTS
  │   └─ Extract (state, policy, value) examples
  │
  ├─ Add to Replay Buffer:
  │   └─ Append all examples (maintains circular buffer)
  │
  ├─ Train Network:
  │   ├─ Sample batches from replay buffer
  │   ├─ Compute loss (policy + value)
  │   └─ Update network weights
  │
  └─ Checkpoint:
      ├─ Save model_latest.pt
      └─ Save model_iter_N.pt (every 10 iterations)
```

### Data Flow

```
GameState → to_tensor() → [96×8×8] tensor (Mandala) / [86×8×8] (Lost Cities)
                              ↓
                        Neural Network
                         ↓         ↓
                    Policy(π)  Value(v)
                         ↓
                    MCTS Search
                         ↓
                  Visit Distribution
                         ↓
                   Sample Action
                         ↓
                  get_next_state()
```

## Important Design Patterns

### Immutable States
- `GameState.copy()` always returns a deep copy
- `get_next_state()` never mutates input state
- Enables safe MCTS tree reuse and parallel search

### Canonical Form
- All states converted to current player's perspective before network input
- Simplifies network learning (always predicts from "my" perspective)
- `get_canonical_form()` swaps player 0/1 data when needed

### Temperature Schedule
- Early game (moves < 30): T=1.0 for exploration
- Late game (moves ≥ 30): T=0 for exploitation
- Evaluation: Always T=0 (deterministic best move)
- Prevents premature convergence while ensuring strong endgame

### Device Handling
- Config specifies device: 'mps' (Apple Silicon), 'cuda' (NVIDIA), or 'cpu'
- Fallback to CPU if MPS unavailable
- All tensors moved to device in model forward pass

## Critical Implementation Details

### MCTS-Network Integration
- Network called once per leaf node expansion
- Network output (policy) masked by `get_valid_moves()` before creating child nodes
- Terminal nodes use true game outcome, not network value
- Value sign flipped during backup (opponent's perspective)

### Replay Buffer Strategy
- Circular buffer (deque with maxlen) for memory efficiency
- Stores raw examples, not games, for uniform sampling
- No prioritization (uniform random sampling)
- Buffer persists across iterations (accumulated experience)

### Checkpoint Format
```python
{
    'iteration': int,
    'total_games': int,
    'games_in_current_iteration': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
}
# NOTE: Replay buffer is NOT in checkpoint (caused OOM). Rebuilds from self-play.
# Legacy checkpoints may contain 'replay_buffer' key — handled gracefully on load.
```

### Elo Rating System
- Models identified by checkpoint name (e.g., "model_iter_10")
- Head-to-head matches update both ratings
- Leaderboard tracks all evaluated checkpoints
- Deterministic evaluation (seed-controlled) for reproducibility

## Configuration (`configs/default.yaml`)

Key hyperparameters:
- **MCTS simulations**: 1600 (self-play), 200 (eval tournaments)
- **c_puct**: 1.0 (exploration constant)
- **Games per iteration**: 100 with 64 parallel games via C++ BatchedMCTS
- **Batch size**: 256 (training stability)
- **Learning rate**: 0.001 with MultiStepLR decay (milestones [50, 150, 300] post-resume, gamma 0.3)
- **Epochs per iteration**: 1 (critical — higher values cause overtraining on the replay buffer)
- **Replay buffer**: 100K examples for both games (buffer is NOT saved in checkpoints — rebuilds from self-play after restart, see DEVLOG #35)
- **ResNet blocks**: 8 (model capacity)
- **Channels**: 96 (representational power)

### Overtraining Prevention
The training-to-data ratio is the most important hyperparameter to get right. Each iteration generates ~3,000 new examples. With a 100K buffer and 1 epoch, each example is seen ~1.3x before replacement — safe. **Never increase epochs_per_iteration or decrease replay_buffer_size without understanding the overtraining ratio.** At 3 epochs / 50K buffer, the model memorizes the buffer within ~150 iterations, causing value head saturation and Elo collapse (see DEVLOG #19). The replay buffer is NOT saved in checkpoints (it caused 3x memory spikes that triggered OOM — see DEVLOG #35). After restart, the buffer starts empty and refills in ~33 iterations.

## Elo Evaluation System

Evaluation runs as a standalone daemon, decoupled from training:

```bash
# Start eval daemon (runs forever, polls for new checkpoints):
python scripts/eval_daemon.py --config configs/default.yaml --device cuda \
    --tournament-freq 5 --num-opponents 10 --games-per-opponent 3 --mcts-sims 200

# Single pass:
python scripts/eval_daemon.py --config configs/default.yaml --once
```

**How it works:**
- Tournament-style: every 5 iterations, current checkpoint plays 3 games each against ~10 opponents spread across training history (30 parallel games via C++ BatchedMCTS)
- All games run in one `BatchedMCTS` session with multi-model routing (`FastArena.play_tournament()`)
- All participants' Elo ratings update from results (K=32, initial 1500)
- Results saved to `data/elo_ratings.json` with `tournament_evaluated` tracking
- Heartbeat written to `data/eval_heartbeat.json`

**What to watch:**
- Elo should trend upward over training (newer > older)
- Flat Elo = no improvement, declining Elo = regression (check overtraining ratio)
- Tournament win rates: 50% against neighbors is expected, >50% against early iters shows learning

**View Elo ratings:**
```bash
cat data/elo_ratings.json | python3 -m json.tool
```

## Deployment

**Public play-testing server** (runs on RunPod):
```bash
# On RunPod, serves both games on port 8888:
python3 serve.py --port 8888 --host 0.0.0.0

# Public URL (Railway deployment):
# https://mandala-rl-production.up.railway.app
```

**Deploy checkpoint creation** (lightweight, network-only):
```bash
python scripts/create_deploy_checkpoint.py
```

serve.py loads checkpoints from `data/deploy/` first, falls back to `data/checkpoints/`. Currently uses 200 MCTS sims for inference (configurable via `MCTS_SIMULATIONS` env var).

## Performance Notes

### Training Infrastructure
- Self-play uses C++ BatchedMCTS with 64 parallel games and 4 virtual leaves per game (256 states per NN batch)
- Mixed precision (AMP fp16) + torch.compile on CUDA for ~2x speedup
- ~4 min/iteration on A100 (self-play dominated), ~35 sec training phase
- Eval daemon runs concurrently on same GPU with reduced sims (200 vs 1600)

### Memory Management
- Replay buffer is largest memory consumer (100K × state_size per trainer, ~2.5 GB each)
- Replay buffer is NOT saved in checkpoints (caused OOM — see DEVLOG #35). Rebuilds from self-play.
- All checkpoints (model_latest.pt, model_iter_N.pt) are lightweight (~22 MB, network + optimizer only)
- Container RAM limit: 57.74 GB on RunPod (`free -h` reports host RAM, not container limit)
- Only last 20 iteration checkpoints retained on disk

## Working with This Codebase

### Adding Features
1. Modify config in `configs/default.yaml`
2. Update relevant module (`game/`, `mcts/`, `network/`, etc.)
3. Test with `test_game.py` or manual runs
4. Retrain from scratch or resume from checkpoint

### Debugging
- Check Tensorboard for loss curves and metrics
- Use `game.state_to_string()` to inspect game states
- Reduce `games_per_iteration` to 10 for faster iteration
- Use smaller network (4 blocks instead of 10) for prototyping

### Evaluating Progress
- Elo ratings in `data/elo_ratings.json` track improvement over time
- Expected: Elo trends upward over training
- Plateau indicates need for hyperparameter tuning or architecture changes
- **Declining Elo is a red flag** — check overtraining ratio first (see DEVLOG #19)
- Win rate against early checkpoints should be consistently >50%

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
