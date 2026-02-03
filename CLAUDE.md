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

AlphaZero-style reinforcement learning system for training a strong Mandala bot through 100% self-play. Uses MCTS + policy/value neural network, optimized for Apple Silicon (MPS backend).

**Mandala** is a 2-player card game with 6 colors (18 cards each = 108 total). Players play cards to 2 Mandalas, each with a Mountain and 2 Fields. When all 6 colors are present in a Mandala, players claim cards to their River and Cup. Scoring is based on River positions (1-6 points). First to 6 River colors or deck exhaustion triggers game end.

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Training
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
- `state.py`: Immutable game state representation with 36-card deck, hands, mountains, fields, river, cups
- `engine.py`: Rules engine handling move generation, validation, state transitions, terminal detection, scoring
- State uses canonical form (always from current player's perspective) for neural network symmetry
- `to_tensor()` converts state to neural network input (50 planes × 8×8)

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
- Default: 10 residual blocks, 128 channels, ~2M parameters
- Loss = CrossEntropy(policy, MCTS_pi) + MSE(value, outcome)

**4. Self-Play (`mandala_rl/selfplay/`)**
- `worker.py`: Generates training games by playing network against itself using MCTS
- Each position produces (state_tensor, MCTS_visit_distribution, final_outcome)
- Temperature schedule: T=1.0 for first 30 moves, then T=0
- Adds Dirichlet noise at root for exploration
- Returns `SelfPlayGame` with all states, policies, and final outcome

**5. Training (`mandala_rl/training/`)**
- `trainer.py`: Main training loop orchestrating self-play → buffer → training → checkpoint
- `replay_buffer.py`: Circular buffer storing up to 500K training examples
- Training iteration:
  1. Generate N self-play games (default: 100 games)
  2. Extract (state, policy, value) tuples and add to replay buffer
  3. Train network on random batches from buffer (default: 10 epochs)
  4. Save checkpoint
- Uses Adam optimizer with MultiStepLR scheduler
- Gradient clipping (max_norm=1.0) for stability

**6. Evaluation (`mandala_rl/evaluation/`)**
- `arena.py`: Plays matches between models with alternating colors (deterministic, T=0, no noise)
- `elo.py`: Maintains Elo rating ladder for tracking model strength over time
- Win threshold default: 55% to consider new model better
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
GameState → to_tensor() → [50×8×8] tensor
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
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
}
```

### Elo Rating System
- Models identified by checkpoint name (e.g., "model_iter_10")
- Head-to-head matches update both ratings
- Leaderboard tracks all evaluated checkpoints
- Deterministic evaluation (seed-controlled) for reproducibility

## Configuration (`configs/default.yaml`)

Key hyperparameters:
- **MCTS simulations**: 800 (balance between strength and speed)
- **c_puct**: 1.0 (exploration constant)
- **Games per iteration**: 100 (data generation rate)
- **Batch size**: 256 (training stability)
- **Learning rate**: 0.001 with MultiStepLR decay
- **Replay buffer**: 500K examples (memory vs. sample diversity)
- **ResNet blocks**: 10 (model capacity)
- **Channels**: 128 (representational power)

## Known TODOs

These areas need implementation:

1. **Game Engine (`mandala_rl/game/engine.py`)**:
   - Complete move encoding/decoding (currently placeholder)
   - Implement full move validation
   - Implement mountain completion logic
   - Implement field collection mechanics
   - Implement cup scoring

2. **State Representation (`mandala_rl/game/state.py`)**:
   - Finalize `to_tensor()` representation (currently returns zeros)
   - Optimize tensor encoding for card locations
   - Consider hand encoding strategies

3. **Symmetries (`mandala_rl/game/engine.py`)**:
   - Determine if useful symmetries exist (color permutations?)
   - Implement `get_symmetries()` if beneficial for data augmentation

## Performance Optimization Tips

### For Apple Silicon (MPS)
- Use channels=128 or 256 (MPS optimized for these sizes)
- Keep batch_size=256 (good utilization without OOM)
- Monitor unified memory usage (no separate GPU memory)
- MPS may be slower than expected for small batches; consider larger batches

### Memory Management
- Replay buffer is largest memory consumer (500K × state_size)
- Consider reducing replay buffer if OOM
- Each game stores ~30-50 positions on average

### Training Speed
- Bottleneck is typically self-play (MCTS is slow)
- Consider reducing mcts_simulations for faster iteration (e.g., 400 instead of 800)
- Parallel self-play workers could help (not currently implemented)

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
- Expected: Elo increases monotonically if training is working
- Plateau indicates need for hyperparameter tuning or architecture changes
- Win rate against baseline should exceed 55% for clear improvement
