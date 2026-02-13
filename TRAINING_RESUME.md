# Training Resume & Checkpoint System

## Overview
The training system now supports **game-level checkpointing** for fine-grained pause/resume capability.

## Features

### ✅ Already Configured
1. **Elo Evaluation Every Iteration** (`eval_frequency: 1`)
2. **Save Every Game** (`save_replay_frequency: 1`)
3. **Checkpoint Every Iteration** (`checkpoint_frequency: 1`)

### 🆕 New: Game-Level Checkpointing
Training can now be paused and resumed at any game, not just at iteration boundaries.

**Example:**
- Training runs iteration 5 (100 games)
- Stops at game 47
- Resuming continues from game 48 (not game 1)

## Configuration

```yaml
# configs/default.yaml

training:
  checkpoint_every_n_games: 10  # Save checkpoint every 10 games
  games_per_iteration: 100       # 100 games per iteration
  checkpoint_frequency: 1        # Save iteration checkpoint every 1 iteration

evaluation:
  eval_frequency: 1              # Evaluate Elo every 1 iteration
  eval_num_games: 20             # Play 20 games for evaluation

selfplay:
  save_replay_frequency: 1       # Save every game as replay
```

## How It Works

### Checkpoint Structure
```python
checkpoint = {
    'iteration': 5,
    'total_games': 447,
    'games_in_current_iteration': 47,  # NEW: Progress within iteration
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'replay_buffer': ...,  # All training examples
}
```

### During Training
1. **Every 10 games**: Saves `model_latest_game{N}.pt`
2. **Every iteration**: Saves `model_iter_{N}.pt`
3. **Always**: Updates `model_latest.pt`

### When Resuming
```python
trainer.load_checkpoint('data/checkpoints/model_latest.pt')
# Output:
# Loaded checkpoint from model_latest.pt
# Resuming from iteration 5, total games: 447
# Mid-iteration: will continue from game 48
```

The trainer automatically:
- Skips already-completed games (1-47)
- Continues from game 48
- Completes the iteration (games 48-100)
- Trains the network
- Evaluates Elo

## Usage

### Start Training
```bash
python scripts/train.py --config configs/default.yaml
```

### Resume After Interruption
```bash
# Automatically resumes from model_latest.pt
python scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest.pt
```

### Resume from Specific Game Checkpoint
```bash
python scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest_game450.pt
```

## Benefits

1. **No Lost Work**: If training stops mid-iteration, progress is preserved
2. **Flexible Interruption**: Can pause training at any time
3. **Quick Resume**: Continues exactly where it left off
4. **Replay Buffer Preserved**: All training examples saved in every checkpoint
5. **Elo Tracking**: Every iteration is evaluated against previous iteration

## Files

### Checkpoints Saved
- `model_latest.pt` - Most recent checkpoint (always)
- `model_iter_{N}.pt` - Saved every iteration
- `model_latest_game{N}.pt` - Saved every 10 games

### Other Training Data
- `data/replays/` - Every game saved as replay file
- `data/elo_ratings.json` - Elo ratings for all checkpoints
- `data/logs/` - TensorBoard logs

## Important Notes

⚠️ **After Fixing Action Encoding Bug**: Old checkpoints (iter_0 through iter_9) are trained on incorrect game encoding. Archive them and start fresh:

```bash
mkdir -p data/checkpoints_old
mv data/checkpoints/*.pt data/checkpoints_old/
rm data/elo_ratings.json
```

Then start training from scratch with the corrected encoding.

## Verification

Run tests before training:
```bash
PYTHONPATH=/Users/paulcapriolo/mandala-rl python3 tests/test_action_encoding.py
```

Expected output:
```
================================================================================
✅ ALL TESTS PASSED!
================================================================================
Action encoding is now VERIFIED CORRECT.
```
