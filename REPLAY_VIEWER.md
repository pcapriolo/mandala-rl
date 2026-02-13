# 🎬 Training Replay Viewer

Web-based interface to browse and replay all training games, organized by iteration.

## Features

✅ **Browse by Iteration** - See all training iterations and game counts
✅ **Game List** - View all games for each iteration with scores
✅ **Interactive Replay** - Step through each move with full board visualization
✅ **Playback Controls** - Play/pause, step forward/back, jump to start/end
✅ **Full Game State** - See hands, rivers, cups, mountains, and fields at each move

## Quick Start

### 1. Start the Replay Viewer

```bash
python3 scripts/replay_viewer.py
```

**Default:** Opens on http://localhost:5003

**Custom port:**
```bash
python3 scripts/replay_viewer.py --port 8080
```

### 2. Open in Browser

Navigate to: **http://localhost:5003**

### 3. Browse Replays

1. **Select Iteration** - Click any iteration card to see games
2. **Select Game** - Click a game to watch the replay
3. **Control Playback**:
   - ⏮ **First** - Jump to first move
   - ◀ **Prev** - Previous move
   - ▶ **Play** - Auto-play at 1 move/second
   - ▶ **Next** - Next move
   - ⏭ **Last** - Jump to last move

## What You'll See

### Iteration Browser
- Lists all training iterations
- Shows game count and total moves per iteration
- Color-coded cards for easy navigation

### Game List
- All games for the selected iteration
- Move count for each game
- Final scores (P0 vs P1)
- Winner indication

### Replay Viewer
- **Player Areas** (left & right):
  - Hand cards (color emojis)
  - River cards (scoring position)
  - Cup count

- **Mandala Area** (center):
  - 2 Mandalas (0 and 1)
  - Mountains (shared)
  - Fields (per player)

- **Move Info**:
  - Current move number
  - Action description (what was played)

## Replay File Format

Games are saved in `data/replays/` as JSON:

```json
{
  "game_id": "20260207_143022_123456",
  "metadata": {
    "iteration": 5,
    "move_count": 45
  },
  "moves": [
    {
      "move_num": 1,
      "player": 0,
      "action_id": 14,
      "action_desc": "GROW_FIELD: Green → Field 0",
      "state": {
        "hands": [[1,1,1,0,5,0], [3,0,4,3,0,5]],
        "rivers": [[], []],
        "cups": [2, 2],
        "mountains": [[4,0], [0,5]],
        "fields": [[[],[]], [[],[]]]
      }
    },
    ...
  ],
  "result": {
    "score0": 18,
    "score1": 15,
    "winner": 0
  }
}
```

## Configuration

### Save Every Game
Already configured in `configs/default.yaml`:
```yaml
save_replay_frequency: 1  # Save every game
```

### Save Specific Games
To save every 10th game:
```yaml
save_replay_frequency: 10
```

## Integration with Training

When training runs, games are automatically saved with iteration metadata:

```python
# In trainer.py
if i % save_replay_freq == 0:
    game = self.selfplay_worker.play_game_with_replay(
        save_dir=replay_dir,
        iteration=self.iteration  # ✅ Now includes iteration!
    )
```

## Use Cases

### 1. Understand Learning Progress
- Watch early iterations (random play)
- Compare to later iterations (strategic play)
- See when model learns key strategies

### 2. Debug Training
- Find games where model makes mistakes
- Identify patterns in wins vs losses
- Spot rule violations or bugs

### 3. Analyze Strategy
- See which moves MCTS chooses
- Understand positional evaluation
- Learn Mandala strategy yourself!

### 4. Generate Training Examples
- Find particularly good/bad games
- Extract interesting positions
- Create targeted training data

## Multiple Viewers

You can run multiple viewers simultaneously:

```bash
# Replay viewer on port 5003
python3 scripts/replay_viewer.py --port 5003

# Human vs AI on port 5002
python3 scripts/play_vs_ai_web.py --port 5002
```

## Troubleshooting

### No Iterations Showing
- Check `data/replays/` has `.json` files
- Ensure replays have iteration in metadata
- Old replays may have `iteration: "training"` instead of number

### Game Won't Load
- Check replay file is valid JSON
- Ensure file follows naming convention: `game_YYYYMMDD_HHMMSS_ID.json`

### Replay Viewer Won't Start
- Check port 5003 isn't already in use
- Try a different port: `--port 8080`
- Check Flask is installed: `pip install flask`

## Tips

- **Keyboard Shortcuts** - Use arrow keys if browser allows
- **Fast Forward** - Click "Play" to watch at 1x speed
- **Slow Motion** - Step manually with "Next" button
- **Jump Around** - Use "First"/"Last" to see game start/end quickly

## Future Enhancements

Potential features to add:
- [ ] Search/filter games by outcome
- [ ] Show policy probabilities for each move
- [ ] Compare two games side-by-side
- [ ] Export interesting positions
- [ ] Download replay as video
- [ ] Show value estimates over time

## Related Files

- **Backend**: `scripts/replay_viewer.py`
- **Frontend**: `templates/replay_viewer.html`
- **Replay Saver**: `mandala_rl/selfplay/worker.py`
- **Replay Format**: `mandala_rl/viewer/replay.py`

Enjoy watching your AI learn to play Mandala! 🎮
