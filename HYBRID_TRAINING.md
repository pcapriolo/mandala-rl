# Hybrid Training Guide

Combine distributed cloud self-play with local GPU training for 10x faster iterations.

## Overview

**Problem:** Self-play is slow (3-4 hours for 100 games) and blocks your MacBook
**Solution:** Offload self-play to cloud workers, keep fast GPU training local

```
┌─────────────────────────────────────────┐
│  Cloud Workers (Distributed Self-Play)  │
│  • 10 workers in parallel               │
│  • Generate 100 games in ~30 minutes    │
│  • Return training examples             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Your MacBook (Local GPU Training)      │
│  • Receive examples via callback        │
│  • Train network on MPS/GPU (~10 min)   │
│  • Save checkpoint                      │
└─────────────────────────────────────────┘
```

## Setup

### 1. Deploy Workflow to Flyte

Your workflow is at: `workflows/distributed_selfplay_webhook.py`
Requirements: `workflows/requirements.txt`

The workflow will install from GitHub:
```
git+https://github.com/pcapriolo/mandala-rl.git
```

### 2. Start Callback Server

```bash
# Terminal 1: Start callback server
python3 workflows/callback_server.py

# Terminal 2: Expose with ngrok
ngrok http 5000
```

Copy the ngrok URL (e.g., `https://abc123.ngrok-free.dev/callback/results`)

### 3. Expose Checkpoint

```bash
# Terminal 3: Serve checkpoints
cd data/checkpoints
python3 -m http.server 8000

# Terminal 4: Expose with ngrok
ngrok http 8000
```

Copy the ngrok URL (e.g., `https://xyz789.ngrok-free.dev/model_latest_lightweight.pt`)

## Usage

### Option A: Automated Hybrid Training

Complete training loop that handles everything:

```bash
python3 scripts/train_hybrid.py \
  --config configs/default.yaml \
  --webhook-url "https://express.chapagent.com/api/webhooks/launch/YOUR_TOKEN" \
  --callback-url "https://abc123.ngrok-free.dev/callback/results" \
  --checkpoint-url "https://xyz789.ngrok-free.dev/model_latest_lightweight.pt" \
  --iterations 10 \
  --games-per-iteration 100 \
  --num-workers 10
```

**What it does:**
1. Uploads lightweight checkpoint
2. Triggers distributed workflow
3. Waits for results (callback)
4. Loads training examples
5. Trains network locally
6. Saves checkpoint
7. Repeats for N iterations

### Option B: Manual Workflow

For more control, trigger and load separately:

**Step 1: Trigger distributed self-play**
```bash
curl -X POST https://express.chapagent.com/api/webhooks/launch/YOUR_TOKEN \
  -H "Content-Type: application/json" \
  -d '{
    "callback_url": "https://abc123.ngrok-free.dev/callback/results",
    "checkpoint_url": "https://xyz789.ngrok-free.dev/model_latest_lightweight.pt",
    "total_games": 100,
    "num_workers": 10,
    "mcts_simulations": 800,
    "num_res_blocks": 8,
    "channels": 96
  }'
```

**Step 2: Wait for callback**
```bash
# Watch callback server logs
tail -f /tmp/callback_server.log

# Results saved to:
# data/distributed_results/examples_EXECUTION_ID_TIMESTAMP.pkl
```

**Step 3: Load and train**
```bash
# Load most recent results and train
python3 scripts/load_distributed_results.py --train

# Or load specific file
python3 scripts/load_distributed_results.py \
  --results-file data/distributed_results/examples_exec123_20260204.pkl \
  --train
```

## Performance

**Before (local sequential):**
- 100 games: ~5 hours
- MacBook blocked
- Single-threaded

**After (distributed hybrid):**
- 100 games: ~30 minutes
- MacBook usable
- 10x faster iteration

## Monitoring

**Callback server:**
```bash
tail -f /tmp/callback_server.log
```

**Results directory:**
```bash
ls -lh data/distributed_results/
```

**Load and inspect results:**
```python
import pickle

# Load examples
with open('data/distributed_results/examples_exec123.pkl', 'rb') as f:
    examples = pickle.load(f)

print(f"Examples: {len(examples)}")
print(f"Format: (state, policy, value)")
print(f"State shape: {len(examples[0][0])}")  # Should be list/array
```

## Troubleshooting

**Callback not receiving results:**
- Check ngrok is running: `curl http://localhost:4040/api/tunnels`
- Check callback server: `curl http://localhost:5000/health`
- Check Flyte logs for errors

**Checkpoint URL not accessible:**
- Verify: `curl -I YOUR_CHECKPOINT_URL`
- Check ngrok tunnel is active
- Make sure HTTP server is running

**Workflow fails:**
- Check error file in `data/distributed_results/error_*.json`
- Verify GitHub repo is public or accessible
- Check workflow logs in Flyte console

## Cost Estimation

Cloud costs (assuming $0.10/hour per CPU):
- 10 workers × 0.5 hours = $0.50 per iteration
- 100 iterations = $50 for full training run

Time saved: ~70 hours per 100 iterations

## Next Steps

Once comfortable with hybrid training:
1. Increase workers (10 → 20 → 50) for faster iterations
2. Automate checkpoint upload to S3 (no ngrok needed)
3. Add automatic evaluation after each iteration
4. Set up continuous training pipeline
