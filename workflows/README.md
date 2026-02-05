# Distributed Self-Play Workflows

This directory contains Flyte workflows for distributed self-play game generation, dramatically speeding up the training bottleneck.

## Architecture

```
MacBook (GPU)           Cloud Workers (CPU)
-------------           -------------------
Train network    --->   Generate games in parallel
     ^                          |
     |                          |
     +--------------------------|
         Download examples
```

## Setup

### 1. Install Flyte CLI

```bash
pip install flytekit
```

### 2. Configure Flyte Connection

Create `~/.flyte/config.yaml`:

```yaml
admin:
  endpoint: your-flyte-server.com
  insecure: false  # true for local testing
  authType: Pkce   # or ClientSecret, ExternalCommand, etc.

storage:
  connection:
    access-key: YOUR_ACCESS_KEY
    secret-key: YOUR_SECRET_KEY
  container: s3://your-bucket  # or gs://, abs://, etc.
```

### 3. Make Package Installable

Add to your `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="mandala-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
    ]
)
```

Then install in editable mode:

```bash
pip install -e .
```

### 4. Build Docker Image (for Flyte)

Create `workflows/Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install your package
COPY . /app
RUN pip install -e .

# Install workflow requirements
COPY workflows/requirements.txt /app/workflows/
RUN pip install -r workflows/requirements.txt
```

Build and push:

```bash
docker build -t your-registry/mandala-rl:latest .
docker push your-registry/mandala-rl:latest
```

### 5. Register Workflow

```bash
pyflyte register workflows/distributed_selfplay.py \
  --project mandala-rl \
  --domain production \
  --image your-registry/mandala-rl:latest
```

## Usage

### Option 1: Direct Python API

```python
from pathlib import Path
from workflows.client import DistributedSelfPlayClient
from flytekit.configuration import Config, PlatformConfig

# Configure
config = Config(
    platform=PlatformConfig(endpoint="your-flyte-server.com")
)

# Generate games
client = DistributedSelfPlayClient(config)
examples = client.generate_games_distributed(
    checkpoint_path=Path("data/checkpoints/model_latest.pt"),
    num_games=100,
    num_workers=10,
    mcts_simulations=800
)

# Use examples for training...
```

### Option 2: Integrate with Trainer

Modify `mandala_rl/training/trainer.py`:

```python
from workflows.client import DistributedSelfPlayClient

class Trainer:
    def __init__(self, ..., use_distributed=False, flyte_config=None):
        self.use_distributed = use_distributed
        if use_distributed:
            self.distributed_client = DistributedSelfPlayClient(flyte_config)

    def _generate_selfplay_games(self):
        if self.use_distributed:
            # Use distributed workers
            examples = self.distributed_client.generate_games_distributed(
                checkpoint_path=self.checkpoint_dir / "model_latest.pt",
                num_games=self.config.get('games_per_iteration', 100),
                num_workers=self.config.get('num_workers', 10),
                mcts_simulations=self.config.get('mcts_simulations', 800)
            )
            # Convert to games format...
        else:
            # Use local self-play (current implementation)
            ...
```

### Option 3: Command Line

```bash
# Test locally (4 workers on your machine)
pyflyte run workflows/distributed_selfplay.py \
  distributed_selfplay_from_dict \
  --checkpoint_file data/checkpoints/model_latest.pt \
  --total_games 20 \
  --num_workers 4

# Execute on Flyte cluster
flytectl create execution \
  --project mandala-rl \
  --domain production \
  --workflow workflows.distributed_selfplay.distributed_selfplay_from_dict \
  --inputs checkpoint_file=s3://bucket/model_latest.pt,total_games=100,num_workers=10
```

## Performance

**Current (local sequential):**
- 100 games × 3 min/game = 300 minutes (~5 hours)
- MacBook unusable during training
- Single-threaded

**With distributed workflow:**
- 100 games ÷ 10 workers × 3 min/game = 30 minutes
- MacBook fully usable
- Easily scalable (20 workers = 15 min, 50 workers = 6 min)

## Cost Estimation

Assuming cloud CPU instances at $0.10/hour:
- 10 workers × 0.5 hours = $0.50 per iteration
- 100 iterations = $50 for full training run

Compare to:
- Your time saved: ~70 hours per 100 iterations
- MacBook availability: Can work while training

## Monitoring

View execution status:
- Flyte Console: `https://your-flyte-server.com/console`
- CLI: `flytectl get execution <execution-id>`

## Troubleshooting

**Worker OOM (Out of Memory):**
- Reduce `mcts_simulations` (800 → 400)
- Increase worker memory in task decorator:
  ```python
  @task(requests=Resources(cpu="4", mem="16Gi"))  # Was 8Gi
  ```

**Slow game generation:**
- Games naturally vary in length (some take 5+ minutes)
- Use more workers to average out variance
- Monitor in Flyte console to identify slow workers

**Checkpoint upload/download slow:**
- Use cloud storage in same region as Flyte cluster
- Enable compression in checkpoint saving
- Consider checkpoint size (network.state_dict only)

## Next Steps

1. **Hyperparameter sweep workflow**: Test multiple configs in parallel
2. **Continuous training**: Auto-trigger on checkpoint upload
3. **Tournament evaluation**: Round-robin between checkpoints
4. **Elo ladder**: Automated rating updates
