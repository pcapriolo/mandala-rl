"""
Standalone distributed self-play workflow with all dependencies bundled.

This workflow is self-contained and doesn't require installing mandala_rl.
All necessary code is included directly in this file.
"""
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from flytekit import task, workflow, Resources
from flytekit.types.file import FlyteFile

# NOTE: You'll need to copy the actual implementation code here
# OR install mandala_rl as a package in the Docker container

@dataclass
class SelfPlayConfig:
    """Configuration for self-play workers."""
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30
    c_puct: float = 1.0
    num_res_blocks: int = 8
    channels: int = 96
    input_channels: int = 50
    num_actions: int = 30
    checkpoint_url: str = ""


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0",
    container_image="{{.image}}"  # Uses image from registration
)
def generate_games_batch(
    checkpoint_url: str,
    num_games: int,
    config: SelfPlayConfig,
    worker_id: int
) -> str:
    """
    Generate self-play games and return serialized results as string.

    Since FlyteFile requires actual files and we're bundling code,
    we'll return a base64-encoded pickle string for simplicity.
    """
    import base64

    print(f"Worker {worker_id}: Starting generation of {num_games} games")
    print(f"Config: {config}")

    # Download checkpoint
    print(f"Downloading checkpoint from {checkpoint_url}")
    import urllib.request
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        urllib.request.urlretrieve(checkpoint_url, tmp.name)
        checkpoint_path = tmp.name

    print(f"Checkpoint downloaded to {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint loaded: iteration {checkpoint.get('iteration', 'unknown')}")

    # TODO: Import and use actual MandalaGame, MandalaNet, SelfPlayWorker
    # For now, return mock data to test the workflow structure

    mock_games = [
        {
            'states': [np.random.rand(50, 8, 8) for _ in range(10)],
            'policies': [np.random.rand(30) for _ in range(10)],
            'outcome': np.random.choice([-1, 0, 1])
        }
        for _ in range(num_games)
    ]

    print(f"Worker {worker_id}: Generated {num_games} games")

    # Serialize and encode
    serialized = pickle.dumps(mock_games)
    encoded = base64.b64encode(serialized).decode('utf-8')

    return encoded


@task(requests=Resources(cpu="2", mem="4Gi"))
def aggregate_games(
    game_batches: List[str],
    config: SelfPlayConfig
) -> str:
    """
    Aggregate games from all workers.

    Returns base64-encoded pickle string of training examples.
    """
    import base64

    print(f"Aggregating {len(game_batches)} game batches")

    # Decode all batches
    all_games = []
    for encoded_batch in game_batches:
        decoded = base64.b64decode(encoded_batch.encode('utf-8'))
        games = pickle.loads(decoded)
        all_games.extend(games)

    print(f"Total games collected: {len(all_games)}")

    # Convert to training examples
    all_examples = []
    for game in all_games:
        for state, policy in zip(game['states'], game['policies']):
            # Example format: (state, policy, value)
            all_examples.append((state, policy, game['outcome']))

    print(f"Total training examples: {len(all_examples)}")

    # Serialize and encode
    serialized = pickle.dumps(all_examples)
    encoded = base64.b64encode(serialized).decode('utf-8')

    return encoded


@workflow
def distributed_selfplay_workflow(
    checkpoint_url: str,
    total_games: int = 100,
    num_workers: int = 10,
    mcts_simulations: int = 800,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    c_puct: float = 1.0,
    num_res_blocks: int = 8,
    channels: int = 96
) -> str:
    """
    Distributed self-play workflow.

    Returns base64-encoded pickle string of training examples.
    """
    config = SelfPlayConfig(
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        temperature_threshold=temperature_threshold,
        c_puct=c_puct,
        num_res_blocks=num_res_blocks,
        channels=channels,
        checkpoint_url=checkpoint_url
    )

    # Calculate games per worker
    games_per_worker = total_games // num_workers
    remainder = total_games % num_workers

    # Generate game batches in parallel
    game_batches = []
    for i in range(num_workers):
        num_games = games_per_worker + (1 if i < remainder else 0)

        batch = generate_games_batch(
            checkpoint_url=checkpoint_url,
            num_games=num_games,
            config=config,
            worker_id=i
        )
        game_batches.append(batch)

    # Aggregate results
    training_examples = aggregate_games(
        game_batches=game_batches,
        config=config
    )

    return training_examples


if __name__ == "__main__":
    print("Standalone Flyte workflow defined.")
    print("\nThis workflow uses base64-encoded strings instead of files")
    print("to avoid FlyteFile complexity during development.")
    print("\nTo register:")
    print("  pyflyte register workflows/distributed_selfplay_standalone.py")
