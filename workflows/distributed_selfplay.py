"""
Flyte workflow for distributed self-play game generation.

This workflow distributes self-play game generation across multiple workers,
dramatically speeding up the training loop bottleneck.

Usage:
    # From your MacBook after training:
    curl -X POST https://your-flyte-server/api/v1/workflows/distributed_selfplay \
      -H "Content-Type: application/json" \
      -d '{
        "checkpoint_url": "s3://bucket/model_iter_5.pt",
        "total_games": 100,
        "num_workers": 10,
        "config": {...}
      }'
"""
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from flytekit import task, workflow, Resources, dynamic
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory

# Import your game modules
# NOTE: Your mandala_rl package needs to be installed in the Flyte container
from mandala_rl.game.engine import MandalaGame
from mandala_rl.game.state import GameState
from mandala_rl.network.model import MandalaNet
from mandala_rl.selfplay.worker import SelfPlayWorker, SelfPlayGame


@dataclass
class SelfPlayConfig:
    """Configuration for self-play workers."""
    mcts_simulations: int = 800
    temperature: float = 1.0
    temperature_threshold: int = 30
    c_puct: float = 1.0
    num_res_blocks: int = 8
    channels: int = 96
    input_channels: int = 96
    num_actions: int = 150


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    cache=True,
    cache_version="1.0"
)
def generate_games_batch(
    checkpoint_file: FlyteFile,
    num_games: int,
    config: SelfPlayConfig,
    worker_id: int
) -> FlyteFile:
    """
    Generate a batch of self-play games on a single worker.

    Args:
        checkpoint_file: Network checkpoint file
        num_games: Number of games to generate
        config: Self-play configuration
        worker_id: Worker identifier for logging

    Returns:
        FlyteFile containing pickled list of SelfPlayGame objects
    """
    print(f"Worker {worker_id}: Starting generation of {num_games} games")

    # Download checkpoint to local file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        checkpoint_path = tmp.name
        checkpoint_file.download(checkpoint_path)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create network
    network = MandalaNet(
        input_channels=config.input_channels,
        num_actions=config.num_actions,
        num_res_blocks=config.num_res_blocks,
        channels=config.channels
    )
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    # Create game engine
    game = MandalaGame()

    # Create self-play worker (use CPU for cloud workers)
    worker = SelfPlayWorker(
        game=game,
        network=network,
        mcts_simulations=config.mcts_simulations,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
        c_puct=config.c_puct,
        device='cpu'
    )

    # Generate games
    games = []
    for i in range(num_games):
        print(f"Worker {worker_id}: Playing game {i+1}/{num_games}")
        game_result = worker.play_game()
        games.append(game_result)

    # Save games to file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as tmp:
        pickle.dump(games, tmp)
        output_path = tmp.name

    print(f"Worker {worker_id}: Completed {num_games} games")
    return FlyteFile(path=output_path)


@task(
    requests=Resources(cpu="2", mem="4Gi")
)
def aggregate_games(
    game_batches: List[FlyteFile],
    config: SelfPlayConfig
) -> FlyteFile:
    """
    Aggregate games from all workers and convert to training examples.

    Args:
        game_batches: List of pickled game batch files from workers
        config: Self-play configuration

    Returns:
        FlyteFile containing pickled training examples
    """
    print(f"Aggregating {len(game_batches)} game batches")

    # Load all games
    all_games = []
    for batch_file in game_batches:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            batch_path = tmp.name
            batch_file.download(batch_path)

        with open(batch_path, 'rb') as f:
            games = pickle.load(f)
            all_games.extend(games)

    print(f"Total games collected: {len(all_games)}")

    # Convert to training examples
    # Create a dummy worker just to use get_training_examples method
    game = MandalaGame()
    network = MandalaNet(
        input_channels=config.input_channels,
        num_actions=config.num_actions,
        num_res_blocks=config.num_res_blocks,
        channels=config.channels
    )
    worker = SelfPlayWorker(
        game=game,
        network=network,
        device='cpu'
    )

    all_examples = []
    for game_record in all_games:
        examples = worker.get_training_examples(game_record)
        all_examples.extend(examples)

    print(f"Total training examples: {len(all_examples)}")

    # Save training examples
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False, mode='wb') as tmp:
        pickle.dump(all_examples, tmp)
        output_path = tmp.name

    return FlyteFile(path=output_path)


@workflow
def distributed_selfplay_workflow(
    checkpoint_file: FlyteFile,
    total_games: int = 100,
    num_workers: int = 10,
    config: SelfPlayConfig = SelfPlayConfig()
) -> FlyteFile:
    """
    Distributed self-play workflow.

    This workflow distributes self-play game generation across multiple workers,
    then aggregates the results into training examples.

    Args:
        checkpoint_file: Network checkpoint to use for self-play
        total_games: Total number of games to generate
        num_workers: Number of parallel workers
        config: Self-play configuration

    Returns:
        FlyteFile containing pickled training examples
        Format: List[Tuple[np.ndarray, np.ndarray, float]]
                (state_tensor, policy, value)

    Example:
        # Upload checkpoint
        checkpoint = FlyteFile(path="data/checkpoints/model_latest.pt")

        # Run workflow
        examples_file = distributed_selfplay_workflow(
            checkpoint_file=checkpoint,
            total_games=100,
            num_workers=10
        )

        # Download and use examples
        examples_file.download("training_examples.pkl")
        with open("training_examples.pkl", "rb") as f:
            examples = pickle.load(f)
    """
    # Calculate games per worker
    games_per_worker = total_games // num_workers
    remainder = total_games % num_workers

    # Generate game batches in parallel
    game_batches = []
    for i in range(num_workers):
        # Distribute remainder games across first few workers
        num_games = games_per_worker + (1 if i < remainder else 0)

        batch = generate_games_batch(
            checkpoint_file=checkpoint_file,
            num_games=num_games,
            config=config,
            worker_id=i
        )
        game_batches.append(batch)

    # Aggregate all games into training examples
    training_examples = aggregate_games(
        game_batches=game_batches,
        config=config
    )

    return training_examples


# Convenience workflow with config from dict
@workflow
def distributed_selfplay_from_dict(
    checkpoint_file: FlyteFile,
    total_games: int = 100,
    num_workers: int = 10,
    mcts_simulations: int = 800,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    c_puct: float = 1.0,
    num_res_blocks: int = 8,
    channels: int = 96
) -> FlyteFile:
    """
    Distributed self-play workflow with individual config parameters.

    This is a convenience wrapper for the main workflow that accepts
    individual config parameters instead of a SelfPlayConfig object.
    """
    config = SelfPlayConfig(
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        temperature_threshold=temperature_threshold,
        c_puct=c_puct,
        num_res_blocks=num_res_blocks,
        channels=channels
    )

    return distributed_selfplay_workflow(
        checkpoint_file=checkpoint_file,
        total_games=total_games,
        num_workers=num_workers,
        config=config
    )


if __name__ == "__main__":
    # For local testing
    print("Flyte workflow defined. Register with:")
    print("  pyflyte register workflows/distributed_selfplay.py")
    print("\nTest locally with:")
    print("  pyflyte run workflows/distributed_selfplay.py distributed_selfplay_workflow \\")
    print("    --checkpoint_file data/checkpoints/model_latest.pt \\")
    print("    --total_games 20 \\")
    print("    --num_workers 4")
