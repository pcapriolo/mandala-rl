"""
Webhook-friendly distributed self-play workflow.

This version accepts simple types (strings, ints) that work well with webhook calls.
"""
import pickle
import tempfile
import urllib.request
from typing import Dict, List, Any
import numpy as np
import torch
from flytekit import task, workflow, Resources

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.selfplay.worker import SelfPlayWorker


@task(
    requests=Resources(cpu="4", mem="8Gi"),
    cache=False  # Disable cache for testing
)
def generate_games_batch(
    checkpoint_url: str,
    num_games: int,
    mcts_simulations: int,
    num_res_blocks: int,
    channels: int,
    worker_id: int
) -> List[Dict[str, Any]]:
    """
    Generate self-play games on a single worker.

    Args:
        checkpoint_url: HTTP URL to checkpoint file
        num_games: Number of games to generate
        mcts_simulations: MCTS simulations per move
        num_res_blocks: Network architecture
        channels: Network architecture
        worker_id: Worker ID for logging

    Returns:
        List of game dictionaries with states, policies, outcome
    """
    print(f"Worker {worker_id}: Starting {num_games} games")
    print(f"Checkpoint: {checkpoint_url}")
    print(f"MCTS sims: {mcts_simulations}")

    # Download checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        print(f"Downloading checkpoint...")
        urllib.request.urlretrieve(checkpoint_url, tmp.name)
        checkpoint_path = tmp.name

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")

    # Create network
    network = MandalaNet(
        input_channels=50,
        num_actions=30,
        num_res_blocks=num_res_blocks,
        channels=channels
    )
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    print(f"Network loaded: {sum(p.numel() for p in network.parameters())} parameters")

    # Create game and worker
    game = MandalaGame()
    worker = SelfPlayWorker(
        game=game,
        network=network,
        mcts_simulations=mcts_simulations,
        temperature=1.0,
        temperature_threshold=30,
        c_puct=1.0,
        device='cpu'
    )

    # Generate games
    games = []
    for i in range(num_games):
        print(f"Worker {worker_id}: Playing game {i+1}/{num_games}")
        game_result = worker.play_game()

        # Convert to serializable format
        games.append({
            'states': [s.tolist() for s in game_result.states],
            'policies': [p.tolist() for p in game_result.policies],
            'current_players': game_result.current_players,
            'outcome': float(game_result.outcome)
        })

    print(f"Worker {worker_id}: Completed {num_games} games")
    return games


@task(requests=Resources(cpu="2", mem="4Gi"))
def aggregate_games(
    game_batches: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Aggregate games from all workers into training examples.

    Returns:
        Dict with training examples and statistics
    """
    print(f"Aggregating {len(game_batches)} game batches")

    # Flatten all games
    all_games = []
    for batch in game_batches:
        all_games.extend(batch)

    print(f"Total games: {len(all_games)}")

    # Convert to training examples
    all_examples = []
    for game in all_games:
        outcome = game['outcome']
        for i, (state, policy, player) in enumerate(zip(
            game['states'],
            game['policies'],
            game['current_players']
        )):
            # Value from current player's perspective
            value = outcome if player == 0 else -outcome

            all_examples.append({
                'state': state,
                'policy': policy,
                'value': float(value)
            })

    print(f"Total training examples: {len(all_examples)}")

    return {
        'num_games': len(all_games),
        'num_examples': len(all_examples),
        'examples': all_examples
    }


@workflow
def distributed_selfplay_webhook(
    checkpoint_url: str,
    total_games: int = 100,
    num_workers: int = 10,
    mcts_simulations: int = 800,
    num_res_blocks: int = 8,
    channels: int = 96
) -> Dict[str, Any]:
    """
    Distributed self-play workflow - webhook friendly.

    Args:
        checkpoint_url: HTTP URL to checkpoint file
        total_games: Total games to generate
        num_workers: Number of parallel workers
        mcts_simulations: MCTS simulations per move
        num_res_blocks: Network depth
        channels: Network width

    Returns:
        Dict with training examples and statistics
    """

    print(f"Starting distributed self-play:")
    print(f"  Checkpoint: {checkpoint_url}")
    print(f"  Total games: {total_games}")
    print(f"  Workers: {num_workers}")
    print(f"  MCTS sims: {mcts_simulations}")

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
            mcts_simulations=mcts_simulations,
            num_res_blocks=num_res_blocks,
            channels=channels,
            worker_id=i
        )
        game_batches.append(batch)

    # Aggregate results
    results = aggregate_games(game_batches=game_batches)

    return results


if __name__ == "__main__":
    print("Webhook-friendly distributed self-play workflow")
    print("\nAccepts:")
    print("  - checkpoint_url: string (HTTP URL)")
    print("  - total_games: int")
    print("  - num_workers: int")
    print("  - config: dict")
    print("\nReturns:")
    print("  - Dict with num_games, num_examples, and examples list")
