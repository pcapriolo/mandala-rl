"""
Client for triggering distributed self-play workflows from your MacBook.

This replaces the local self-play generation with distributed cloud workers.
"""
import pickle
import tempfile
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from flytekit.remote import FlyteRemote
from flytekit.configuration import Config
from flytekit.types.file import FlyteFile


class DistributedSelfPlayClient:
    """Client for distributed self-play via Flyte."""

    def __init__(self, flyte_config: Config):
        """
        Initialize client.

        Args:
            flyte_config: Flyte configuration
                Config(
                    platform=PlatformConfig(endpoint="your-flyte-server.com")
                )
        """
        self.remote = FlyteRemote(config=flyte_config, default_project="mandala-rl")

    def generate_games_distributed(
        self,
        checkpoint_path: Path,
        num_games: int = 100,
        num_workers: int = 10,
        mcts_simulations: int = 800,
        poll_interval: int = 30
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate self-play games using distributed workers.

        Args:
            checkpoint_path: Local path to checkpoint file
            num_games: Total number of games to generate
            num_workers: Number of parallel workers
            mcts_simulations: MCTS simulations per move
            poll_interval: Seconds between status checks

        Returns:
            List of training examples (state, policy, value)
        """
        print(f"Uploading checkpoint: {checkpoint_path}")
        checkpoint_file = FlyteFile(path=str(checkpoint_path))

        # Fetch workflow
        workflow = self.remote.fetch_workflow(
            name="workflows.distributed_selfplay.distributed_selfplay_from_dict"
        )

        print(f"Launching distributed self-play:")
        print(f"  Games: {num_games}")
        print(f"  Workers: {num_workers}")
        print(f"  MCTS simulations: {mcts_simulations}")

        # Execute workflow
        execution = self.remote.execute(
            workflow,
            inputs={
                "checkpoint_file": checkpoint_file,
                "total_games": num_games,
                "num_workers": num_workers,
                "mcts_simulations": mcts_simulations
            }
        )

        print(f"Execution started: {execution.id.name}")
        print(f"Monitor at: {self.remote.generate_console_url(execution)}")

        # Poll for completion
        print("Waiting for completion...")
        start_time = time.time()
        while not execution.is_done:
            elapsed = time.time() - start_time
            print(f"  Status: {execution.closure.phase} (elapsed: {elapsed:.0f}s)")
            time.sleep(poll_interval)
            execution = self.remote.sync(execution)

        if not execution.is_done or execution.error:
            raise Exception(f"Workflow failed: {execution.error}")

        elapsed = time.time() - start_time
        print(f"Workflow completed in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")

        # Download results
        print("Downloading training examples...")
        examples_file = execution.outputs["o0"]

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            examples_path = tmp.name
            examples_file.download(examples_path)

        with open(examples_path, 'rb') as f:
            examples = pickle.load(f)

        print(f"Downloaded {len(examples)} training examples")
        return examples


def example_usage():
    """Example of how to use the distributed client."""
    from flytekit.configuration import Config, PlatformConfig

    # Configure Flyte connection
    config = Config(
        platform=PlatformConfig(
            endpoint="your-flyte-server.com",  # Replace with your Flyte endpoint
            insecure=False  # Set to True for local/testing
        )
    )

    # Create client
    client = DistributedSelfPlayClient(config)

    # Generate games
    examples = client.generate_games_distributed(
        checkpoint_path=Path("data/checkpoints/model_latest.pt"),
        num_games=100,
        num_workers=10,
        mcts_simulations=800
    )

    print(f"Generated {len(examples)} training examples")
    print(f"Example format: state shape={examples[0][0].shape}, "
          f"policy shape={examples[0][1].shape}, value={examples[0][2]}")


if __name__ == "__main__":
    example_usage()
