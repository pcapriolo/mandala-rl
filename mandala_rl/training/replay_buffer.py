"""Replay buffer for storing training examples."""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from collections import deque


class ReplayBuffer:
    """
    Circular replay buffer for training examples.

    Stores (state, policy, value) tuples from self-play games.
    """

    def __init__(self, max_size: int = 500000):
        """
        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_examples(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Add training examples to buffer.

        Args:
            examples: List of (state, policy, value) tuples
        """
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            (states, policies, values) tuple of batched arrays
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([ex[0] for ex in batch])
        policies = np.array([ex[1] for ex in batch])
        values = np.array([ex[2] for ex in batch])

        return states, policies, values

    def save(self, filepath: Path):
        """Save buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filepath: Path):
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            examples = pickle.load(f)
            self.buffer = deque(examples, maxlen=self.max_size)

    def get_all_data(self) -> list:
        """Get all data for checkpointing."""
        return list(self.buffer)

    def load_data(self, data: list):
        """Load data from checkpoint."""
        self.buffer = deque(data, maxlen=self.max_size)

    def __len__(self):
        return len(self.buffer)
