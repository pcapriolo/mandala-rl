"""Replay buffer for storing training examples."""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from collections import deque

# Color channel groups in 121-channel Mandala tensor (each group has 6 channels, one per color)
_COLOR_GROUPS = [
    (0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42),
    (42, 48), (48, 54), (59, 65), (65, 71), (71, 77), (77, 83),
    (84, 90), (90, 96), (96, 102), (102, 108), (108, 114), (114, 120),
]
# Scalar channels NOT permuted: 54, 55, 56, 57, 58, 83, 120


def augment_color_permutation(state, policy, belief):
    """Apply random color permutation to state tensor, policy, and belief labels.

    Mandala has 6 symmetric colors — permuting them is a free 720x data multiplier.
    Returns (augmented_state, augmented_policy, augmented_belief).
    """
    perm = np.random.permutation(6)

    # Permute state channels
    aug_state = state.copy()
    for start, end in _COLOR_GROUPS:
        for i in range(6):
            aug_state[start + perm[i]] = state[start + i]

    # Permute policy actions
    aug_policy = policy.copy()
    # BUILD_MOUNTAIN: mandala*6 + color (actions 0-11)
    for mandala in range(2):
        for c in range(6):
            aug_policy[mandala * 6 + perm[c]] = policy[mandala * 6 + c]
    # GROW_FIELD: 12 + mandala*42 + color*7 + (count-1) (actions 12-95)
    for mandala in range(2):
        for c in range(6):
            for cnt in range(7):
                src = 12 + mandala * 42 + c * 7 + cnt
                dst = 12 + mandala * 42 + perm[c] * 7 + cnt
                aug_policy[dst] = policy[src]
    # DISCARD: 96 + color*8 + (count-1) (actions 96-143)
    for c in range(6):
        for cnt in range(8):
            aug_policy[96 + perm[c] * 8 + cnt] = policy[96 + c * 8 + cnt]
    # CLAIM_COLOR: 144 + color (actions 144-149)
    for c in range(6):
        aug_policy[144 + perm[c]] = policy[144 + c]

    # Permute belief labels (12 = 6 hand colors + 6 cup colors)
    aug_belief = belief.copy()
    for c in range(6):
        aug_belief[perm[c]] = belief[c]
        aug_belief[6 + perm[c]] = belief[6 + c]

    return aug_state, aug_policy, aug_belief


def augment_color_permutation_batch(states, policies, beliefs):
    """Batch-vectorized color permutation — same permutation applied to entire batch.

    ~100x faster than per-example loop. One random perm per batch is sufficient
    since there are 720 possible permutations and thousands of batches per epoch.
    """
    perm = np.random.permutation(6)

    # Build channel index map: aug[:, dst] = orig[:, channel_map[dst]]
    channel_map = np.arange(states.shape[1])
    for start, _ in _COLOR_GROUPS:
        for i in range(6):
            channel_map[start + perm[i]] = start + i

    # Build policy index map: aug[:, dst] = orig[:, policy_map[dst]]
    num_actions = policies.shape[1]
    policy_map = np.arange(num_actions)
    for mandala in range(2):
        for c in range(6):
            policy_map[mandala * 6 + perm[c]] = mandala * 6 + c
    for mandala in range(2):
        for c in range(6):
            for cnt in range(7):
                policy_map[12 + mandala * 42 + perm[c] * 7 + cnt] = 12 + mandala * 42 + c * 7 + cnt
    for c in range(6):
        for cnt in range(8):
            policy_map[96 + perm[c] * 8 + cnt] = 96 + c * 8 + cnt
        policy_map[144 + perm[c]] = 144 + c

    # Build belief index map
    belief_map = np.arange(12)
    for c in range(6):
        belief_map[perm[c]] = c
        belief_map[6 + perm[c]] = 6 + c

    # Apply all at once via numpy fancy indexing (no Python loop over batch)
    return states[:, channel_map], policies[:, policy_map], beliefs[:, belief_map]


class ReplayBuffer:
    """
    Circular replay buffer for training examples.

    Stores (state, policy, value, score, belief) tuples from self-play games.
    """

    def __init__(self, max_size: int = 500000):
        """
        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_examples(self, examples: List[Tuple]):
        """
        Add training examples to buffer.

        Args:
            examples: List of (state, policy, value, score, belief) or shorter tuples
        """
        for ex in examples:
            if len(ex) == 3:
                self.buffer.append((ex[0], ex[1], ex[2], 0.0, np.zeros(12, dtype=np.float32)))
            elif len(ex) == 4:
                self.buffer.append((ex[0], ex[1], ex[2], ex[3], np.zeros(12, dtype=np.float32)))
            else:
                self.buffer.append(ex)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            (states, policies, values, scores, beliefs) tuple of batched arrays
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([ex[0] for ex in batch])
        policies = np.array([ex[1] for ex in batch])
        values = np.array([ex[2] for ex in batch])
        scores = np.array([ex[3] for ex in batch])
        beliefs = np.array([ex[4] for ex in batch])

        return states, policies, values, scores, beliefs

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
