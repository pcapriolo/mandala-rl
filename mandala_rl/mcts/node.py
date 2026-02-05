"""MCTS node representation."""
import numpy as np
from typing import Optional, Dict


class MCTSNode:
    """
    Node in the Monte Carlo search tree.

    Each node stores:
    - Visit count N(s, a)
    - Total action value W(s, a)
    - Mean action value Q(s, a) = W(s, a) / N(s, a)
    - Prior probability P(s, a) from policy network
    """

    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}

        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is root (no parent)."""
        return self.parent is None

    def get_value(self) -> float:
        """Get mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct: float) -> tuple[int, 'MCTSNode']:
        """
        Select child with highest UCB score.

        UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum(N(s, b))) / (1 + N(s, a))

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node) tuple
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total = np.sqrt(total_visits)

        for action, child in self.children.items():
            # Q value (exploitation)
            q_value = child.get_value()

            # U value (exploration)
            u_value = c_puct * child.prior * sqrt_total / (1 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: np.ndarray, valid_moves: np.ndarray, state_hash: int = 0):
        """
        Expand node by adding children for all valid actions.

        Args:
            action_priors: Policy network output (prior probabilities)
            valid_moves: Binary mask of valid moves
            state_hash: Hash of state for debugging
        """
        # Store state hash for debugging
        self.expansion_state_hash = state_hash

        # Mask invalid moves
        action_priors = action_priors * valid_moves

        # Renormalize
        prob_sum = np.sum(action_priors)
        if prob_sum > 0:
            action_priors = action_priors / prob_sum
        else:
            # Uniform over valid moves
            action_priors = valid_moves / np.sum(valid_moves)

        # Create child nodes
        for action in range(len(action_priors)):
            if valid_moves[action] > 0:
                self.children[action] = MCTSNode(
                    prior=action_priors[action],
                    parent=self
                )

    def update(self, value: float):
        """
        Update node statistics.

        Args:
            value: Value from current player's perspective
        """
        self.visit_count += 1
        self.total_value += value

    def backup(self, value: float):
        """
        Backup value up the tree to root.

        Args:
            value: Terminal value or value estimate
        """
        node = self
        while node is not None:
            node.update(value)
            value = -value  # Flip value for opponent
            node = node.parent
