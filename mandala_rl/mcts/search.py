"""MCTS algorithm implementation."""
import numpy as np
from typing import Callable
from .node import MCTSNode
from ..game.state import GameState
from ..game.engine import MandalaGame


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses policy and value networks to guide search.
    """

    def __init__(
        self,
        game: MandalaGame,
        network: Callable[[GameState], tuple[np.ndarray, float]],
        num_simulations: int = 800,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Args:
            game: Game engine
            network: Function (state -> (policy, value))
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            temperature: Temperature for action selection
            dirichlet_alpha: Dirichlet noise alpha (for exploration at root)
            dirichlet_epsilon: Fraction of Dirichlet noise to add at root
        """
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, state: GameState, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS from root state.

        Args:
            state: Root game state
            add_noise: Whether to add Dirichlet noise at root (for exploration)

        Returns:
            np.ndarray: Visit count distribution (pi) over actions
        """
        # Initialize root node
        root = MCTSNode(prior=1.0)

        # Get initial policy and valid moves
        canonical_state = state.get_canonical_form()
        policy, _ = self.network(canonical_state)
        valid_moves = self.game.get_valid_moves(canonical_state)

        # Add Dirichlet noise at root for exploration
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise

        # Expand root
        root.expand(policy, valid_moves)

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, state)

        # Return visit count distribution
        visit_counts = np.zeros(len(policy))
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def _simulate(self, root: MCTSNode, root_state: GameState):
        """
        Run a single MCTS simulation from root.

        Phases:
        1. Selection: Traverse tree using UCB
        2. Expansion: Expand leaf with network policy
        3. Evaluation: Evaluate leaf with network value
        4. Backup: Propagate value up tree
        """
        node = root
        state = root_state.copy()
        search_path = [node]

        # Selection: traverse to leaf
        while not node.is_leaf() and not self.game.is_terminal(state):
            action, node = node.select_child(self.c_puct)
            state = self.game.get_next_state(state, action)
            search_path.append(node)

        # Evaluation
        if self.game.is_terminal(state):
            # Terminal node: use true outcome
            value = self.game.get_reward(state, state.current_player)
        else:
            # Leaf node: expand and evaluate with network
            canonical_state = state.get_canonical_form()
            policy, value = self.network(canonical_state)
            valid_moves = self.game.get_valid_moves(canonical_state)
            node.expand(policy, valid_moves)

        # Backup
        node.backup(value)

    def get_action_prob(
        self,
        state: GameState,
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get action probability distribution from MCTS.

        Args:
            state: Current game state
            temperature: Temperature for action selection (0 = deterministic)
            add_noise: Whether to add Dirichlet noise

        Returns:
            (action_probs, visit_counts) tuple
        """
        visit_counts = self.search(state, add_noise=add_noise)

        if temperature == 0:
            # Deterministic: choose most visited action
            action_probs = np.zeros_like(visit_counts)
            action_probs[np.argmax(visit_counts)] = 1.0
        else:
            # Stochastic: sample proportional to visit counts^(1/T)
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            action_probs = visit_counts_temp / np.sum(visit_counts_temp)

        return action_probs, visit_counts
