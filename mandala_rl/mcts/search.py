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
        root.expand(policy, valid_moves, state_hash=hash(state))

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
            # Debug: check if action is valid for current state
            valid_now = self.game.get_valid_moves(state)
            if valid_now[action] == 0:
                print(f"\n=== MCTS TREE MISMATCH ===")
                print(f"Action {action} was stored in tree but is now invalid!")
                print(f"State hash when expanded: {getattr(node, 'expansion_state_hash', 'unknown')}")
                print(f"State hash now: {hash(state)}")
                print(f"Current player: {state.current_player}")
                hand = state.hands[state.current_player]
                print(f"Hand: {[c.color for c in hand]}")
                print(f"Valid moves now: {np.where(valid_now > 0)[0]}")
                print(f"==========================\n")
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
            # CRITICAL: Use valid moves from NON-canonical state
            # Actions will be applied to non-canonical states during traversal
            # so children must be valid for the actual traversal state
            valid_moves = self.game.get_valid_moves(state)
            node.expand(policy, valid_moves, state_hash=hash(state))

        # Backup
        node.backup(value)

    def simulate_to_leaf(self, root: MCTSNode, root_state: GameState):
        """
        Phase 1 of split simulation: traverse tree to leaf node.

        Returns leaf info for external batch NN evaluation.

        Returns:
            (node, state, is_terminal) - leaf node, leaf state, whether terminal
        """
        node = root
        state = root_state.copy()

        # Selection: traverse to leaf
        while not node.is_leaf() and not self.game.is_terminal(state):
            action, node = node.select_child(self.c_puct)
            state = self.game.get_next_state(state, action)

        is_terminal = self.game.is_terminal(state)
        return node, state, is_terminal

    def expand_and_backup(self, node: MCTSNode, state: GameState,
                          policy: np.ndarray, value: float, is_terminal: bool):
        """
        Phase 2 of split simulation: expand leaf and backup value.

        Args:
            node: Leaf node from simulate_to_leaf
            state: Leaf state from simulate_to_leaf
            policy: NN policy output (or None if terminal)
            value: NN value output (or game reward if terminal)
            is_terminal: Whether the leaf is a terminal state
        """
        if not is_terminal:
            valid_moves = self.game.get_valid_moves(state)
            node.expand(policy, valid_moves, state_hash=hash(state))
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
