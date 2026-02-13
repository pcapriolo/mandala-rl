"""
Self-play worker for generating training games.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from pathlib import Path
from ..game.engine import MandalaGame
from ..game.state import GameState
from ..mcts.search import MCTS
from ..mcts.node import MCTSNode
from ..network.model import MandalaNet


class SelfPlayGame:
    """Represents a completed self-play game."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.current_players: List[int] = []
        self.outcome: float = 0.0  # Final game outcome from player 0's perspective

    def __len__(self):
        return len(self.states)


class SelfPlayWorker:
    """
    Worker for generating self-play games.

    Plays games using MCTS + neural network and collects training data.
    """

    def __init__(
        self,
        game: MandalaGame,
        network: MandalaNet,
        mcts_simulations: int = 800,
        temperature: float = 1.0,
        temperature_threshold: int = 30,  # Move to switch to temp=0
        c_puct: float = 1.0,
        device: str = "mps"
    ):
        """
        Args:
            game: Game engine
            network: Neural network
            mcts_simulations: Number of MCTS simulations per move
            temperature: Temperature for action sampling
            temperature_threshold: Move number to switch to deterministic play
            c_puct: MCTS exploration constant
            device: PyTorch device (mps for Apple Silicon)
        """
        self.game = game
        self.network = network.to(device)
        self.network.eval()
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.c_puct = c_puct
        self.device = device

        # Create MCTS instance
        self.mcts = MCTS(
            game=game,
            network=self.network.predict,
            num_simulations=mcts_simulations,
            c_puct=c_puct
        )

    def play_game(self) -> SelfPlayGame:
        """
        Play a single self-play game.

        Returns:
            SelfPlayGame: Completed game with states, policies, and outcome
        """
        game_record = SelfPlayGame()
        state = self.game.get_initial_state()
        move_count = 0

        while not self.game.is_terminal(state):
            # Get canonical state (from current player's perspective)
            canonical_state = state.get_canonical_form()

            # Determine temperature
            temp = self.temperature if move_count < self.temperature_threshold else 0.0

            # Run MCTS
            action_probs, visit_counts = self.mcts.get_action_prob(
                canonical_state,
                temperature=temp,
                add_noise=True
            )

            # Record state and policy
            game_record.states.append(canonical_state.to_tensor())
            game_record.policies.append(action_probs)
            game_record.current_players.append(state.current_player)

            # Sample action
            action = np.random.choice(len(action_probs), p=action_probs)

            # Apply action
            state = self.game.get_next_state(state, action)
            move_count += 1

        # Record outcome
        outcome_p0 = self.game.get_reward(state, player=0)
        outcome_p1 = self.game.get_reward(state, player=1)
        game_record.outcome = outcome_p0

        return game_record

    def generate_games(self, num_games: int) -> List[SelfPlayGame]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to play

        Returns:
            List of completed games
        """
        games = []
        for _ in range(num_games):
            game = self.play_game()
            games.append(game)
        return games

    def get_training_examples(self, game: SelfPlayGame) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert game to training examples.

        Each example is (state, policy, value) where value is the final
        outcome from the perspective of the current player in that state.

        Args:
            game: Completed self-play game

        Returns:
            List of (state_tensor, policy, value) tuples
        """
        examples = []
        outcome = game.outcome

        for i, (state, policy, player) in enumerate(
            zip(game.states, game.policies, game.current_players)
        ):
            # Value from current player's perspective
            value = outcome if player == 0 else -outcome

            examples.append((state, policy, value))

        return examples

    def play_game_with_replay(self, save_dir: Optional[Path] = None, iteration: int = 0):
        """
        Play game and optionally save replay.

        Args:
            save_dir: Directory to save replay (None = don't save)
            iteration: Current training iteration number

        Returns:
            SelfPlayGame with full history
        """
        from ..viewer.replay import GameReplay

        replay = GameReplay() if save_dir else None
        game_record = SelfPlayGame()
        state = self.game.get_initial_state()
        move_count = 0

        while not self.game.is_terminal(state):
            canonical_state = state.get_canonical_form()
            temp = self.temperature if move_count < self.temperature_threshold else 0.0

            action_probs, visit_counts = self.mcts.get_action_prob(
                canonical_state, temperature=temp, add_noise=True
            )

            game_record.states.append(canonical_state.to_tensor())
            game_record.policies.append(action_probs)
            game_record.current_players.append(state.current_player)

            action = np.random.choice(len(action_probs), p=action_probs)

            # Save to replay using game engine methods
            if replay:
                replay.add_move(
                    move_num=move_count + 1,
                    player=state.current_player,
                    action_id=int(action),
                    action_desc=self.game.action_to_string(int(action)),
                    state_summary=self.game.state_to_summary(state)
                )

            state = self.game.get_next_state(state, action)
            move_count += 1

        # Record outcome
        outcome_p0 = self.game.get_reward(state, player=0)
        game_record.outcome = outcome_p0

        # Save replay
        if replay and save_dir:
            score0 = self.game._calculate_score(state, 0)
            score1 = self.game._calculate_score(state, 1)
            winner = 0 if score0 > score1 else (1 if score1 > score0 else None)
            replay.set_result(score0, score1, winner)
            replay.metadata = {'iteration': iteration, 'move_count': move_count}
            replay.save(save_dir)

        return game_record

    def play_games_batched(self, num_games: int, batch_size: int = 8,
                           save_dir: Optional[Path] = None, iteration: int = 0,
                           save_replay_freq: int = 10,
                           on_game_complete=None) -> List[SelfPlayGame]:
        """
        Play multiple games with batched neural network inference.

        Runs batch_size games simultaneously, batching NN calls across all
        active games during MCTS search for ~3-8x speedup.

        Args:
            num_games: Total games to generate
            batch_size: Number of simultaneous games
            save_dir: Directory to save replays
            iteration: Current training iteration
            save_replay_freq: Save replay every N games
            on_game_complete: Callback(game_idx, game_record) called per completion

        Returns:
            List of completed SelfPlayGame objects
        """
        completed = []
        games_started = 0

        # Initialize active game slots
        active = []
        for _ in range(min(batch_size, num_games)):
            save_replay = save_dir and (games_started % save_replay_freq == 0)
            active.append(self._init_game_slot(games_started, save_replay))
            games_started += 1

        while active:
            # Run one batched MCTS move for all active games
            self._batched_mcts_move(active)

            # Check for finished games and replace
            still_active = []
            for ag in active:
                if self.game.is_terminal(ag['state']):
                    game_record = self._finalize_game_slot(ag, save_dir, iteration)
                    completed.append(game_record)
                    if on_game_complete:
                        on_game_complete(ag['game_idx'], game_record)

                    # Start a new game if more remain
                    if games_started < num_games:
                        save_replay = save_dir and (games_started % save_replay_freq == 0)
                        still_active.append(self._init_game_slot(games_started, save_replay))
                        games_started += 1
                else:
                    still_active.append(ag)
            active = still_active

        return completed

    def _init_game_slot(self, game_idx: int, save_replay: bool) -> dict:
        """Initialize an active game slot."""
        from ..viewer.replay import GameReplay
        return {
            'game_idx': game_idx,
            'state': self.game.get_initial_state(),
            'record': SelfPlayGame(),
            'move_count': 0,
            'replay': GameReplay() if save_replay else None,
        }

    def _finalize_game_slot(self, ag: dict, save_dir: Optional[Path],
                            iteration: int) -> SelfPlayGame:
        """Finalize a completed game slot, save replay if needed."""
        record = ag['record']
        state = ag['state']
        record.outcome = self.game.get_reward(state, player=0)

        if ag['replay'] and save_dir:
            score0 = self.game._calculate_score(state, 0)
            score1 = self.game._calculate_score(state, 1)
            winner = 0 if score0 > score1 else (1 if score1 > score0 else None)
            ag['replay'].set_result(score0, score1, winner)
            ag['replay'].metadata = {'iteration': iteration, 'move_count': ag['move_count']}
            ag['replay'].save(save_dir)

        return record

    def _batched_mcts_move(self, active_games: list):
        """
        Run one MCTS search + action selection for all active games,
        batching neural network calls across games.
        """
        n = len(active_games)
        num_actions = self.network.num_actions

        # --- Root expansion (one batch NN call for all roots) ---
        roots = [MCTSNode(prior=1.0) for _ in range(n)]
        canonical_states = [ag['state'].get_canonical_form() for ag in active_games]
        root_policies, _ = self.network.predict_batch(canonical_states)

        for i in range(n):
            policy = root_policies[i].copy()
            # Add Dirichlet noise for exploration
            noise = np.random.dirichlet([self.mcts.dirichlet_alpha] * len(policy))
            policy = (1 - self.mcts.dirichlet_epsilon) * policy + self.mcts.dirichlet_epsilon * noise
            valid_moves = self.game.get_valid_moves(active_games[i]['state'])
            roots[i].expand(policy, valid_moves, state_hash=hash(active_games[i]['state']))

        # --- Run simulations with batched leaf evaluation ---
        for _ in range(self.mcts_simulations):
            # Phase 1: traverse all trees to leaves
            leaves = []
            for i in range(n):
                node, leaf_state, is_terminal = self.mcts.simulate_to_leaf(
                    roots[i], active_games[i]['state']
                )
                leaves.append((node, leaf_state, is_terminal))

            # Phase 2: batch NN call for non-terminal leaves
            non_terminal_indices = [i for i, (_, _, t) in enumerate(leaves) if not t]

            if non_terminal_indices:
                batch_states = [leaves[i][1].get_canonical_form() for i in non_terminal_indices]
                batch_policies, batch_values = self.network.predict_batch(batch_states)

                for j, idx in enumerate(non_terminal_indices):
                    node, leaf_state, _ = leaves[idx]
                    self.mcts.expand_and_backup(
                        node, leaf_state, batch_policies[j], float(batch_values[j]), False
                    )

            # Handle terminal leaves
            terminal_indices = [i for i, (_, _, t) in enumerate(leaves) if t]
            for idx in terminal_indices:
                node, leaf_state, _ = leaves[idx]
                value = self.game.get_reward(leaf_state, leaf_state.current_player)
                self.mcts.expand_and_backup(node, leaf_state, None, value, True)

        # --- Action selection and game advancement ---
        for i in range(n):
            ag = active_games[i]
            root = roots[i]

            # Extract visit counts
            visit_counts = np.zeros(num_actions)
            for action, child in root.children.items():
                visit_counts[action] = child.visit_count

            # Temperature
            temp = self.temperature if ag['move_count'] < self.temperature_threshold else 0.0

            if temp == 0:
                action_probs = np.zeros_like(visit_counts)
                best = np.argmax(visit_counts)
                action_probs[best] = 1.0
            else:
                visit_counts_temp = visit_counts ** (1.0 / temp)
                total = np.sum(visit_counts_temp)
                action_probs = visit_counts_temp / total if total > 0 else np.ones_like(visit_counts) / len(visit_counts)

            # Record training data
            canonical_state = ag['state'].get_canonical_form()
            ag['record'].states.append(canonical_state.to_tensor())
            ag['record'].policies.append(action_probs)
            ag['record'].current_players.append(ag['state'].current_player)

            # Sample action
            action = np.random.choice(len(action_probs), p=action_probs)

            # Record replay
            if ag['replay']:
                ag['replay'].add_move(
                    move_num=ag['move_count'] + 1,
                    player=ag['state'].current_player,
                    action_id=int(action),
                    action_desc=self.game.action_to_string(int(action)),
                    state_summary=self.game.state_to_summary(ag['state'])
                )

            # Apply action
            ag['state'] = self.game.get_next_state(ag['state'], action)
            ag['move_count'] += 1
