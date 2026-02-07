#!/usr/bin/env python3
"""
Human vs AI gameplay for Mandala RL.

Allows humans to play against trained models for:
1. Validating bot quality
2. Generating expert training data
3. Testing specific scenarios
"""

import argparse
import os
import sys
import pickle
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.game.state import GameState
from mandala_rl.network.model import MandalaNet
from mandala_rl.mcts.search import MCTS
from mandala_rl.selfplay.worker import SelfPlayGame
import yaml


class HumanVsAI:
    def __init__(self, checkpoint_path, config_path='configs/default.yaml', mcts_simulations=800, show_mcts_stats=False):
        """Initialize human vs AI game.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            mcts_simulations: Number of MCTS simulations per move
            show_mcts_stats: Whether to show MCTS statistics
        """
        self.engine = MandalaGame()
        self.show_mcts_stats = show_mcts_stats

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = MandalaNet(
            input_channels=self.config['network'].get('input_channels', 50),
            num_actions=self.config['network'].get('num_actions', 256),
            num_res_blocks=self.config['network']['num_res_blocks'],
            channels=self.config['network']['channels']
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        iteration = checkpoint.get('iteration', 'unknown')
        total_games = checkpoint.get('total_games', 'unknown')
        print(f"Model loaded: iteration {iteration}, {total_games} games trained")

        # Create network wrapper for MCTS
        def network_fn(state):
            """Wrapper to convert GameState to network output."""
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.item()
            return policy, value

        # Set up MCTS
        self.mcts = MCTS(
            game=self.engine,
            network=network_fn,
            num_simulations=mcts_simulations,
            c_puct=self.config['mcts']['c_puct']
        )

        # Game history for saving
        self.game_history = []

    def action_to_string(self, action, state):
        """Convert action ID to human-readable string.

        Action encoding:
        - Actions 0-11: BUILD_MOUNTAIN (6 colors × 2 mandalas)
        - Actions 12-23: GROW_FIELD (6 colors × 2 mandalas)
        - Actions 24-29: DISCARD (6 colors)
        """
        color_names = ['Red', 'Green', 'Purple', 'Orange', 'Yellow', 'White']
        color_emojis = ['🔴', '🟢', '🟣', '🟠', '🟡', '⚪']

        if action < 12:  # BUILD_MOUNTAIN
            color_idx = action % 6
            mandala_idx = action // 6
            return f"BUILD_MOUNTAIN: Play {color_names[color_idx]} {color_emojis[color_idx]} to Mandala {mandala_idx}"
        elif action < 24:  # GROW_FIELD
            action -= 12
            color_idx = action % 6
            mandala_idx = action // 6
            return f"GROW_FIELD: Play {color_names[color_idx]} {color_emojis[color_idx]} to Field in Mandala {mandala_idx}"
        else:  # DISCARD
            color_idx = action - 24
            return f"DISCARD: Discard {color_names[color_idx]} {color_emojis[color_idx]} cards"

    def display_state(self, state):
        """Display the current game state in a nice format."""
        print("\n" + "="*80)
        print(self.engine.state_to_string(state))
        print("="*80)

    def display_valid_moves(self, state):
        """Display valid moves with numbers for selection."""
        valid_moves = self.engine.get_valid_moves(state)
        valid_actions = [i for i, valid in enumerate(valid_moves) if valid]

        print("\nValid moves:")
        for idx, action in enumerate(valid_actions):
            move_desc = self.action_to_string(action, state)
            print(f"  {idx + 1}. {move_desc}")

        return valid_actions

    def get_human_move(self, state):
        """Get move from human player."""
        valid_actions = self.display_valid_moves(state)

        while True:
            try:
                choice = input("\nEnter move number (or 'q' to quit, 's' to save): ").strip()

                if choice.lower() == 'q':
                    return None
                if choice.lower() == 's':
                    return 'save'

                idx = int(choice) - 1
                if 0 <= idx < len(valid_actions):
                    return valid_actions[idx]
                else:
                    print(f"Please enter a number between 1 and {len(valid_actions)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def get_ai_move(self, state, temperature=0.0):
        """Get move from AI using MCTS."""
        print("\nAI is thinking...")

        # Run MCTS
        policy = self.mcts.search(state, temperature=temperature, add_noise=False)

        # Get best action
        valid_moves = self.engine.get_valid_moves(state)
        valid_policy = policy * valid_moves
        action = valid_policy.argmax()

        if self.show_mcts_stats:
            self.display_mcts_stats(state, policy)

        move_desc = self.engine.action_to_string(action, state)
        print(f"AI chooses: {move_desc}")

        return action, policy

    def display_mcts_stats(self, state, policy):
        """Display MCTS statistics for top moves."""
        valid_moves = self.engine.get_valid_moves(state)
        valid_policy = policy * valid_moves

        # Get top 5 moves
        top_actions = valid_policy.argsort()[-5:][::-1]

        print("\nTop AI moves (MCTS visit distribution):")
        for action in top_actions:
            if valid_moves[action]:
                move_desc = self.action_to_string(action, state)
                prob = policy[action]
                print(f"  {prob*100:5.1f}% - {move_desc}")

    def play_game(self, human_player=0):
        """Play a game with human vs AI.

        Args:
            human_player: Which player is human (0 or 1)

        Returns:
            Game history if completed, None if quit
        """
        state = self.engine.get_initial_state()
        self.game_history = []
        move_count = 0

        print(f"\n{'='*80}")
        print(f"GAME START - You are Player {human_player}")
        print(f"{'='*80}")

        while not self.engine.is_terminal(state):
            move_count += 1
            print(f"\n{'='*80}")
            print(f"MOVE {move_count}")
            print(f"{'='*80}")

            self.display_state(state)

            current_player = state.current_player

            if current_player == human_player:
                # Human move
                print(f"\nYour turn (Player {current_player})")
                action = self.get_human_move(state)

                if action is None:
                    print("\nGame quit.")
                    return None
                elif action == 'save':
                    self.save_game_in_progress()
                    continue

                # Store state and action (policy will be uniform for human moves)
                state_tensor = state.to_tensor()
                policy = torch.zeros(self.engine.get_action_size())
                policy[action] = 1.0  # One-hot for human move

            else:
                # AI move
                print(f"\nAI turn (Player {current_player})")
                action, policy = self.get_ai_move(state, temperature=0.0)
                state_tensor = state.to_tensor()

            # Store state, policy for this move
            self.game_history.append({
                'state': state_tensor,
                'policy': policy,
                'player': current_player
            })

            # Execute move
            state = self.engine.get_next_state(state, action)

        # Game over
        print(f"\n{'='*80}")
        print("GAME OVER")
        print(f"{'='*80}")
        self.display_state(state)

        # Get final scores
        player0_score, player1_score = self.engine.get_scores(state)
        print(f"\nFinal Scores:")
        print(f"  Player 0: {player0_score} points")
        print(f"  Player 1: {player1_score} points")

        if player0_score > player1_score:
            winner = 0
            print(f"\n🎉 Player 0 WINS!")
        elif player1_score > player0_score:
            winner = 1
            print(f"\n🎉 Player 1 WINS!")
        else:
            winner = -1
            print(f"\n🤝 DRAW!")

        if winner == human_player:
            print("Congratulations! You beat the AI! 🏆")
        elif winner == 1 - human_player:
            print("The AI wins this time. Better luck next game!")

        # Add outcomes to history
        outcome = 1 if winner == 0 else (-1 if winner == 1 else 0)
        for entry in self.game_history:
            entry['outcome'] = outcome

        return self.game_history

    def save_game_in_progress(self):
        """Save current game in progress."""
        if not self.game_history:
            print("No moves to save yet.")
            return

        filename = input("Enter filename (or Enter for default): ").strip()
        if not filename:
            filename = f"human_game_partial_{len(self.game_history)}_moves.pkl"

        self.save_game_history(filename)

    def save_game_history(self, filename=None):
        """Save game history to file for training."""
        if not self.game_history:
            print("No game history to save.")
            return

        # Create directory if needed
        save_dir = Path("data/human_games")
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"human_game_{len(list(save_dir.glob('*.pkl'))) + 1}.pkl"

        filepath = save_dir / filename

        # Convert to SelfPlayGame format for compatibility with training
        states = []
        policies = []
        for entry in self.game_history:
            states.append(entry['state'])
            policies.append(entry['policy'])

        outcome = self.game_history[0]['outcome'] if self.game_history else 0

        game_data = SelfPlayGame(
            states=states,
            policies=policies,
            outcome=outcome
        )

        with open(filepath, 'wb') as f:
            pickle.dump(game_data, f)

        print(f"\n✓ Game saved to {filepath}")
        print(f"  Moves: {len(self.game_history)}")
        print(f"  This can be loaded into the replay buffer for training!")


def list_available_checkpoints():
    """List all available model checkpoints."""
    checkpoint_dir = Path("data/checkpoints")
    if not checkpoint_dir.exists():
        return []

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Play Mandala against trained AI")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: latest)')
    parser.add_argument('--list', action='store_true',
                        help='List available checkpoints')
    parser.add_argument('--player', type=int, choices=[0, 1], default=0,
                        help='Which player are you? (0 or 1)')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per move (default: 800)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show MCTS statistics for AI moves')
    parser.add_argument('--save', action='store_true',
                        help='Save game after completion')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # List checkpoints
    if args.list:
        checkpoints = list_available_checkpoints()
        if not checkpoints:
            print("No checkpoints found in data/checkpoints/")
            return

        print("Available checkpoints:")
        for i, cp in enumerate(checkpoints):
            # Try to load iteration info
            try:
                checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
                iteration = checkpoint.get('iteration', '?')
                games = checkpoint.get('total_games', '?')
                print(f"  {i+1}. {cp.name}")
                print(f"      Iteration: {iteration}, Games: {games}")
            except:
                print(f"  {i+1}. {cp.name}")
        return

    # Select checkpoint
    if args.checkpoint is None:
        # Use latest
        checkpoints = list_available_checkpoints()
        if not checkpoints:
            print("No checkpoints found. Please train a model first.")
            print("Run: python scripts/train.py --config configs/default.yaml")
            return
        checkpoint_path = checkpoints[0]
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    # Create game
    game = HumanVsAI(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        mcts_simulations=args.simulations,
        show_mcts_stats=args.show_stats
    )

    # Play
    print("\n" + "="*80)
    print("HUMAN vs AI - MANDALA")
    print("="*80)
    print("\nInstructions:")
    print("  - Select moves by entering the number")
    print("  - Type 'q' to quit")
    print("  - Type 's' to save game in progress")
    print("  - Games are automatically saved at the end if --save is used")
    print()

    history = game.play_game(human_player=args.player)

    # Save if requested and game completed
    if history and args.save:
        game.save_game_history()
    elif history:
        save = input("\nSave this game for training? (y/n): ").strip().lower()
        if save == 'y':
            game.save_game_history()


if __name__ == '__main__':
    main()
