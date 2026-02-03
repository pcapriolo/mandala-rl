#!/usr/bin/env python3
"""
Play and visualize a Mandala game.

Usage:
    python scripts/play_game.py                    # Random vs Random
    python scripts/play_game.py --delay 1.0        # With 1 second delay per move
    python scripts/play_game.py --seed 42          # Reproducible game
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.game.state import GameState


def random_policy(state: GameState, game: MandalaGame) -> int:
    """Simple random policy for playing."""
    valid_moves = game.get_valid_moves(state)
    valid_indices = np.where(valid_moves > 0)[0]
    if len(valid_indices) == 0:
        return 0
    return np.random.choice(valid_indices)


def action_to_string(action_id: int) -> str:
    """Convert action ID to human-readable string."""
    if action_id < 12:
        # BUILD_MOUNTAIN
        color = action_id // 2
        mandala = action_id % 2
        colors_names = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']
        return f"BUILD_MOUNTAIN: Play {colors_names[color]} to Mandala {mandala}"
    elif action_id < 24:
        # GROW_FIELD
        action_id -= 12
        color = action_id // 2
        mandala = action_id % 2
        colors_names = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']
        return f"GROW_FIELD: Play {colors_names[color]} to Field in Mandala {mandala}"
    else:
        # DISCARD
        color = action_id - 24
        colors_names = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']
        return f"DISCARD: Discard {colors_names[color]} cards"


def play_game(game: MandalaGame, delay: float = 0.5, max_moves: int = 200):
    """
    Play a complete game and display each move.

    Args:
        game: Game engine
        delay: Delay in seconds between moves
        max_moves: Maximum number of moves before stopping
    """
    state = game.get_initial_state()
    move_count = 0

    print("\n" + "="*80)
    print("MANDALA GAME START")
    print("="*80)
    print("\nInitial State:")
    print(game.state_to_string(state))

    if delay > 0:
        input("\nPress Enter to start...")

    while not game.is_terminal(state) and move_count < max_moves:
        move_count += 1

        print("\n" + "="*80)
        print(f"MOVE {move_count}")
        print("="*80)

        # Get valid moves
        valid_moves = game.get_valid_moves(state)
        num_valid = int(np.sum(valid_moves))

        if num_valid == 0:
            print("No valid moves! Game should be terminal.")
            break

        # Choose random action
        action = random_policy(state, game)
        action_str = action_to_string(action)

        print(f"\nPlayer {state.current_player} chooses:")
        print(f"  {action_str}")
        print(f"  (Action ID: {action}, {num_valid} valid moves available)")

        # Apply action
        new_state = game.get_next_state(state, action)

        # Show result
        print(f"\nAfter move:")
        print(game.state_to_string(new_state))

        state = new_state

        if delay > 0:
            time.sleep(delay)

    # Game over
    print("\n" + "="*80)
    print("GAME COMPLETE")
    print("="*80)
    print(f"Total moves: {move_count}")
    print(f"\nFinal state:")
    print(game.state_to_string(state))

    return state


def main():
    parser = argparse.ArgumentParser(description='Play and watch a Mandala game')
    parser.add_argument('--delay', type=float, default=0.5,
                      help='Delay in seconds between moves (0 for no delay)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--max-moves', type=int, default=200,
                      help='Maximum number of moves')
    parser.add_argument('--interactive', action='store_true',
                      help='Pause after each move (press Enter to continue)')
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Create game
    game = MandalaGame(seed=args.seed)

    # Play game
    delay = 0 if args.interactive else args.delay
    final_state = play_game(game, delay=delay, max_moves=args.max_moves)

    # Show final scores
    score0 = game._calculate_score(final_state, 0)
    score1 = game._calculate_score(final_state, 1)

    print("\n" + "="*80)
    print("FINAL SCORES")
    print("="*80)
    print(f"Player 0: {score0} points")
    print(f"Player 1: {score1} points")

    if score0 > score1:
        print("\n🎉 Player 0 WINS!")
    elif score1 > score0:
        print("\n🎉 Player 1 WINS!")
    else:
        print("\n🤝 DRAW!")


if __name__ == '__main__':
    main()
