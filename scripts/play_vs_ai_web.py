#!/usr/bin/env python3
"""
Web-based Human vs AI gameplay for Mandala RL.

Run this to start a web server where you can play against trained models.
"""

import argparse
import os
import sys
import pickle
import torch
import yaml
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.game.state import GameState
from mandala_rl.network.model import MandalaNet
from mandala_rl.mcts.search import MCTS
from mandala_rl.selfplay.worker import SelfPlayGame

# Set up Flask with correct template directory
template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# Global game state
game_manager = None
config_path = None
mcts_simulations = None


class WebGameManager:
    """Manages web-based human vs AI games."""

    def __init__(self, checkpoint_path, config_path, mcts_simulations=400):
        self.engine = MandalaGame()
        self.checkpoint_path = checkpoint_path
        self.mcts_simulations = mcts_simulations

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

        self.iteration = checkpoint.get('iteration', 'unknown')
        self.total_games = checkpoint.get('total_games', 'unknown')
        print(f"Model loaded: iteration {self.iteration}, {self.total_games} games trained")

        # Create network wrapper for MCTS
        def network_fn(state):
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, value = self.model(state_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.item()
            return policy, value

        self.mcts = MCTS(
            game=self.engine,
            network=network_fn,
            num_simulations=mcts_simulations,
            c_puct=self.config['mcts']['c_puct']
        )

        # Current game state
        self.state = None
        self.human_player = 0
        self.game_history = []
        self.move_count = 0

    def start_new_game(self, human_player=0):
        """Start a new game."""
        self.state = self.engine.get_initial_state()
        self.human_player = human_player
        self.game_history = []
        self.move_count = 0
        return self.get_game_state_dict()

    def get_game_state_dict(self):
        """Convert game state to dictionary for web display."""
        if self.state is None:
            return None

        valid_moves = self.engine.get_valid_moves(self.state)
        valid_actions = [(i, self.action_to_string(i)) for i, valid in enumerate(valid_moves) if valid]

        is_terminal = self.engine.is_terminal(self.state)
        winner = None
        scores = None

        if is_terminal:
            p0_score, p1_score = self.engine.get_scores(self.state)
            scores = {'player0': p0_score, 'player1': p1_score}
            if p0_score > p1_score:
                winner = 0
            elif p1_score > p0_score:
                winner = 1
            else:
                winner = -1

        return {
            'state': self.format_state_for_display(),
            'current_player': self.state.current_player,
            'human_player': self.human_player,
            'valid_moves': valid_actions,
            'is_terminal': is_terminal,
            'winner': winner,
            'scores': scores,
            'move_count': self.move_count,
            'model_info': {
                'iteration': self.iteration,
                'total_games': self.total_games,
                'checkpoint': Path(self.checkpoint_path).name
            }
        }

    def format_state_for_display(self):
        """Format state for web display."""
        color_emojis = ['🔴', '🟢', '🟣', '🟠', '🟡', '⚪']

        def cards_to_emojis(cards):
            return [color_emojis[card.color] for card in cards]

        return {
            'hands': {
                'player0': cards_to_emojis(self.state.hands[0]),
                'player1': cards_to_emojis(self.state.hands[1])
            },
            'rivers': {
                'player0': cards_to_emojis(self.state.rivers[0]),
                'player1': cards_to_emojis(self.state.rivers[1])
            },
            'cups': {
                'player0': len(self.state.cups[0]),
                'player1': len(self.state.cups[1])
            },
            'mandalas': [
                {
                    'mountain': cards_to_emojis(self.state.mountains[0]),
                    'field_p0': cards_to_emojis(self.state.fields[0][0]),
                    'field_p1': cards_to_emojis(self.state.fields[0][1]),
                    'colors': len(set(card.color for card in self.state.mountains[0]))
                },
                {
                    'mountain': cards_to_emojis(self.state.mountains[1]),
                    'field_p0': cards_to_emojis(self.state.fields[1][0]),
                    'field_p1': cards_to_emojis(self.state.fields[1][1]),
                    'colors': len(set(card.color for card in self.state.mountains[1]))
                }
            ],
            'deck_size': len(self.state.deck),
            'discard_size': len(self.state.discard),
            'deck_reshuffled': self.state.deck_reshuffled
        }

    def action_to_string(self, action):
        """Convert action ID to string."""
        color_names = ['Red', 'Green', 'Purple', 'Orange', 'Yellow', 'White']
        color_emojis = ['🔴', '🟢', '🟣', '🟠', '🟡', '⚪']

        if action < 12:
            color_idx = action % 6
            mandala_idx = action // 6
            return f"Play {color_emojis[color_idx]} {color_names[color_idx]} to Mountain {mandala_idx}"
        elif action < 24:
            action -= 12
            color_idx = action % 6
            mandala_idx = action // 6
            return f"Play {color_emojis[color_idx]} {color_names[color_idx]} to Field {mandala_idx}"
        else:
            color_idx = action - 24
            return f"Discard {color_emojis[color_idx]} {color_names[color_idx]}"

    def make_move(self, action):
        """Execute a move and return new state."""
        if self.state is None or self.engine.is_terminal(self.state):
            return {'error': 'No active game'}

        # Validate move
        valid_moves = self.engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': 'Invalid move'}

        # Store state for history
        state_tensor = self.state.to_tensor()
        policy = torch.zeros(self.engine.get_action_size())
        policy[action] = 1.0

        self.game_history.append({
            'state': state_tensor,
            'policy': policy,
            'player': self.state.current_player
        })

        # Execute move
        self.state = self.engine.get_next_state(self.state, action)
        self.move_count += 1

        return self.get_game_state_dict()

    def get_ai_move(self):
        """Get AI's move using MCTS."""
        if self.state is None or self.engine.is_terminal(self.state):
            return {'error': 'No active game'}

        # Run MCTS
        policy = self.mcts.search(self.state, temperature=0.0, add_noise=False)

        # Get best action
        valid_moves = self.engine.get_valid_moves(self.state)
        valid_policy = policy * valid_moves
        action = valid_policy.argmax()

        # Get top moves for display
        top_actions = valid_policy.argsort()[-5:][::-1]
        top_moves = [
            {
                'action': int(a),
                'description': self.action_to_string(a),
                'probability': float(policy[a] * 100)
            }
            for a in top_actions if valid_moves[a]
        ]

        return {
            'action': int(action),
            'description': self.action_to_string(action),
            'top_moves': top_moves
        }

    def save_game(self):
        """Save game history."""
        if not self.game_history:
            return {'error': 'No game to save'}

        save_dir = Path("data/human_games")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"web_game_{timestamp}.pkl"
        filepath = save_dir / filename

        states = [entry['state'] for entry in self.game_history]
        policies = [entry['policy'] for entry in self.game_history]

        is_terminal = self.engine.is_terminal(self.state)
        if is_terminal:
            p0_score, p1_score = self.engine.get_scores(self.state)
            outcome = 1 if p0_score > p1_score else (-1 if p1_score > p0_score else 0)
        else:
            outcome = 0

        game_data = SelfPlayGame(states=states, policies=policies, outcome=outcome)

        with open(filepath, 'wb') as f:
            pickle.dump(game_data, f)

        return {
            'success': True,
            'filename': filename,
            'moves': len(self.game_history)
        }


# Flask routes
@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('play_vs_ai.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    data = request.json or {}
    human_player = data.get('human_player', 0)
    state = game_manager.start_new_game(human_player)
    return jsonify(state)

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    state = game_manager.get_game_state_dict()
    return jsonify(state)

@app.route('/api/move', methods=['POST'])
def make_move():
    """Make a move."""
    data = request.json
    action = data.get('action')
    if action is None:
        return jsonify({'error': 'No action provided'}), 400

    result = game_manager.make_move(action)
    return jsonify(result)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Get and execute AI move."""
    ai_decision = game_manager.get_ai_move()
    if 'error' in ai_decision:
        return jsonify(ai_decision), 400

    # Execute the AI's move
    result = game_manager.make_move(ai_decision['action'])
    result['ai_decision'] = ai_decision
    return jsonify(result)

@app.route('/api/save', methods=['POST'])
def save_game():
    """Save current game."""
    result = game_manager.save_game()
    return jsonify(result)

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available checkpoints."""
    checkpoint_dir = Path("data/checkpoints")
    if not checkpoint_dir.exists():
        return jsonify([])

    checkpoints = []
    for cp in sorted(checkpoint_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
            checkpoints.append({
                'name': cp.name,
                'path': str(cp),
                'iteration': checkpoint.get('iteration', '?'),
                'total_games': checkpoint.get('total_games', '?')
            })
        except:
            checkpoints.append({
                'name': cp.name,
                'path': str(cp),
                'iteration': '?',
                'total_games': '?'
            })

    return jsonify(checkpoints)

@app.route('/api/load_checkpoint', methods=['POST'])
def load_checkpoint():
    """Load a different checkpoint."""
    global game_manager
    data = request.json
    checkpoint_name = data.get('checkpoint')

    if not checkpoint_name:
        return jsonify({'error': 'No checkpoint specified'}), 400

    checkpoint_path = Path("data/checkpoints") / checkpoint_name
    if not checkpoint_path.exists():
        return jsonify({'error': f'Checkpoint not found: {checkpoint_name}'}), 404

    try:
        print(f"Loading new checkpoint: {checkpoint_path}")
        game_manager = WebGameManager(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            mcts_simulations=mcts_simulations
        )
        return jsonify({
            'success': True,
            'checkpoint': checkpoint_name,
            'iteration': game_manager.iteration,
            'total_games': game_manager.total_games
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Web-based Mandala vs AI")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: latest)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run web server (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to (default: 127.0.0.1)')

    args = parser.parse_args()

    # Select checkpoint
    if args.checkpoint is None:
        checkpoint_dir = Path("data/checkpoints")
        if not checkpoint_dir.exists():
            print("No checkpoints found. Please train a model first.")
            return
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not checkpoints:
            print("No checkpoints found. Please train a model first.")
            return
        checkpoint_path = checkpoints[0]
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    # Initialize game manager
    global game_manager, config_path, mcts_simulations
    config_path = args.config
    mcts_simulations = args.simulations

    game_manager = WebGameManager(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mcts_simulations=mcts_simulations
    )

    # Create templates directory if needed
    templates_dir = Path(__file__).parent.parent / 'templates'
    templates_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MANDALA WEB PLAYER")
    print(f"{'='*80}")
    print(f"\n🎮 Starting web server on http://{args.host}:{args.port}")
    print(f"📊 Model: Iteration {game_manager.iteration}, {game_manager.total_games} games")
    print(f"\nOpen your browser and navigate to: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
