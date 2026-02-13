#!/usr/bin/env python3
"""
Web-based Human vs AI gameplay for Lost Cities.

Run this to start a web server where you can play against trained models.
"""
import argparse
import sys
import pickle
import torch
import yaml
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lost_cities.game.engine import LostCitiesGame, COLOR_NAMES, NUM_ACTIONS
from lost_cities.game.state import LostCitiesState, LCCard, NUM_COLORS, HAND_SIZE
from mandala_rl.network.model import MandalaNet
from mandala_rl.mcts.search import MCTS

template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

game_manager = None
config_path = None
mcts_simulations = None


class WebGameManager:
    """Manages web-based human vs AI Lost Cities games."""

    def __init__(self, checkpoint_path, config_path, mcts_simulations=400):
        self.engine = LostCitiesGame()
        self.checkpoint_path = checkpoint_path
        self.mcts_simulations = mcts_simulations

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model from {checkpoint_path}...")
        self.model = MandalaNet(
            input_channels=self.config['network'].get('input_channels', 50),
            num_actions=self.config['network'].get('num_actions', NUM_ACTIONS),
            num_res_blocks=self.config['network']['num_res_blocks'],
            channels=self.config['network']['channels']
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.iteration = checkpoint.get('iteration', 'unknown')
        self.total_games = checkpoint.get('total_games', 'unknown')
        print(f"Model loaded: iteration {self.iteration}, {self.total_games} games trained")

        def network_fn(state):
            state_tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)
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

        self.state = None
        self.human_player = 0
        self.move_count = 0

    def start_new_game(self, human_player=0):
        self.state = self.engine.get_initial_state()
        self.human_player = human_player
        self.move_count = 0
        return self.get_game_state_dict()

    def get_game_state_dict(self):
        if self.state is None:
            return None

        valid_moves = self.engine.get_valid_moves(self.state)
        valid_actions = [
            {'action': int(i), 'description': self._action_to_display(i, self.state)}
            for i, valid in enumerate(valid_moves) if valid
        ]

        is_terminal = self.engine.is_terminal(self.state)
        winner = None
        scores = None

        if is_terminal:
            s0 = self.state.compute_score(0)
            s1 = self.state.compute_score(1)
            scores = {'player0': s0, 'player1': s1}
            winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)

        return {
            'state': self._format_state(),
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

    def _action_to_display(self, action, state):
        """Convert action to display string with card info."""
        hand_pos = action // 12
        dest = (action % 12) // 6
        draw_src = action % 6

        hand = state.hands[state.current_player]
        if hand_pos < len(hand):
            card = hand[hand_pos]
            card_str = f"{'W' if card.value == 0 else card.value} {COLOR_NAMES[card.color]}"
        else:
            card_str = f"Hand[{hand_pos}]"

        dest_str = f"Play to {COLOR_NAMES[hand[hand_pos].color] if hand_pos < len(hand) else '?'} expedition" if dest == 0 else f"Discard"
        draw_str = "Draw from deck" if draw_src == 0 else f"Draw from {COLOR_NAMES[draw_src - 1]} pile"

        return f"{card_str}: {dest_str}, {draw_str}"

    def _format_state(self):
        """Format state for web display."""
        def cards_to_list(cards):
            return [{'color': c.color, 'value': c.value, 'display': repr(c)} for c in cards]

        def exp_summary(exp):
            if not exp:
                return {'cards': [], 'wagers': 0, 'top': 0, 'count': 0}
            return {
                'cards': cards_to_list(exp),
                'wagers': sum(1 for c in exp if c.value == 0),
                'top': max(c.value for c in exp),
                'count': len(exp),
            }

        return {
            'hands': {
                'player0': cards_to_list(self.state.hands[0]),
                'player1': cards_to_list(self.state.hands[1]),
            },
            'expeditions': {
                f'player{p}': {
                    COLOR_NAMES[c].lower(): exp_summary(self.state.expeditions[p][c])
                    for c in range(NUM_COLORS)
                }
                for p in range(2)
            },
            'discard_piles': {
                COLOR_NAMES[c].lower(): cards_to_list(self.state.discard_piles[c])
                for c in range(NUM_COLORS)
            },
            'deck_size': len(self.state.deck),
            'scores': {
                'player0': self.state.compute_score(0),
                'player1': self.state.compute_score(1),
            },
            'turns_played': self.state.turns_played,
        }

    def make_move(self, action):
        if self.state is None or self.engine.is_terminal(self.state):
            return {'error': 'No active game'}

        valid_moves = self.engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': f'Invalid move: {self.engine.action_to_string(action)}'}

        self.state = self.engine.get_next_state(self.state, action)
        self.move_count += 1
        return self.get_game_state_dict()

    def get_ai_move(self):
        if self.state is None or self.engine.is_terminal(self.state):
            return {'error': 'No active game'}

        policy, _ = self.mcts.get_action_prob(self.state, temperature=0.0, add_noise=False)
        valid_moves = self.engine.get_valid_moves(self.state)
        valid_policy = policy * valid_moves
        action = int(valid_policy.argmax())

        top_actions = valid_policy.argsort()[-5:][::-1]
        top_moves = [
            {
                'action': int(a),
                'description': self._action_to_display(a, self.state),
                'probability': float(policy[a] * 100)
            }
            for a in top_actions if valid_moves[a]
        ]

        return {
            'action': action,
            'description': self._action_to_display(action, self.state),
            'top_moves': top_moves
        }


@app.route('/')
def index():
    return render_template('play_vs_ai_lc.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json or {}
    human_player = data.get('human_player', 0)
    state = game_manager.start_new_game(human_player)
    return jsonify(state)

@app.route('/api/state', methods=['GET'])
def get_state():
    state = game_manager.get_game_state_dict()
    return jsonify(state)

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    action = data.get('action')
    if action is None:
        return jsonify({'error': 'No action provided'}), 400
    result = game_manager.make_move(action)
    return jsonify(result)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    ai_decision = game_manager.get_ai_move()
    if 'error' in ai_decision:
        return jsonify(ai_decision), 400
    result = game_manager.make_move(ai_decision['action'])
    result['ai_decision'] = ai_decision
    return jsonify(result)

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    checkpoint_dir = Path("data/lost_cities/checkpoints")
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
        except Exception:
            checkpoints.append({'name': cp.name, 'path': str(cp), 'iteration': '?', 'total_games': '?'})
    return jsonify(checkpoints)


def main():
    parser = argparse.ArgumentParser(description="Web-based Lost Cities vs AI")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/lost_cities.yaml', help='Config file')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--port', type=int, default=5002, help='Port (default: 5002)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host')
    args = parser.parse_args()

    if args.checkpoint is None:
        checkpoint_dir = Path("data/lost_cities/checkpoints")
        if not checkpoint_dir.exists():
            print("No checkpoints found. Train a model first.")
            return
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not checkpoints:
            print("No checkpoints found. Train a model first.")
            return
        checkpoint_path = checkpoints[0]
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    global game_manager, config_path, mcts_simulations
    config_path = args.config
    mcts_simulations = args.simulations

    game_manager = WebGameManager(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mcts_simulations=mcts_simulations
    )

    print(f"\n{'='*60}")
    print(f"LOST CITIES WEB PLAYER")
    print(f"{'='*60}")
    print(f"\nStarting web server on http://{args.host}:{args.port}")
    print(f"Model: Iteration {game_manager.iteration}, {game_manager.total_games} games")
    print(f"\nOpen your browser: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
