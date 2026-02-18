#!/usr/bin/env python3
"""
Web-based Human vs AI gameplay for Mandala RL.

Can run standalone (python scripts/play_vs_ai_web.py) or as a Blueprint
imported by the combined server (serve.py).
"""

import argparse
import json
import sys
import time
import uuid
import torch
import yaml
import numpy as np
from pathlib import Path
from flask import Flask, Blueprint, render_template, jsonify, request
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.game.engine import MandalaGame
from mandala_rl.network.model import MandalaNet
from mandala_rl.mcts.search import MCTS

template_dir = Path(__file__).parent.parent / 'templates'

COLOR_NAMES = ['Red', 'Green', 'Purple', 'Orange', 'Yellow', 'White']
COLOR_SHORT = ['R', 'G', 'P', 'O', 'Y', 'W']


def action_to_string(action):
    """Convert action ID to short string for experienced players."""
    if action < 0 or action >= 30:
        return f"Invalid: {action}"
    if action < 12:
        color_idx = action % 6
        mandala_idx = action // 6
        return f"{COLOR_SHORT[color_idx]} → Mt{mandala_idx}"
    elif action < 24:
        action_offset = action - 12
        color_idx = action_offset % 6
        mandala_idx = action_offset // 6
        return f"{COLOR_SHORT[color_idx]} → Fd{mandala_idx}"
    else:
        color_idx = action - 24
        return f"Discard {COLOR_SHORT[color_idx]}"


class MandalaModelServer:
    """Shared model + MCTS. Loaded once, used by all game sessions."""

    def __init__(self, checkpoint_path, config_path, mcts_simulations=400):
        self.checkpoint_path = str(checkpoint_path)
        self.mcts_simulations = mcts_simulations
        self.engine = MandalaGame()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"[Mandala] Using device: {self.device}")

        print(f"[Mandala] Loading model from {checkpoint_path}...")
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
        print(f"[Mandala] Model loaded: iteration {self.iteration}, {self.total_games} games")

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

    def get_ai_move(self, state):
        """Run MCTS on state, return decision dict."""
        t0 = time.time()
        policy, _ = self.mcts.get_action_prob(state, temperature=0.0, add_noise=False)
        think_ms = int((time.time() - t0) * 1000)

        canonical = state.get_canonical_form()
        state_tensor = torch.from_numpy(canonical.to_tensor()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_logits, val = self.model(state_tensor)
            value = val.item()
            raw_policy = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]

        valid_moves = self.engine.get_valid_moves(state)
        valid_policy = policy * valid_moves
        action = int(valid_policy.argmax())

        top_actions = valid_policy.argsort()[-5:][::-1]
        top_moves = [
            {
                'action': int(a),
                'description': action_to_string(a),
                'probability': float(policy[a] * 100)
            }
            for a in top_actions if valid_moves[a]
        ]

        # Raw network policy (before MCTS) for top valid actions
        raw_valid = raw_policy * valid_moves
        raw_top = raw_valid.argsort()[-5:][::-1]
        network_top = [
            {
                'action': int(a),
                'description': action_to_string(a),
                'probability': float(raw_policy[a] * 100)
            }
            for a in raw_top if valid_moves[a]
        ]

        return {
            'action': action,
            'description': action_to_string(action),
            'top_moves': top_moves,
            'network_top': network_top,
            'policy': [float(p) for p in policy],
            'value': float(value),
            'think_time_ms': think_ms,
        }


class MandalaGameSession:
    """Per-player game state. Created on new_game, identified by UUID."""

    def __init__(self, server, human_player=0):
        self.server = server
        self.state = server.engine.get_initial_state()
        self.human_player = human_player
        self.move_count = 0
        self.game_history = []
        self.game_start_time = datetime.now().isoformat()
        self.last_activity = time.time()
        self.last_save_filepath = None

    def get_game_state_dict(self):
        if self.state is None:
            return None

        engine = self.server.engine
        valid_moves = engine.get_valid_moves(self.state)
        valid_actions = [
            (i, action_to_string(i))
            for i, valid in enumerate(valid_moves) if valid
        ]

        is_terminal = engine.is_terminal(self.state)
        winner = None
        scores = None

        if is_terminal:
            p0_score = engine._calculate_score(self.state, 0)
            p1_score = engine._calculate_score(self.state, 1)
            scores = {'player0': p0_score, 'player1': p1_score}
            if p0_score > p1_score:
                winner = 0
            elif p1_score > p0_score:
                winner = 1
            else:
                winner = -1

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
                'iteration': self.server.iteration,
                'total_games': self.server.total_games,
                'checkpoint': Path(self.server.checkpoint_path).name
            }
        }

    def _format_state(self):
        def cards_to_colors(cards):
            return [card.color for card in cards]

        return {
            'hands': {
                'player0': cards_to_colors(self.state.hands[0]),
                'player1': cards_to_colors(self.state.hands[1])
            },
            'rivers': {
                'player0': cards_to_colors(self.state.rivers[0]),
                'player1': cards_to_colors(self.state.rivers[1])
            },
            'cups': {
                'player0': len(self.state.cups[0]),
                'player1': len(self.state.cups[1])
            },
            'mandalas': [
                {
                    'mountain': cards_to_colors(self.state.mountains[0]),
                    'field_p0': cards_to_colors(self.state.fields[0][0]),
                    'field_p1': cards_to_colors(self.state.fields[0][1]),
                    'colors': len(set(card.color for card in self.state.mountains[0]))
                },
                {
                    'mountain': cards_to_colors(self.state.mountains[1]),
                    'field_p0': cards_to_colors(self.state.fields[1][0]),
                    'field_p1': cards_to_colors(self.state.fields[1][1]),
                    'colors': len(set(card.color for card in self.state.mountains[1]))
                }
            ],
            'deck_size': len(self.state.deck),
            'discard_size': len(self.state.discard),
            'deck_reshuffled': self.state.deck_reshuffled
        }

    def make_move(self, action, think_time_ms=None, ai_data=None):
        engine = self.server.engine
        if self.state is None or engine.is_terminal(self.state):
            return {'error': 'No active game'}

        valid_moves = engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': f'Invalid move: {action_to_string(action)}'}

        is_human = self.state.current_player == self.human_player

        move_record = {
            'move_num': self.move_count + 1,
            'player': int(self.state.current_player),
            'is_human': is_human,
            'action': int(action),
            'action_description': action_to_string(action),
            'timestamp': datetime.now().isoformat(),
            'think_time_ms': think_time_ms,
            'ai_policy': None,
            'ai_value': None,
            'ai_top_moves': None,
        }

        if ai_data:
            move_record['ai_policy'] = ai_data.get('policy')
            move_record['ai_value'] = ai_data.get('value')
            move_record['ai_top_moves'] = ai_data.get('top_moves')
            move_record['think_time_ms'] = ai_data.get('think_time_ms')

        self.game_history.append(move_record)
        self.state = engine.get_next_state(self.state, action)
        self.move_count += 1

        return self.get_game_state_dict()

    def save_game(self):
        if not self.game_history:
            return {'error': 'No game to save'}

        save_dir = Path("data/human_games/mandala")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = f"mandala_{timestamp}"
        filename = f"{game_id}.json"
        filepath = save_dir / filename

        engine = self.server.engine
        is_terminal = engine.is_terminal(self.state)
        winner = None
        final_scores = None
        if is_terminal:
            p0 = engine._calculate_score(self.state, 0)
            p1 = engine._calculate_score(self.state, 1)
            final_scores = {'player0': p0, 'player1': p1}
            winner = 0 if p0 > p1 else (1 if p1 > p0 else -1)

        game_data = {
            'game_id': game_id,
            'game': 'mandala',
            'timestamp': self.game_start_time,
            'model_checkpoint': Path(self.server.checkpoint_path).name,
            'model_iteration': self.server.iteration,
            'mcts_simulations': self.server.mcts_simulations,
            'human_player': self.human_player,
            'winner': winner,
            'final_scores': final_scores,
            'total_moves': len(self.game_history),
            'is_complete': is_terminal,
            'moves': self.game_history,
        }

        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)

        self.last_save_filepath = str(filepath)

        return {
            'success': True,
            'filename': filename,
            'moves': len(self.game_history)
        }


def create_mandala_blueprint(checkpoint_path, config_path, simulations=400,
                              base_url='', checkpoint_dir='data/checkpoints'):
    """Create Flask Blueprint for Mandala game with session-based game management."""
    bp = Blueprint('mandala', __name__, template_folder=str(template_dir))
    server = MandalaModelServer(checkpoint_path, config_path, simulations)
    sessions = {}

    def get_session(game_id):
        session = sessions.get(game_id)
        if session:
            session.last_activity = time.time()
        return session

    def cleanup_expired():
        now = time.time()
        expired = [gid for gid, s in sessions.items() if now - s.last_activity > 3600]
        for gid in expired:
            del sessions[gid]

    @bp.route('/')
    def index():
        return render_template('play_vs_ai.html', base_url=base_url)

    @bp.route('/api/info', methods=['GET'])
    def info():
        return jsonify({
            'game': 'mandala',
            'iteration': server.iteration,
            'total_games': server.total_games,
            'checkpoint': Path(server.checkpoint_path).name,
            'active_sessions': len(sessions),
        })

    @bp.route('/api/new_game', methods=['POST'])
    def new_game():
        cleanup_expired()
        data = request.json or {}
        human_player = data.get('human_player', 0)
        game_id = str(uuid.uuid4())
        sessions[game_id] = MandalaGameSession(server, human_player)
        state = sessions[game_id].get_game_state_dict()
        state['game_id'] = game_id
        return jsonify(state)

    @bp.route('/api/state', methods=['GET'])
    def get_state():
        game_id = request.args.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        state = session.get_game_state_dict()
        state['game_id'] = game_id
        return jsonify(state)

    @bp.route('/api/move', methods=['POST'])
    def make_move():
        data = request.json
        game_id = data.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        action = data.get('action')
        if action is None:
            return jsonify({'error': 'No action provided'}), 400
        think_time_ms = data.get('think_time_ms')
        result = session.make_move(action, think_time_ms=think_time_ms)
        result['game_id'] = game_id
        return jsonify(result)

    @bp.route('/api/ai_move', methods=['POST'])
    def ai_move():
        data = request.json or {}
        game_id = data.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404

        ai_decision = server.get_ai_move(session.state)
        if 'error' in ai_decision:
            return jsonify(ai_decision), 400

        ai_data = {
            'policy': ai_decision['policy'],
            'value': ai_decision['value'],
            'top_moves': ai_decision['top_moves'],
            'think_time_ms': ai_decision['think_time_ms'],
        }
        result = session.make_move(ai_decision['action'], ai_data=ai_data)
        result['ai_decision'] = {
            'action': ai_decision['action'],
            'description': ai_decision['description'],
            'top_moves': ai_decision['top_moves'],
        }
        result['game_id'] = game_id
        return jsonify(result)

    @bp.route('/api/save', methods=['POST'])
    def save_game():
        data = request.json or {}
        game_id = data.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        return jsonify(session.save_game())

    @bp.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        data = request.json or {}
        game_id = data.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        if not session.last_save_filepath:
            return jsonify({'error': 'Game not saved yet'}), 400

        filepath = Path(session.last_save_filepath)
        if not filepath.exists():
            return jsonify({'error': 'Save file not found'}), 404

        feedback = {'submitted_at': datetime.now().isoformat()}
        my_rating = data.get('my_play_rating')
        bot_rating = data.get('bot_play_rating')
        comment = data.get('comment', '').strip()
        if my_rating is not None:
            feedback['my_play_rating'] = max(1, min(5, int(my_rating)))
        if bot_rating is not None:
            feedback['bot_play_rating'] = max(1, min(5, int(bot_rating)))
        if comment:
            feedback['comment'] = comment

        with open(filepath, 'r') as f:
            game_data = json.load(f)
        game_data['feedback'] = feedback
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)

        return jsonify({'success': True})

    @bp.route('/api/checkpoints', methods=['GET'])
    def list_checkpoints():
        checkpoints = []
        seen = set()
        # Scan both training and deploy directories
        dirs = [Path(checkpoint_dir), Path(checkpoint_path).parent]
        for cp_dir in dirs:
            if not cp_dir.exists():
                continue
            for cp in sorted(cp_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
                if cp.resolve() in seen:
                    continue
                seen.add(cp.resolve())
                try:
                    data = torch.load(cp, map_location='cpu', weights_only=False)
                    checkpoints.append({
                        'name': cp.name,
                        'path': str(cp),
                        'iteration': data.get('iteration', '?'),
                        'total_games': data.get('total_games', '?')
                    })
                except Exception:
                    checkpoints.append({
                        'name': cp.name, 'path': str(cp),
                        'iteration': '?', 'total_games': '?'
                    })
        return jsonify(checkpoints)

    @bp.route('/api/load_checkpoint', methods=['POST'])
    def load_checkpoint():
        nonlocal server
        data = request.json
        checkpoint_name = data.get('checkpoint')
        if not checkpoint_name:
            return jsonify({'error': 'No checkpoint specified'}), 400
        cp_path = Path(checkpoint_dir) / checkpoint_name
        if not cp_path.exists():
            # Also check deploy directory
            cp_path = Path(checkpoint_path).parent / checkpoint_name
        if not cp_path.exists():
            return jsonify({'error': f'Checkpoint not found: {checkpoint_name}'}), 404
        try:
            server = MandalaModelServer(cp_path, config_path, simulations)
            sessions.clear()
            return jsonify({
                'success': True,
                'checkpoint': checkpoint_name,
                'iteration': server.iteration,
                'total_games': server.total_games
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return bp, server


def find_latest_checkpoint(checkpoint_dir):
    """Find most recent checkpoint in directory."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    checkpoints = sorted(cp_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


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

    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint("data/checkpoints")
        if not checkpoint_path:
            print("No checkpoints found. Please train a model first.")
            return
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    app = Flask(__name__, template_folder=str(template_dir))
    bp, server = create_mandala_blueprint(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        simulations=args.simulations,
        base_url='',
        checkpoint_dir='data/checkpoints'
    )
    app.register_blueprint(bp)

    print(f"\n{'='*60}")
    print(f"MANDALA WEB PLAYER")
    print(f"{'='*60}")
    print(f"\nStarting web server on http://{args.host}:{args.port}")
    print(f"Model: Iteration {server.iteration}, {server.total_games} games")
    print(f"\nOpen your browser: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
