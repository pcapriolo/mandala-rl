#!/usr/bin/env python3
"""
Web-based Human vs AI gameplay for Lost Cities.

Can run standalone (python scripts/play_vs_ai_web_lc.py) or as a Blueprint
imported by the combined server (serve.py).
"""
import argparse
import json
import sys
import time
import uuid
import torch
import yaml
from pathlib import Path
from flask import Flask, Blueprint, render_template, jsonify, request
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lost_cities.game.engine import LostCitiesGame, COLOR_NAMES, NUM_ACTIONS
from lost_cities.game.state import LostCitiesState, LCCard, NUM_COLORS, HAND_SIZE
from mandala_rl.network.model import MandalaNet
from mandala_rl.mcts.search import MCTS

template_dir = Path(__file__).parent.parent / 'templates'


def action_to_display(action, state):
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

    dest_str = f"Play to {COLOR_NAMES[hand[hand_pos].color] if hand_pos < len(hand) else '?'} expedition" if dest == 0 else "Discard"
    draw_str = "Draw from deck" if draw_src == 0 else f"Draw from {COLOR_NAMES[draw_src - 1]} pile"

    return f"{card_str}: {dest_str}, {draw_str}"


class LCModelServer:
    """Shared model + MCTS for Lost Cities. Loaded once, used by all sessions."""

    def __init__(self, checkpoint_path, config_path, mcts_simulations=400):
        self.checkpoint_path = str(checkpoint_path)
        self.mcts_simulations = mcts_simulations
        self.engine = LostCitiesGame()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"[Lost Cities] Using device: {self.device}")

        print(f"[Lost Cities] Loading model from {checkpoint_path}...")
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
        print(f"[Lost Cities] Model loaded: iteration {self.iteration}, {self.total_games} games")

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
            c_puct=self.config['mcts']['c_puct'],
            time_limit=5.0  # Hard 5s cap for web serving
        )

    def get_ai_move(self, state):
        """Get AI move using network policy (with optional MCTS refinement)."""
        t0 = time.time()

        # Always compute raw network policy (fast, single forward pass)
        t_canon = time.time()
        canonical = state.get_canonical_form()
        t_tensor = time.time()
        state_tensor = torch.from_numpy(canonical.to_tensor()).unsqueeze(0).to(self.device)
        t_fwd = time.time()
        with torch.no_grad():
            raw_logits, val = self.model(state_tensor)
            value = val.item()
            raw_policy = torch.softmax(raw_logits, dim=1).cpu().numpy()[0]
        t_valid = time.time()

        valid_moves = self.engine.get_valid_moves(state)
        t_mcts = time.time()

        # Use MCTS if sims > 0 and time permits, otherwise use raw network policy
        if self.mcts_simulations > 0:
            policy, _ = self.mcts.get_action_prob(state, temperature=0.0, add_noise=False)
        else:
            policy = raw_policy
        t_done = time.time()

        print(f"[LC AI] Timing: canon={t_tensor-t_canon:.3f}s tensor={t_fwd-t_tensor:.3f}s "
              f"fwd={t_valid-t_fwd:.3f}s valid={t_mcts-t_valid:.3f}s "
              f"mcts={t_done-t_mcts:.3f}s total={t_done-t0:.3f}s (sims={self.mcts_simulations})")

        think_ms = int((time.time() - t0) * 1000)

        valid_policy = policy * valid_moves
        action = int(valid_policy.argmax())

        top_actions = valid_policy.argsort()[-5:][::-1]
        top_moves = [
            {
                'action': int(a),
                'description': action_to_display(a, state),
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
                'description': action_to_display(a, state),
                'probability': float(raw_policy[a] * 100)
            }
            for a in raw_top if valid_moves[a]
        ]

        return {
            'action': action,
            'description': action_to_display(action, state),
            'top_moves': top_moves,
            'network_top': network_top,
            'policy': [float(p) for p in policy],
            'value': float(value),
            'think_time_ms': think_ms,
        }


class LCGameSession:
    """Per-player Lost Cities game state."""

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
            {'action': int(i), 'description': action_to_display(i, self.state)}
            for i, valid in enumerate(valid_moves) if valid
        ]

        is_terminal = engine.is_terminal(self.state)
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
                'iteration': self.server.iteration,
                'total_games': self.server.total_games,
                'checkpoint': Path(self.server.checkpoint_path).name
            }
        }

    def _format_state(self):
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

    def make_move(self, action, think_time_ms=None, ai_data=None):
        engine = self.server.engine
        if self.state is None or engine.is_terminal(self.state):
            return {'error': 'No active game'}

        valid_moves = engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': f'Invalid move: {engine.action_to_string(action)}'}

        is_human = self.state.current_player == self.human_player

        move_record = {
            'move_num': self.move_count + 1,
            'player': int(self.state.current_player),
            'is_human': is_human,
            'action': int(action),
            'action_description': action_to_display(action, self.state),
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

        save_dir = Path("data/human_games/lost_cities")
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = f"lost_cities_{timestamp}"
        filename = f"{game_id}.json"
        filepath = save_dir / filename

        engine = self.server.engine
        is_terminal = engine.is_terminal(self.state)
        winner = None
        final_scores = None
        if is_terminal:
            s0 = self.state.compute_score(0)
            s1 = self.state.compute_score(1)
            final_scores = {'player0': s0, 'player1': s1}
            winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)

        game_data = {
            'game_id': game_id,
            'game': 'lost_cities',
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


def create_lc_blueprint(checkpoint_path, config_path, simulations=400,
                         base_url='', checkpoint_dir='data/lost_cities/checkpoints'):
    """Create Flask Blueprint for Lost Cities with session-based game management."""
    bp = Blueprint('lost_cities', __name__, template_folder=str(template_dir))
    server = LCModelServer(checkpoint_path, config_path, simulations)
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
        return render_template('play_vs_ai_lc.html', base_url=base_url)

    @bp.route('/api/info', methods=['GET'])
    def info():
        return jsonify({
            'game': 'lost_cities',
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
        sessions[game_id] = LCGameSession(server, human_player)
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
        t_req = time.time()
        data = request.json or {}
        game_id = data.get('game_id')
        session = get_session(game_id)
        if not session:
            return jsonify({'error': 'Game not found'}), 404

        t_ai = time.time()
        ai_decision = server.get_ai_move(session.state)
        t_after_ai = time.time()
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
        t_resp = time.time()
        print(f"[LC /api/ai_move] parse={t_ai-t_req:.3f}s ai={t_after_ai-t_ai:.3f}s "
              f"response={t_resp-t_after_ai:.3f}s total={t_resp-t_req:.3f}s")
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
            server = LCModelServer(cp_path, config_path, simulations)
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
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None
    checkpoints = sorted(cp_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description="Web-based Lost Cities vs AI")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/lost_cities.yaml', help='Config file')
    parser.add_argument('--simulations', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--port', type=int, default=5002, help='Port (default: 5002)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host')
    args = parser.parse_args()

    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint("data/lost_cities/checkpoints")
        if not checkpoint_path:
            print("No checkpoints found. Train a model first.")
            return
        print(f"Using latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    app = Flask(__name__, template_folder=str(template_dir))
    bp, server = create_lc_blueprint(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        simulations=args.simulations,
        base_url='',
        checkpoint_dir='data/lost_cities/checkpoints'
    )
    app.register_blueprint(bp)

    print(f"\n{'='*60}")
    print(f"LOST CITIES WEB PLAYER")
    print(f"{'='*60}")
    print(f"\nStarting web server on http://{args.host}:{args.port}")
    print(f"Model: Iteration {server.iteration}, {server.total_games} games")
    print(f"\nOpen your browser: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
