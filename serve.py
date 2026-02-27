#!/usr/bin/env python3
"""
Combined production server for Mandala RL games.

Uses ONNX Runtime for inference (no PyTorch dependency).
Serves both Mandala and Lost Cities from a single Flask app.

Usage:
    # Local dev:
    python serve.py --port 5000

    # Production (Railway):
    gunicorn serve:app --bind 0.0.0.0:$PORT --timeout 30 --workers 1
"""

import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from flask import Flask, Blueprint, render_template, jsonify, request

def softmax(x):
    """Numpy softmax (replaces scipy.special.softmax)."""
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

template_dir = project_root / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# ──────────────────────────────────────────────────────────────
# Analytics -- lightweight request logging
# ──────────────────────────────────────────────────────────────

ANALYTICS_DIR = Path(os.environ.get('ANALYTICS_DIR', 'data/analytics'))
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
_analytics_lock = Lock()

def _hash_ip(ip):
    """Hash IP for privacy -- we only need uniqueness, not the raw IP."""
    return hashlib.sha256((ip or 'unknown').encode()).hexdigest()[:12]

def _log_request():
    """Log page view to daily JSONL file."""
    # Skip static assets, health checks, stats page, and non-game-start API calls
    path = request.path
    if path.startswith('/static') or path == '/health' or path == '/favicon.ico':
        return
    if path == '/stats' or path.startswith('/api/stats') or path.startswith('/api/game-stats'):
        return
    if '/api/' in path and '/api/new_game' not in path:
        return

    entry = {
        'ts': datetime.utcnow().isoformat() + 'Z',
        'path': path,
        'visitor': _hash_ip(request.remote_addr),
        'ref': request.referrer or '',
        'ua': request.headers.get('User-Agent', '')[:200],
    }

    _write_analytics_entry(entry)


def _log_event(event_name, extra=None):
    """Log a named event to the daily JSONL file."""
    entry = {
        'ts': datetime.utcnow().isoformat() + 'Z',
        'event': event_name,
        'visitor': _hash_ip(request.remote_addr),
    }
    if extra:
        entry.update(extra)
    _write_analytics_entry(entry)


def _write_analytics_entry(entry):
    """Write a single analytics entry to today's JSONL file."""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    log_file = ANALYTICS_DIR / f'{today}.jsonl'
    try:
        with _analytics_lock:
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    except Exception:
        pass  # never break the app over analytics

@app.before_request
def before_request_analytics():
    _log_request()

VALID_EVENTS = {'play_clicked', 'game_loaded', 'first_move', 'bounce_feedback'}

@app.route('/api/event', methods=['POST'])
def api_event():
    """Log a custom analytics event."""
    data = request.json or {}
    event_name = data.get('event', '')
    if event_name not in VALID_EVENTS:
        return jsonify({'error': f'Unknown event: {event_name}'}), 400
    extra = {}
    if event_name == 'bounce_feedback':
        reason = data.get('reason', '').strip()[:200]
        comment = data.get('comment', '').strip()[:500]
        if reason:
            extra['reason'] = reason
        if comment:
            extra['comment'] = comment
    _log_event(event_name, extra if extra else None)
    return jsonify({'ok': True})


@app.route('/api/stats')
def api_stats():
    """Return visitor stats + funnel events. Query params: ?days=7 (default)"""
    try:
        days = int(request.args.get('days', 7))
    except (ValueError, TypeError):
        days = 7
    days = max(1, min(90, days))
    funnel_keys = ['play_clicked', 'game_loaded', 'first_move']
    stats = {
        'period_days': days,
        'daily': [],
        'totals': {'page_views': 0, 'unique_visitors': 0, 'game_starts': 0},
        'funnel': {k: 0 for k in funnel_keys},
        'bounce_feedback': [],
    }
    all_visitors = set()

    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d')
        log_file = ANALYTICS_DIR / f'{date}.jsonl'
        day_views = 0
        day_visitors = set()
        day_game_starts = 0
        day_funnel = {k: 0 for k in funnel_keys}

        if log_file.exists():
            try:
                with open(log_file) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        event = entry.get('event')
                        if event:
                            if event in day_funnel:
                                day_funnel[event] += 1
                            elif event == 'bounce_feedback':
                                stats['bounce_feedback'].append({
                                    'date': date,
                                    'ts': entry.get('ts', ''),
                                    'reason': entry.get('reason', ''),
                                    'comment': entry.get('comment', ''),
                                })
                        else:
                            day_views += 1
                            day_visitors.add(entry.get('visitor', ''))
                            if '/api/new_game' in entry.get('path', ''):
                                day_game_starts += 1
            except Exception:
                pass

        stats['daily'].append({
            'date': date,
            'page_views': day_views,
            'unique_visitors': len(day_visitors),
            'game_starts': day_game_starts,
            'funnel': day_funnel,
        })
        stats['totals']['page_views'] += day_views
        all_visitors.update(day_visitors)
        stats['totals']['game_starts'] += day_game_starts
        for k in funnel_keys:
            stats['funnel'][k] += day_funnel[k]

    stats['totals']['unique_visitors'] = len(all_visitors)
    return jsonify(stats)

@app.route('/api/game-stats')
def api_game_stats():
    """Return stats computed from saved game JSON files."""
    games_root = Path('data/human_games')
    result = {}

    for game_type in ('mandala', 'lost_cities'):
        game_dir = games_root / game_type
        games = []
        if game_dir.exists():
            for fp in game_dir.glob('*.json'):
                try:
                    with open(fp) as f:
                        games.append(json.load(f))
                except Exception:
                    pass

        total = len(games)
        complete = [g for g in games if g.get('is_complete')]
        human_wins = sum(1 for g in complete if g.get('winner') == g.get('human_player'))
        ai_wins = sum(1 for g in complete if g.get('winner') not in (None, -1, g.get('human_player')))
        draws = sum(1 for g in complete if g.get('winner') == -1)

        visitors = set(g.get('visitor') for g in games if g.get('visitor'))
        unique_players = len(visitors)

        scores = []
        for g in complete:
            fs = g.get('final_scores', {})
            hp = g.get('human_player', 0)
            s = fs.get(f'player{hp}')
            if s is not None:
                scores.append(s)

        moves_list = [g.get('total_moves', 0) for g in complete]

        durations = []
        for g in complete:
            mvs = g.get('moves', [])
            if len(mvs) >= 2:
                try:
                    t0 = datetime.fromisoformat(mvs[0]['timestamp'])
                    t1 = datetime.fromisoformat(mvs[-1]['timestamp'])
                    durations.append((t1 - t0).total_seconds())
                except Exception:
                    pass

        recent = sorted(complete, key=lambda g: g.get('timestamp', ''), reverse=True)[:10]
        recent_list = []
        for g in recent:
            hp = g.get('human_player', 0)
            w = g.get('winner')
            if w == hp:
                res = 'win'
            elif w == -1:
                res = 'draw'
            else:
                res = 'loss'
            fs = g.get('final_scores', {})
            dur = None
            mvs = g.get('moves', [])
            if len(mvs) >= 2:
                try:
                    t0 = datetime.fromisoformat(mvs[0]['timestamp'])
                    t1 = datetime.fromisoformat(mvs[-1]['timestamp'])
                    dur = round((t1 - t0).total_seconds())
                except Exception:
                    pass
            recent_list.append({
                'date': g.get('timestamp', ''),
                'game': game_type,
                'result': res,
                'score': fs.get(f'player{hp}'),
                'moves': g.get('total_moves', 0),
                'duration_s': dur,
            })

        feedback_list = []
        for g in games:
            fb = g.get('feedback')
            if fb:
                feedback_list.append({
                    'game': game_type,
                    'game_id': g.get('game_id', ''),
                    'date': fb.get('submitted_at', ''),
                    'my_play_rating': fb.get('my_play_rating'),
                    'bot_play_rating': fb.get('bot_play_rating'),
                    'comment': fb.get('comment', ''),
                })

        result[game_type] = {
            'total_games': total,
            'complete_games': len(complete),
            'human_wins': human_wins,
            'ai_wins': ai_wins,
            'draws': draws,
            'win_rate': round(human_wins / len(complete) * 100, 1) if complete else 0,
            'avg_score': round(sum(scores) / len(scores), 1) if scores else None,
            'avg_moves': round(sum(moves_list) / len(moves_list), 1) if moves_list else None,
            'avg_duration_s': round(sum(durations) / len(durations)) if durations else None,
            'unique_players': unique_players,
            'games_per_player': round(total / unique_players, 1) if unique_players else None,
            'recent': recent_list,
            'feedback': feedback_list,
        }

    return jsonify(result)

# Configuration
DEPLOY_DIR = Path(os.environ.get('DEPLOY_DIR', 'data/deploy'))
loaded_games = {}

# ──────────────────────────────────────────────────────────────
# ONNX Model Server — lightweight inference, no PyTorch needed
# ──────────────────────────────────────────────────────────────

class OnnxModelServer:
    """Game-agnostic ONNX inference server."""

    def __init__(self, onnx_path, meta_path, engine):
        self.engine = engine
        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(str(onnx_path))

        with open(meta_path) as f:
            meta = json.load(f)
        self.iteration = meta.get('iteration', 'unknown')
        self.total_games = meta.get('total_games', 'unknown')
        self.checkpoint_path = str(onnx_path)

        # Warmup
        input_shape = self.session.get_inputs()[0].shape
        channels = input_shape[1] if isinstance(input_shape[1], int) else meta.get('input_channels', 86)
        dummy = np.random.randn(1, channels, 8, 8).astype(np.float32)
        t0 = time.time()
        self.session.run(None, {'state': dummy})
        print(f"[serve] ONNX warmup: {(time.time()-t0)*1000:.1f}ms ({onnx_path.name})")

    def predict(self, state_tensor):
        """Run ONNX inference on a single state tensor. Returns (policy, value)."""
        inp = state_tensor[np.newaxis].astype(np.float32) if state_tensor.ndim == 3 else state_tensor.astype(np.float32)
        outputs = self.session.run(None, {'state': inp})
        policy_logits, value = outputs[0], outputs[1]
        policy = softmax(policy_logits[0])
        return policy, float(value[0, 0])


# ──────────────────────────────────────────────────────────────
# Lost Cities
# ──────────────────────────────────────────────────────────────

from lost_cities.game.engine import LostCitiesGame, COLOR_NAMES, NUM_ACTIONS
from lost_cities.game.state import NUM_COLORS

def _lc_action_to_display(action, state):
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


class LCGameSession:
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
        engine = self.server.engine
        valid_moves = engine.get_valid_moves(self.state)
        valid_actions = [
            {'action': int(i), 'description': _lc_action_to_display(i, self.state)}
            for i, valid in enumerate(valid_moves) if valid
        ]
        is_terminal = engine.is_terminal(self.state)
        winner, scores = None, None
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
                'checkpoint': Path(self.server.checkpoint_path).name,
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
            'hands': {f'player{p}': cards_to_list(self.state.hands[p]) for p in range(2)},
            'expeditions': {
                f'player{p}': {COLOR_NAMES[c].lower(): exp_summary(self.state.expeditions[p][c]) for c in range(NUM_COLORS)}
                for p in range(2)
            },
            'discard_piles': {COLOR_NAMES[c].lower(): cards_to_list(self.state.discard_piles[c]) for c in range(NUM_COLORS)},
            'deck_size': len(self.state.deck),
            'scores': {f'player{p}': self.state.compute_score(p) for p in range(2)},
            'turns_played': self.state.turns_played,
        }

    def make_move(self, action, think_time_ms=None, ai_data=None):
        engine = self.server.engine
        if engine.is_terminal(self.state):
            return {'error': 'No active game'}
        valid_moves = engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': f'Invalid move: {action}'}
        move_record = {
            'move_num': self.move_count + 1,
            'player': int(self.state.current_player),
            'is_human': self.state.current_player == self.human_player,
            'action': int(action),
            'action_description': _lc_action_to_display(action, self.state),
            'timestamp': datetime.now().isoformat(),
            'think_time_ms': think_time_ms,
        }
        if ai_data:
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
        filepath = save_dir / f"{game_id}.json"
        is_terminal = self.server.engine.is_terminal(self.state)
        winner, final_scores = None, None
        if is_terminal:
            s0 = self.state.compute_score(0)
            s1 = self.state.compute_score(1)
            final_scores = {'player0': s0, 'player1': s1}
            winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
        game_data = {
            'game_id': game_id, 'game': 'lost_cities',
            'timestamp': self.game_start_time,
            'model_iteration': self.server.iteration,
            'human_player': self.human_player,
            'visitor': _hash_ip(request.remote_addr),
            'winner': winner, 'final_scores': final_scores,
            'total_moves': len(self.game_history),
            'is_complete': is_terminal,
            'moves': self.game_history,
        }
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        self.last_save_filepath = str(filepath)
        return {'success': True, 'filename': filepath.name, 'moves': len(self.game_history)}


def create_lc_blueprint(server):
    bp = Blueprint('lost_cities', __name__, template_folder=str(template_dir))
    sessions = {}

    def get_session(game_id):
        s = sessions.get(game_id)
        if s:
            s.last_activity = time.time()
        return s

    def cleanup():
        now = time.time()
        for gid in [g for g, s in sessions.items() if now - s.last_activity > 3600]:
            del sessions[gid]

    @bp.route('/')
    def index():
        return render_template('play_vs_ai_lc.html', base_url='/lost-cities')

    @bp.route('/api/info')
    def info():
        return jsonify({
            'game': 'lost_cities', 'iteration': server.iteration,
            'total_games': server.total_games,
            'checkpoint': Path(server.checkpoint_path).name,
            'active_sessions': len(sessions),
        })

    @bp.route('/api/new_game', methods=['POST'])
    def new_game():
        cleanup()
        data = request.json or {}
        game_id = str(uuid.uuid4())
        sessions[game_id] = LCGameSession(server, data.get('human_player', 0))
        state = sessions[game_id].get_game_state_dict()
        state['game_id'] = game_id
        return jsonify(state)

    @bp.route('/api/state')
    def get_state():
        session = get_session(request.args.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        state = session.get_game_state_dict()
        state['game_id'] = request.args.get('game_id')
        return jsonify(state)

    @bp.route('/api/move', methods=['POST'])
    def make_move():
        data = request.json
        session = get_session(data.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        try:
            result = session.make_move(data.get('action'), think_time_ms=data.get('think_time_ms'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Move failed: {e}'}), 500
        result['game_id'] = data.get('game_id')
        return jsonify(result)

    @bp.route('/api/ai_move', methods=['POST'])
    def ai_move():
        try:
            t_req = time.time()
            data = request.json or {}
            session = get_session(data.get('game_id'))
            if not session:
                return jsonify({'error': 'Game not found'}), 404

            state = session.state
            canonical = state.get_canonical_form()
            tensor = canonical.to_tensor()
            t_infer = time.time()
            policy, value = server.predict(tensor)
            t_done = time.time()

            valid_moves = server.engine.get_valid_moves(state)
            valid_policy = policy * valid_moves
            action = int(valid_policy.argmax())

            top_actions = valid_policy.argsort()[-5:][::-1]
            top_moves = [
                {'action': int(a), 'description': _lc_action_to_display(a, state),
                 'probability': float(policy[a] * 100)}
                for a in top_actions if valid_moves[a]
            ]

            think_ms = int((t_done - t_req) * 1000)
            print(f"[LC AI] prep={t_infer-t_req:.3f}s infer={t_done-t_infer:.3f}s total={t_done-t_req:.3f}s")

            ai_data = {'value': value, 'top_moves': top_moves, 'think_time_ms': think_ms}
            result = session.make_move(action, ai_data=ai_data)
            result['ai_decision'] = {'action': action, 'description': _lc_action_to_display(action, state), 'top_moves': top_moves}
            result['game_id'] = data.get('game_id')
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'AI move failed: {e}'}), 500

    @bp.route('/api/save', methods=['POST'])
    def save_game():
        session = get_session((request.json or {}).get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        return jsonify(session.save_game())

    @bp.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        data = request.json or {}
        session = get_session(data.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        if not session.last_save_filepath:
            return jsonify({'error': 'Game not saved yet'}), 400
        filepath = Path(session.last_save_filepath)
        if not filepath.exists():
            return jsonify({'error': 'Save file not found'}), 404
        feedback = {'submitted_at': datetime.now().isoformat()}
        for key in ['my_play_rating', 'bot_play_rating']:
            val = data.get(key)
            if val is not None:
                feedback[key] = max(1, min(5, int(val)))
        comment = data.get('comment', '').strip()
        if comment:
            feedback['comment'] = comment
        with open(filepath, 'r') as f:
            game_data = json.load(f)
        game_data['feedback'] = feedback
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        return jsonify({'success': True})

    @bp.route('/api/checkpoints')
    def list_checkpoints():
        return jsonify([{
            'name': Path(server.checkpoint_path).name,
            'path': server.checkpoint_path,
            'iteration': server.iteration,
            'total_games': server.total_games,
        }])

    return bp


# ──────────────────────────────────────────────────────────────
# Mandala
# ──────────────────────────────────────────────────────────────

from mandala_rl.game.engine import MandalaGame

COLOR_SHORT = ['R', 'G', 'P', 'O', 'Y', 'W']
MANDALA_COLOR_NAMES = ['Red', 'Green', 'Purple', 'Orange', 'Yellow', 'White']

def _mandala_action_to_string(action):
    if action < 12:
        return f"{COLOR_SHORT[action % 6]} → Mt{action // 6 + 1}"
    elif action < 96:
        a = action - 12
        mandala = a // 42
        r = a % 42
        color = r // 7
        count = r % 7 + 1
        return f"{count}x {COLOR_SHORT[color]} → Fd{mandala + 1}"
    elif action < 144:
        a = action - 96
        color = a // 8
        count = a % 8 + 1
        return f"Discard {count}x {COLOR_SHORT[color]}"
    else:
        return f"Claim {MANDALA_COLOR_NAMES[action - 144]}"


class MandalaGameSession:
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
        engine = self.server.engine
        valid_moves = engine.get_valid_moves(self.state)
        valid_actions = [(i, _mandala_action_to_string(i)) for i, valid in enumerate(valid_moves) if valid]
        is_terminal = engine.is_terminal(self.state)
        winner, scores = None, None
        if is_terminal:
            s0 = engine._calculate_score(self.state, 0)
            s1 = engine._calculate_score(self.state, 1)
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
                'checkpoint': Path(self.server.checkpoint_path).name,
            }
        }

    def _format_state(self):
        s = self.state
        def cards_to_colors(card_list):
            return [card.color for card in card_list]

        is_terminal = self.server.engine.is_terminal(s)

        # At game end, reveal all cup cards
        if is_terminal:
            cups = {
                'player0': {'hidden': 0, 'visible': cards_to_colors(s.cups[0])},
                'player1': {'hidden': 0, 'visible': cards_to_colors(s.cups[1])},
            }
        else:
            cups = {}
            for p in range(2):
                if p == self.human_player:
                    cups[f'player{p}'] = {'hidden': 0, 'visible': cards_to_colors(s.cups[p])}
                else:
                    cups[f'player{p}'] = {'hidden': 2, 'visible': cards_to_colors(s.cups[p][2:])}

        return {
            'hands': {
                'player0': cards_to_colors(s.hands[0]),
                'player1': cards_to_colors(s.hands[1]),
            },
            'mandalas': [
                {
                    'mountain': cards_to_colors(s.mountains[i]),
                    'field_p0': cards_to_colors(s.fields[i][0]),
                    'field_p1': cards_to_colors(s.fields[i][1]),
                    'colors': len(s.get_colors_in_mandala(i)),
                }
                for i in range(2)
            ],
            'rivers': {
                'player0': cards_to_colors(s.rivers[0]),
                'player1': cards_to_colors(s.rivers[1]),
            },
            'cups': cups,
            'deck_size': len(s.deck),
            'discard_size': len(s.discard),
            'deck_reshuffled': s.deck_reshuffled,
            'game_ends_next_mandala': s.game_ends_next_mandala,
            'phase': s.phase,
            'claiming_mandala': s.claiming_mandala,
        }

    def make_move(self, action, think_time_ms=None, ai_data=None):
        engine = self.server.engine
        if engine.is_terminal(self.state):
            return {'error': 'No active game'}
        valid_moves = engine.get_valid_moves(self.state)
        if not valid_moves[action]:
            return {'error': f'Invalid move: {action}'}
        move_record = {
            'move_num': self.move_count + 1,
            'player': int(self.state.current_player),
            'is_human': self.state.current_player == self.human_player,
            'action': int(action),
            'action_description': _mandala_action_to_string(action),
            'timestamp': datetime.now().isoformat(),
            'think_time_ms': think_time_ms,
        }
        if ai_data:
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
        filepath = save_dir / f"{game_id}.json"
        engine = self.server.engine
        is_terminal = engine.is_terminal(self.state)
        winner, final_scores = None, None
        if is_terminal:
            s0 = engine._calculate_score(self.state, 0)
            s1 = engine._calculate_score(self.state, 1)
            final_scores = {'player0': s0, 'player1': s1}
            winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
        game_data = {
            'game_id': game_id, 'game': 'mandala',
            'timestamp': self.game_start_time,
            'model_iteration': self.server.iteration,
            'human_player': self.human_player,
            'visitor': _hash_ip(request.remote_addr),
            'winner': winner, 'final_scores': final_scores,
            'total_moves': len(self.game_history),
            'is_complete': is_terminal,
            'moves': self.game_history,
        }
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        self.last_save_filepath = str(filepath)
        return {'success': True, 'filename': filepath.name, 'moves': len(self.game_history)}


def create_mandala_blueprint(server):
    bp = Blueprint('mandala', __name__, template_folder=str(template_dir))
    sessions = {}

    def get_session(game_id):
        s = sessions.get(game_id)
        if s:
            s.last_activity = time.time()
        return s

    def cleanup():
        now = time.time()
        for gid in [g for g, s in sessions.items() if now - s.last_activity > 3600]:
            del sessions[gid]

    @bp.route('/')
    def index():
        return render_template('play_vs_ai.html', base_url='/mandala')

    @bp.route('/api/info')
    def info():
        return jsonify({
            'game': 'mandala', 'iteration': server.iteration,
            'total_games': server.total_games,
            'checkpoint': Path(server.checkpoint_path).name,
            'active_sessions': len(sessions),
        })

    @bp.route('/api/new_game', methods=['POST'])
    def new_game():
        cleanup()
        data = request.json or {}
        game_id = str(uuid.uuid4())
        sessions[game_id] = MandalaGameSession(server, data.get('human_player', 0))
        state = sessions[game_id].get_game_state_dict()
        state['game_id'] = game_id
        return jsonify(state)

    @bp.route('/api/state')
    def get_state():
        session = get_session(request.args.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        state = session.get_game_state_dict()
        state['game_id'] = request.args.get('game_id')
        return jsonify(state)

    @bp.route('/api/move', methods=['POST'])
    def make_move():
        data = request.json
        session = get_session(data.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        try:
            result = session.make_move(data.get('action'), think_time_ms=data.get('think_time_ms'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Move failed: {e}'}), 500
        result['game_id'] = data.get('game_id')
        return jsonify(result)

    @bp.route('/api/ai_move', methods=['POST'])
    def ai_move():
        try:
            t_req = time.time()
            data = request.json or {}
            session = get_session(data.get('game_id'))
            if not session:
                return jsonify({'error': 'Game not found'}), 404

            state = session.state
            canonical = state.get_canonical_form()
            tensor = canonical.to_tensor()
            t_infer = time.time()
            policy, value = server.predict(tensor)
            t_done = time.time()

            valid_moves = server.engine.get_valid_moves(state)
            valid_policy = policy * valid_moves
            action = int(valid_policy.argmax())

            top_actions = valid_policy.argsort()[-5:][::-1]
            top_moves = [
                {'action': int(a), 'description': _mandala_action_to_string(a),
                 'probability': float(policy[a] * 100)}
                for a in top_actions if valid_moves[a]
            ]

            think_ms = int((t_done - t_req) * 1000)
            print(f"[Mandala AI] prep={t_infer-t_req:.3f}s infer={t_done-t_infer:.3f}s total={t_done-t_req:.3f}s")

            ai_data = {'value': value, 'top_moves': top_moves, 'think_time_ms': think_ms}
            result = session.make_move(action, ai_data=ai_data)
            result['ai_decision'] = {
                'action': action,
                'description': _mandala_action_to_string(action),
                'top_moves': top_moves,
                'network_top': top_moves,  # serve.py uses raw network (no MCTS)
                'value': value,
            }
            result['game_id'] = data.get('game_id')
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'AI move failed: {e}'}), 500

    @bp.route('/api/save', methods=['POST'])
    def save_game():
        session = get_session((request.json or {}).get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        return jsonify(session.save_game())

    @bp.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        data = request.json or {}
        session = get_session(data.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404
        if not session.last_save_filepath:
            return jsonify({'error': 'Game not saved yet'}), 400
        filepath = Path(session.last_save_filepath)
        if not filepath.exists():
            return jsonify({'error': 'Save file not found'}), 404
        feedback = {'submitted_at': datetime.now().isoformat()}
        for key in ['my_play_rating', 'bot_play_rating']:
            val = data.get(key)
            if val is not None:
                feedback[key] = max(1, min(5, int(val)))
        comment = data.get('comment', '').strip()
        if comment:
            feedback['comment'] = comment
        with open(filepath, 'r') as f:
            game_data = json.load(f)
        game_data['feedback'] = feedback
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
        return jsonify({'success': True})

    @bp.route('/api/ai_hint', methods=['POST'])
    def ai_hint():
        """Get AI's recommended move for the current position."""
        data = request.json or {}
        session = get_session(data.get('game_id'))
        if not session:
            return jsonify({'error': 'Game not found'}), 404

        state = session.state
        canonical = state.get_canonical_form()
        tensor = canonical.to_tensor()
        policy, value = server.predict(tensor)

        valid_moves = server.engine.get_valid_moves(state)
        valid_policy = policy * valid_moves
        action = int(valid_policy.argmax())

        top_actions = valid_policy.argsort()[-3:][::-1]
        top_moves = [
            {'action': int(a), 'description': _mandala_action_to_string(a),
             'probability': float(policy[a] * 100)}
            for a in top_actions if valid_moves[a]
        ]

        return jsonify({
            'recommended_action': action,
            'recommended_description': _mandala_action_to_string(action),
            'value': float(value),
            'top_moves': top_moves,
        })

    @bp.route('/api/checkpoints')
    def list_checkpoints():
        return jsonify([{
            'name': Path(server.checkpoint_path).name,
            'path': server.checkpoint_path,
            'iteration': server.iteration,
            'total_games': server.total_games,
        }])

    return bp


# ──────────────────────────────────────────────────────────────
# Load games
# ──────────────────────────────────────────────────────────────

def _load_onnx_game(name, deploy_subdir, engine):
    onnx_path = DEPLOY_DIR / deploy_subdir / 'model.onnx'
    meta_path = DEPLOY_DIR / deploy_subdir / 'model.json'
    if not onnx_path.exists():
        print(f"[serve] No ONNX model for {name} at {onnx_path}")
        return None
    if not meta_path.exists():
        print(f"[serve] No metadata for {name} at {meta_path}")
        return None
    try:
        server = OnnxModelServer(onnx_path, meta_path, engine)
        print(f"[serve] {name} loaded: iter {server.iteration}")
        return server
    except Exception as e:
        print(f"[serve] Failed to load {name}: {e}")
        return None


# Load Lost Cities
_lc_server = _load_onnx_game('Lost Cities', 'lost_cities', LostCitiesGame())
if _lc_server:
    app.register_blueprint(create_lc_blueprint(_lc_server), url_prefix='/lost-cities')
    loaded_games['lost_cities'] = {
        'iteration': _lc_server.iteration,
        'total_games': _lc_server.total_games,
        'checkpoint': Path(_lc_server.checkpoint_path).name,
    }

# Load Mandala
_mandala_server = _load_onnx_game('Mandala', 'mandala', MandalaGame())
if _mandala_server:
    app.register_blueprint(create_mandala_blueprint(_mandala_server), url_prefix='/mandala')
    loaded_games['mandala'] = {
        'iteration': _mandala_server.iteration,
        'total_games': _mandala_server.total_games,
        'checkpoint': Path(_mandala_server.checkpoint_path).name,
    }


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.route('/stats')
def stats_page():
    return render_template('stats.html')

@app.route('/')
def landing():
    return render_template('index.html', games=loaded_games)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'games': list(loaded_games.keys()),
        'inference': 'onnx',
    })


@app.route('/debug/bench')
def debug_bench():
    results = {}
    for name, server in [('mandala', _mandala_server), ('lost_cities', _lc_server)]:
        if server is None:
            continue
        try:
            inp_shape = server.session.get_inputs()[0].shape
            channels = inp_shape[1] if isinstance(inp_shape[1], int) else 86
            dummy = np.random.randn(1, channels, 8, 8).astype(np.float32)
            server.session.run(None, {'state': dummy})  # warmup
            times = []
            for _ in range(10):
                t0 = time.time()
                server.session.run(None, {'state': dummy})
                times.append(time.time() - t0)
            results[name] = {
                'times_ms': [round(t * 1000, 2) for t in times],
                'mean_ms': round(np.mean(times) * 1000, 2),
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    return jsonify(results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Combined game server (ONNX)")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '5000')))
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()

    if not loaded_games:
        print("\nNo games loaded! Export ONNX models first:")
        print("  python scripts/export_onnx.py <checkpoint> --config <config> --output data/deploy/<game>/model.onnx")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"MANDALA RL GAME SERVER (ONNX)")
    print(f"{'='*60}")
    print(f"Games: {', '.join(loaded_games.keys())}")
    print(f"\nhttp://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)
