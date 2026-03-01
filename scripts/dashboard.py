#!/usr/bin/env python3
"""
Unified training dashboard for all games.

Pick a game, browse iterations, drill into games, replay move-by-move.

Usage:
    python3 scripts/dashboard.py
    python3 scripts/dashboard.py --port 5001
"""
import argparse
import json
from pathlib import Path
from flask import Flask, render_template, jsonify
from collections import defaultdict

template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

GAMES = {
    'mandala': {
        'name': 'Mandala',
        'replay_dir': Path('data/replays'),
        'elo_file': Path('data/elo_ratings.json'),
        'template': 'replay_viewer.html',
    },
    'lost-cities': {
        'name': 'Lost Cities',
        'replay_dir': Path('data/lost_cities/replays'),
        'elo_file': Path('data/lost_cities/elo_ratings.json'),
        'template': 'replay_viewer_lc.html',
    },
    'dominion': {
        'name': 'Dominion',
        'replay_dir': Path('data/dominion/replays'),
        'elo_file': Path('data/dominion/elo_ratings.json'),
        'losses_file': Path('data/dominion/losses.jsonl'),
        'template': 'dashboard_dominion.html',
    },
}


def load_replay_index(replay_dir: Path):
    """Index all replay files by iteration."""
    replays_by_iteration = defaultdict(list)

    for replay_file in sorted(replay_dir.glob("game_*.json")):
        try:
            with open(replay_file, 'r') as f:
                data = json.load(f)

            iteration = data.get('metadata', {}).get('iteration', 'unknown')
            game_id = data.get('game_id', replay_file.stem)
            move_count = len(data.get('moves', []))

            final_score = data.get('final_score')
            if final_score:
                score0, score1 = final_score
            else:
                score0, score1 = 0, 0
            winner = data.get('winner')

            replays_by_iteration[str(iteration)].append({
                'game_id': game_id,
                'filename': replay_file.name,
                'move_count': move_count,
                'score0': score0,
                'score1': score1,
                'winner': winner
            })
        except Exception as e:
            print(f"Error loading {replay_file}: {e}")

    return dict(replays_by_iteration)


@app.route('/')
def index():
    """Landing page: pick a game."""
    game_info = []
    for key, cfg in GAMES.items():
        replay_count = len(list(cfg['replay_dir'].glob("game_*.json"))) if cfg['replay_dir'].exists() else 0
        # For games with losses.jsonl, show iteration count as subtitle
        losses_file = cfg.get('losses_file')
        if losses_file and losses_file.exists():
            iter_count = sum(1 for line in losses_file.read_text().strip().split('\n') if line.strip())
            subtitle = f'{iter_count} iterations'
        else:
            subtitle = f'{replay_count} replays'
        game_info.append({'key': key, 'name': cfg['name'], 'replay_count': replay_count, 'subtitle': subtitle})
    return render_template('dashboard.html', games=game_info)


@app.route('/<game>/')
def game_view(game):
    """Game-specific replay viewer."""
    if game not in GAMES:
        return "Game not found", 404
    return render_template(GAMES[game]['template'], api_base=f'/{game}')


@app.route('/<game>/api/iterations')
def list_iterations(game):
    """List all iterations with game counts."""
    if game not in GAMES:
        return jsonify({'error': 'Game not found'}), 404

    cfg = GAMES[game]
    if not cfg['replay_dir'].exists():
        return jsonify({'iterations': []})

    replays_by_iter = load_replay_index(cfg['replay_dir'])

    elo_ratings = {}
    if cfg['elo_file'].exists():
        with open(cfg['elo_file'], 'r') as f:
            elo_data = json.load(f)
            elo_ratings = elo_data.get('ratings', {})

    iterations = []
    for iter_name in sorted(replays_by_iter.keys(), key=lambda x: int(x) if x.isdigit() else -1):
        games = replays_by_iter[iter_name]
        elo_key = f'iter_{iter_name}'
        elo_rating = elo_ratings.get(elo_key)

        iter_data = {
            'iteration': iter_name,
            'game_count': len(games),
            'total_moves': sum(g['move_count'] for g in games)
        }
        if elo_rating is not None:
            iter_data['elo'] = round(elo_rating, 1)
        iterations.append(iter_data)

    return jsonify({'iterations': iterations})


@app.route('/<game>/api/games/<iteration>')
def list_games(game, iteration):
    """List all games for an iteration."""
    if game not in GAMES:
        return jsonify({'error': 'Game not found'}), 404

    replays_by_iter = load_replay_index(GAMES[game]['replay_dir'])
    games = replays_by_iter.get(iteration, [])
    return jsonify({'games': games})


@app.route('/<game>/api/training')
def training_data(game):
    """Return per-iteration training metrics from losses.jsonl."""
    if game not in GAMES:
        return jsonify({'error': 'Game not found'}), 404

    losses_file = GAMES[game].get('losses_file')
    if not losses_file or not losses_file.exists():
        return jsonify({'iterations': []})

    iterations = []
    for line in losses_file.read_text().strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            iterations.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return jsonify({'iterations': iterations})


@app.route('/<game>/api/replay/<game_id>')
def get_replay(game, game_id):
    """Get full replay data for a game."""
    if game not in GAMES:
        return jsonify({'error': 'Game not found'}), 404

    replay_dir = GAMES[game]['replay_dir']
    for replay_file in replay_dir.glob(f"game_{game_id}.json"):
        with open(replay_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)

    return jsonify({'error': 'Game not found'}), 404


def main():
    parser = argparse.ArgumentParser(description="Training Dashboard")
    parser.add_argument('--port', type=int, default=5001, help='Port (default: 5001)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"TRAINING DASHBOARD")
    print(f"{'='*60}")
    for key, cfg in GAMES.items():
        print(f"  {cfg['name']}: /{key}/")
    print(f"\nOpen: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
