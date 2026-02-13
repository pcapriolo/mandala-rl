#!/usr/bin/env python3
"""
Web-based replay viewer for Lost Cities training games.

Browse and replay all training games organized by iteration.
"""
import argparse
import json
from pathlib import Path
from flask import Flask, render_template, jsonify
from collections import defaultdict

template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

REPLAY_DIR = Path("data/lost_cities/replays")
ELO_FILE = Path("data/lost_cities/elo_ratings.json")


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
            score0, score1 = final_score if final_score else (0, 0)
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
            continue

    return dict(replays_by_iteration)


@app.route('/')
def index():
    return render_template('replay_viewer_lc.html')


@app.route('/api/iterations')
def list_iterations():
    if not REPLAY_DIR.exists():
        return jsonify({'iterations': []})

    replays_by_iter = load_replay_index(REPLAY_DIR)

    elo_ratings = {}
    if ELO_FILE.exists():
        with open(ELO_FILE, 'r') as f:
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


@app.route('/api/games/<iteration>')
def list_games(iteration):
    replays_by_iter = load_replay_index(REPLAY_DIR)
    games = replays_by_iter.get(iteration, [])
    return jsonify({'games': games})


@app.route('/api/replay/<game_id>')
def get_replay(game_id):
    for replay_file in REPLAY_DIR.glob(f"game_{game_id}.json"):
        with open(replay_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({'error': 'Game not found'}), 404


def main():
    parser = argparse.ArgumentParser(description="Lost Cities Replay Viewer")
    parser.add_argument('--port', type=int, default=5004, help='Port (default: 5004)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"LOST CITIES REPLAY VIEWER")
    print(f"{'='*60}")
    print(f"\nStarting on http://{args.host}:{args.port}")
    print(f"Replay directory: {REPLAY_DIR}")
    print(f"Open your browser: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
