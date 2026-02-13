#!/usr/bin/env python3
"""
Web-based replay viewer for training games.

Browse and replay all training games organized by iteration.
"""
import argparse
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory
from collections import defaultdict

# Set up Flask
template_dir = Path(__file__).parent.parent / 'templates'
app = Flask(__name__, template_folder=str(template_dir))


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

            # Get scores and winner
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
            continue

    return dict(replays_by_iteration)


@app.route('/')
def index():
    """Show replay browser."""
    return render_template('replay_viewer.html')


@app.route('/api/iterations')
def list_iterations():
    """List all iterations with game counts."""
    replay_dir = Path("data/replays")
    if not replay_dir.exists():
        return jsonify({'iterations': []})

    replays_by_iter = load_replay_index(replay_dir)

    # Load Elo ratings
    elo_file = Path("data/elo_ratings.json")
    elo_ratings = {}
    if elo_file.exists():
        with open(elo_file, 'r') as f:
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
    """List all games for an iteration."""
    replay_dir = Path("data/replays")
    replays_by_iter = load_replay_index(replay_dir)

    games = replays_by_iter.get(iteration, [])
    return jsonify({'games': games})


@app.route('/api/replay/<game_id>')
def get_replay(game_id):
    """Get full replay data for a game."""
    replay_dir = Path("data/replays")

    # Find file by game_id
    for replay_file in replay_dir.glob(f"game_{game_id}.json"):
        with open(replay_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)

    return jsonify({'error': 'Game not found'}), 404


def main():
    parser = argparse.ArgumentParser(description="Replay Viewer")
    parser.add_argument('--port', type=int, default=5003, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"REPLAY VIEWER")
    print(f"{'='*80}")
    print(f"\n🎬 Starting replay viewer on http://{args.host}:{args.port}")
    print(f"📁 Replay directory: data/replays")
    print(f"\nOpen your browser and navigate to: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop the server\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
