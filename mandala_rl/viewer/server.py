"""Flask web server for training observation."""
from flask import Flask, render_template, jsonify, send_from_directory, request
from pathlib import Path
import json
from collections import defaultdict
from typing import List, Dict
import yaml


class TrainingObserver:
    """Web server for observing training progress."""

    def __init__(self, data_dir: Path, config_path: Path = None, game_name: str = "Mandala"):
        self.data_dir = Path(data_dir)
        self.replays_dir = self.data_dir / 'replays'
        self.checkpoints_dir = self.data_dir / 'checkpoints'
        self.logs_dir = self.data_dir / 'logs'
        self.config_path = config_path
        self.game_name = game_name

        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))

        self._setup_routes()

    def _replays_dir_for(self, game: str) -> Path:
        if game == 'mandala':
            return self.replays_dir
        elif game == 'lost_cities':
            return self.data_dir / 'lost_cities' / 'replays'
        return None

    def _elo_file_for(self, game: str) -> Path:
        if game == 'mandala':
            return self.data_dir / 'elo_ratings.json'
        elif game == 'lost_cities':
            return self.data_dir / 'lost_cities' / 'elo_ratings.json'
        return None

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main dashboard."""
            return render_template('index.html', game_name=self.game_name)

        @self.app.route('/game/<game>')
        def game_detail(game):
            """Game detail page showing iterations with Elo and replays."""
            names = {'mandala': 'Mandala', 'lost_cities': 'Lost Cities'}
            return render_template('game_detail.html', game=game, game_name=names.get(game, game))

        @self.app.route('/api/status')
        def api_status():
            """Get training status."""
            mandala_replays = list(self.replays_dir.glob('*.json')) if self.replays_dir.exists() else []
            lc_replays_dir = self.data_dir / 'lost_cities' / 'replays'
            lc_replays = list(lc_replays_dir.glob('*.json')) if lc_replays_dir.exists() else []

            status = {
                'mandala_replays': len(mandala_replays),
                'lc_replays': len(lc_replays),
            }

            return jsonify(status)

        @self.app.route('/api/replays/<game>')
        def api_replays(game):
            """Get list of available replays for a game."""
            replays_dir = self._replays_dir_for(game)
            if not replays_dir or not replays_dir.exists():
                return jsonify([])

            limit = int(request.args.get('limit', 20))
            iteration = request.args.get('iteration')

            replays = []
            for replay_file in sorted(replays_dir.glob('*.json'), reverse=True):
                try:
                    with open(replay_file, 'r') as f:
                        data = json.load(f)
                    iter_num = data.get('metadata', {}).get('iteration')
                    if iteration is not None and str(iter_num) != iteration:
                        continue
                    replays.append({
                        'filename': replay_file.name,
                        'game': game,
                        'game_id': data['game_id'],
                        'iteration': iter_num,
                        'num_moves': len(data['moves']),
                        'final_score': data.get('final_score'),
                        'winner': data.get('winner')
                    })
                    if len(replays) >= limit:
                        break
                except:
                    pass

            return jsonify(replays)

        @self.app.route('/api/iterations/<game>')
        def api_iterations(game):
            """Get iterations with Elo ratings and replay counts."""
            # Get Elo ratings
            elo_file = self._elo_file_for(game)
            elo_map = {}
            if elo_file and elo_file.exists():
                with open(elo_file) as f:
                    data = json.load(f)
                for k, v in data.get('ratings', {}).items():
                    elo_map[int(k.split('_')[1])] = v

            # Count replays per iteration
            replays_dir = self._replays_dir_for(game)
            replay_counts = defaultdict(int)
            if replays_dir and replays_dir.exists():
                for replay_file in replays_dir.glob('*.json'):
                    try:
                        with open(replay_file) as f:
                            data = json.load(f)
                        it = data.get('metadata', {}).get('iteration')
                        if it is not None:
                            replay_counts[it] += 1
                    except:
                        pass

            # Merge into iteration list
            all_iters = set(elo_map.keys()) | set(replay_counts.keys())
            iterations = sorted([{
                'iteration': i,
                'elo': elo_map.get(i),
                'num_replays': replay_counts.get(i, 0),
            } for i in all_iters], key=lambda x: x['iteration'], reverse=True)

            return jsonify(iterations)

        @self.app.route('/api/replay/<game>/<filename>')
        def api_replay(game, filename):
            """Get specific replay data."""
            replays_dir = self._replays_dir_for(game)
            if not replays_dir:
                return jsonify({'error': 'Unknown game'}), 404

            replay_file = replays_dir / filename
            if not replay_file.exists():
                return jsonify({'error': 'Replay not found'}), 404

            with open(replay_file, 'r') as f:
                data = json.load(f)

            return jsonify(data)

        @self.app.route('/api/elo')
        def api_elo():
            """Get Elo ratings for all games."""
            games = {}
            # Mandala: data/elo_ratings.json
            mandala_elo = self.data_dir / 'elo_ratings.json'
            if mandala_elo.exists():
                with open(mandala_elo) as f:
                    data = json.load(f)
                ratings = data.get('ratings', {})
                games['Mandala'] = sorted(
                    [{'iter': int(k.split('_')[1]), 'elo': v} for k, v in ratings.items()],
                    key=lambda x: x['iter']
                )
            # Lost Cities: data/lost_cities/elo_ratings.json
            lc_elo = self.data_dir / 'lost_cities' / 'elo_ratings.json'
            if lc_elo.exists():
                with open(lc_elo) as f:
                    data = json.load(f)
                ratings = data.get('ratings', {})
                games['Lost Cities'] = sorted(
                    [{'iter': int(k.split('_')[1]), 'elo': v} for k, v in ratings.items()],
                    key=lambda x: x['iter']
                )
            return jsonify(games)

        @self.app.route('/replay/<game>/<filename>')
        def view_replay(game, filename):
            """View replay in browser."""
            return render_template('replay.html', game=game, filename=filename)

    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Start the web server."""
        print(f"\n{'='*60}")
        print(f"🌐 {self.game_name} Training Observer")
        print(f"{'='*60}")
        print(f"Dashboard: http://{host}:{port}")
        print(f"Tensorboard: http://{host}:6006 (start separately)")
        print(f"{'='*60}\n")

        self.app.run(host=host, port=port, debug=debug)
