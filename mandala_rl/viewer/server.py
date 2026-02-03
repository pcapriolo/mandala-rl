"""Flask web server for training observation."""
from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
import json
from typing import List, Dict
import yaml


class TrainingObserver:
    """Web server for observing training progress."""

    def __init__(self, data_dir: Path, config_path: Path = None):
        self.data_dir = Path(data_dir)
        self.replays_dir = self.data_dir / 'replays'
        self.checkpoints_dir = self.data_dir / 'checkpoints'
        self.logs_dir = self.data_dir / 'logs'
        self.config_path = config_path

        self.app = Flask(__name__,
                        template_folder=str(Path(__file__).parent / 'templates'),
                        static_folder=str(Path(__file__).parent / 'static'))

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main dashboard."""
            return render_template('index.html')

        @self.app.route('/api/status')
        def api_status():
            """Get training status."""
            checkpoints = list(self.checkpoints_dir.glob('*.pt')) if self.checkpoints_dir.exists() else []
            replays = list(self.replays_dir.glob('*.json')) if self.replays_dir.exists() else []

            status = {
                'num_checkpoints': len(checkpoints),
                'latest_checkpoint': checkpoints[-1].name if checkpoints else None,
                'num_replays': len(replays),
                'tensorboard_url': 'http://localhost:6006'
            }

            return jsonify(status)

        @self.app.route('/api/replays')
        def api_replays():
            """Get list of available replays."""
            if not self.replays_dir.exists():
                return jsonify([])

            replays = []
            for replay_file in sorted(self.replays_dir.glob('*.json'), reverse=True):
                try:
                    with open(replay_file, 'r') as f:
                        data = json.load(f)
                    replays.append({
                        'filename': replay_file.name,
                        'game_id': data['game_id'],
                        'num_moves': len(data['moves']),
                        'final_score': data.get('final_score'),
                        'winner': data.get('winner')
                    })
                except:
                    pass

            return jsonify(replays[:50])  # Latest 50 games

        @self.app.route('/api/replay/<filename>')
        def api_replay(filename):
            """Get specific replay data."""
            replay_file = self.replays_dir / filename
            if not replay_file.exists():
                return jsonify({'error': 'Replay not found'}), 404

            with open(replay_file, 'r') as f:
                data = json.load(f)

            return jsonify(data)

        @self.app.route('/replay/<filename>')
        def view_replay(filename):
            """View replay in browser."""
            return render_template('replay.html', filename=filename)

    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Start the web server."""
        print(f"\n{'='*60}")
        print("🌐 Mandala Training Observer")
        print(f"{'='*60}")
        print(f"Dashboard: http://{host}:{port}")
        print(f"Tensorboard: http://{host}:6006 (start separately)")
        print(f"{'='*60}\n")

        self.app.run(host=host, port=port, debug=debug)
