"""Flask web server for training observation."""
from flask import Flask, render_template, jsonify, send_from_directory, request
from pathlib import Path
import json
import re
import time
from collections import defaultdict
from typing import List, Dict
import yaml

# KPI thresholds
KPI_ELO_WINDOW = 20
KPI_VALUE_LOSS_GREEN = (0.15, 0.40)
KPI_VALUE_LOSS_YELLOW = (0.05, 0.60)
KPI_EVAL_LAG_YELLOW = 10
KPI_EVAL_LAG_RED = 30
KPI_AVG_POSITIONS = 35

# Milestone thresholds
MS_RECOVERY_ELO = 1555
MS_RECOVERY_STREAK = 5
MS_PLATEAU_WINDOW = 50
MS_PLATEAU_BAND = 20
MS_STRONG_PLAY_ELO = 1700


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

    def _checkpoints_dir_for(self, game: str) -> Path:
        if game == 'mandala':
            return self.checkpoints_dir
        elif game == 'lost_cities':
            return self.data_dir / 'lost_cities' / 'checkpoints'
        return None

    def _game_health(self, game: str) -> dict:
        """Get health info for a single game."""
        now = time.time()
        cp_dir = self._checkpoints_dir_for(game)
        elo_file = self._elo_file_for(game)

        # Training heartbeat (live progress from trainer process)
        heartbeat = {}
        hb_dir = self.data_dir if game == 'mandala' else self.data_dir / 'lost_cities'
        hb_file = hb_dir / 'heartbeat.json'
        if hb_file.exists():
            try:
                with open(hb_file) as f:
                    heartbeat = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        # Training: find latest model_iter_*.pt
        train_iter = 0
        train_mtime = 0
        if cp_dir and cp_dir.exists():
            for cp in cp_dir.glob('model_iter_*.pt'):
                m = re.search(r'model_iter_(\d+)\.pt$', cp.name)
                if m:
                    it = int(m.group(1))
                    if it > train_iter:
                        train_iter = it
                        train_mtime = cp.stat().st_mtime

        # Eval: parse elo file for latest entry + count
        elo_iter = 0
        elo_count = 0
        elo_mtime = 0
        if elo_file and elo_file.exists():
            elo_mtime = elo_file.stat().st_mtime
            try:
                with open(elo_file) as f:
                    data = json.load(f)
                for k in data.get('ratings', {}):
                    m_elo = re.match(r'iter_(\d+)$', k)
                    if m_elo:
                        elo_count += 1
                        it = int(m_elo.group(1))
                        if it > elo_iter:
                            elo_iter = it
            except (json.JSONDecodeError, KeyError):
                pass

        # Pending evals: checkpoints that have a predecessor but no Elo
        pending = max(0, train_iter - elo_iter) if train_iter > 0 and elo_iter > 0 else 0

        # Eval heartbeat (live progress from eval daemon)
        eval_hb = {}
        eval_hb_file = hb_dir / 'eval_heartbeat.json'
        if eval_hb_file.exists():
            try:
                with open(eval_hb_file) as f:
                    eval_hb = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        # Status thresholds
        train_age = now - train_mtime if train_mtime else None
        elo_age = now - elo_mtime if elo_mtime else None

        if train_mtime == 0:
            train_status = 'unknown'
        elif train_age < 1800:  # 30 min
            train_status = 'active'
        elif train_age < 7200:  # 2 hours
            train_status = 'slow'
        else:
            train_status = 'stale'

        if elo_mtime == 0:
            eval_status = 'unknown'
        elif elo_age < 7200:  # 2 hours
            eval_status = 'active'
        else:
            eval_status = 'stale'

        # Use heartbeat for live iteration/phase if it's fresher than checkpoint
        hb_iter = heartbeat.get('iteration', 0)
        hb_ts = heartbeat.get('timestamp', 0)
        if hb_iter >= train_iter and hb_ts > 0:
            train_status_from_hb = 'active' if (now - hb_ts) < 300 else train_status
        else:
            train_status_from_hb = train_status

        return {
            'training': {
                'status': train_status_from_hb if hb_ts else train_status,
                'iteration': max(train_iter, hb_iter),
                'last_update': max(train_mtime, hb_ts),
                'seconds_ago': round(now - max(train_mtime, hb_ts)) if (train_mtime or hb_ts) else None,
                'phase': heartbeat.get('phase', ''),
                'detail': heartbeat.get('detail', ''),
                'game': heartbeat.get('game', 0),
                'games_total': heartbeat.get('games_total', 0),
                'total_games': heartbeat.get('total_games', 0),
            },
            'eval': {
                'status': eval_status,
                'iteration': elo_iter,
                'total_entries': elo_count,
                'pending': pending,
                'last_update': elo_mtime,
                'seconds_ago': round(elo_age) if elo_age else None,
                'evaluating_iter': eval_hb.get('evaluating_iter', 0),
                'eval_completed': eval_hb.get('completed', 0),
                'eval_total': eval_hb.get('total', 0),
                'eval_batch_start': eval_hb.get('batch_start', 0),
            },
        }

    def _config_for(self, game: str) -> dict:
        """Load training config for a game."""
        base = self.data_dir.parent
        path = base / 'configs' / ('default.yaml' if game == 'mandala' else 'lost_cities.yaml')
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f)
        return {}

    def _compute_kpis(self, game: str) -> dict:
        """Compute training KPIs and milestones for a game."""
        # Load Elo data
        elo_file = self._elo_file_for(game)
        elo_points = []
        if elo_file and elo_file.exists():
            try:
                with open(elo_file) as f:
                    data = json.load(f)
                for k, v in data.get('ratings', {}).items():
                    m = re.match(r'iter_(\d+)$', k)
                    if m:
                        elo_points.append((int(m.group(1)), v))
                elo_points.sort()
            except (json.JSONDecodeError, KeyError):
                pass

        # Load loss data
        subdir = '' if game == 'mandala' else 'lost_cities'
        losses_path = self.data_dir / subdir / 'losses.jsonl' if subdir else self.data_dir / 'losses.jsonl'
        loss_entries = []
        if losses_path.exists():
            for line in open(losses_path):
                line = line.strip()
                if line:
                    try:
                        loss_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # Get training/eval iterations from health
        health = self._game_health(game)
        train_iter = health['training']['iteration']
        eval_iter = health['eval']['iteration']

        # Load config
        cfg = self._config_for(game)
        buffer_size = cfg.get('training', {}).get('replay_buffer_size', 500000)
        games_per_iter = cfg.get('selfplay', {}).get('games_per_iteration', 100)
        epochs = cfg.get('training', {}).get('epochs_per_iteration', 1)

        kpis = {}

        # 1. Elo Trend — linear regression on last N eval points
        if len(elo_points) >= 3:
            window = elo_points[-KPI_ELO_WINDOW:]
            n = len(window)
            x = list(range(n))
            y = [elo for _, elo in window]
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            den = sum((xi - mean_x) ** 2 for xi in x)
            slope = num / den if den > 0 else 0.0
            if slope > 1.0:
                status = 'green'
            elif slope >= -1.0:
                status = 'yellow'
            else:
                status = 'red'
            kpis['elo_trend'] = {'slope': round(slope, 2), 'status': status,
                                 'label': f'{slope:+.1f}/eval'}
        else:
            kpis['elo_trend'] = {'slope': 0, 'status': 'gray', 'label': 'insufficient data'}

        # 2. Value Loss Health
        if loss_entries:
            val = loss_entries[-1].get('value', 0)
            lo_g, hi_g = KPI_VALUE_LOSS_GREEN
            lo_y, hi_y = KPI_VALUE_LOSS_YELLOW
            if lo_g <= val <= hi_g:
                status = 'green'
            elif lo_y <= val <= hi_y:
                status = 'yellow'
            else:
                status = 'red'
            kpis['value_loss'] = {'value': val, 'status': status, 'label': f'{val:.4f}'}
        else:
            kpis['value_loss'] = {'value': None, 'status': 'gray', 'label': 'no data'}

        # 3. Overtraining Ratio (higher = more diverse = better)
        denom = games_per_iter * KPI_AVG_POSITIONS * epochs
        ratio = buffer_size / denom if denom > 0 else 0
        if ratio > 50:
            status = 'green'
        elif ratio > 10:
            status = 'yellow'
        else:
            status = 'red'
        kpis['overtraining_ratio'] = {'ratio': round(ratio, 1), 'status': status,
                                      'label': f'{ratio:.0f}x diversity'}

        # 4. Eval Lag
        lag = max(0, train_iter - eval_iter)
        if lag < KPI_EVAL_LAG_YELLOW:
            status = 'green'
        elif lag < KPI_EVAL_LAG_RED:
            status = 'yellow'
        else:
            status = 'red'
        kpis['eval_lag'] = {'lag': lag, 'status': status,
                            'label': 'caught up' if lag == 0 else f'{lag} iters behind'}

        # 5. Peak Elo
        if elo_points:
            best_iter, best_elo = max(elo_points, key=lambda p: p[1])
            kpis['peak_elo'] = {'elo': round(best_elo, 1), 'iteration': best_iter,
                                'label': f'{best_elo:.0f} @ iter {best_iter}'}
        else:
            kpis['peak_elo'] = {'elo': None, 'iteration': None, 'label': 'no data'}

        # Milestones
        milestones = {}

        # Recovery: Elo > threshold for N consecutive evals
        streak = 0
        max_streak = 0
        for _, elo in elo_points:
            if elo > MS_RECOVERY_ELO:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        milestones['recovery'] = {
            'condition': f'Elo >{MS_RECOVERY_ELO} for {MS_RECOVERY_STREAK} evals',
            'status': 'achieved' if max_streak >= MS_RECOVERY_STREAK else 'pending',
            'detail': f'best streak: {max_streak}',
        }

        # Plateau: Elo range <= 2*band over last N eval points
        if len(elo_points) >= MS_PLATEAU_WINDOW:
            recent = [elo for _, elo in elo_points[-MS_PLATEAU_WINDOW:]]
            elo_range = max(recent) - min(recent)
            milestones['plateau'] = {
                'condition': f'Elo flat +/-{MS_PLATEAU_BAND} over {MS_PLATEAU_WINDOW} evals',
                'status': 'triggered' if elo_range <= MS_PLATEAU_BAND * 2 else 'inactive',
                'detail': f'range: {elo_range:.0f}' if elo_range <= MS_PLATEAU_BAND * 2 else None,
            }
        else:
            milestones['plateau'] = {
                'condition': f'Elo flat +/-{MS_PLATEAU_BAND} over {MS_PLATEAU_WINDOW} evals',
                'status': 'inactive',
                'detail': 'not enough data',
            }

        # Strong Play: peak Elo > threshold
        peak = kpis['peak_elo']['elo']
        milestones['strong_play'] = {
            'condition': f'Elo >{MS_STRONG_PLAY_ELO}',
            'status': 'achieved' if peak and peak > MS_STRONG_PLAY_ELO else 'pending',
            'detail': f'peak: {peak:.0f}' if peak else None,
        }

        return {'kpis': kpis, 'milestones': milestones}

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main dashboard."""
            return render_template('index.html', game_name=self.game_name)

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

        @self.app.route('/api/heartbeat')
        def api_heartbeat():
            """Get health status for all games."""
            return jsonify({
                'mandala': self._game_health('mandala'),
                'lost_cities': self._game_health('lost_cities'),
                'timestamp': time.time(),
            })

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
                        'winner': data.get('winner'),
                        'timestamp': replay_file.stat().st_mtime
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

        @self.app.route('/api/losses')
        def api_losses():
            """Get training loss history for all games."""
            games = {}
            for key, subdir in [('Mandala', ''), ('Lost Cities', 'lost_cities')]:
                path = self.data_dir / subdir / 'losses.jsonl' if subdir else self.data_dir / 'losses.jsonl'
                if path.exists():
                    entries = []
                    for line in open(path):
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                    games[key] = entries
            return jsonify(games)

        @self.app.route('/api/elo')
        def api_elo():
            """Get Elo ratings for all games, including per-iter win/loss stats."""
            games = {}
            for name, path in [('Mandala', self.data_dir / 'elo_ratings.json'),
                               ('Lost Cities', self.data_dir / 'lost_cities' / 'elo_ratings.json')]:
                if not path.exists():
                    continue
                with open(path) as f:
                    data = json.load(f)
                ratings = data.get('ratings', {})
                stats = data.get('stats', {})
                bench = data.get('benchmark_stats', {})
                points = []
                for k, v in ratings.items():
                    m = re.match(r'iter_(\d+)$', k)
                    if not m:
                        continue
                    it = int(m.group(1))
                    s = stats.get(str(it), {})
                    b = bench.get(str(it), {})
                    points.append({
                        'iter': it, 'elo': v,
                        'wins': s.get('wins'), 'losses': s.get('losses'), 'draws': s.get('draws'),
                        'vs_random': b.get('vs_random'),
                        'vs_strategy': b.get('vs_strategy'),
                    })
                games[name] = sorted(points, key=lambda x: x['iter'])
            return jsonify(games)

        @self.app.route('/api/kpis')
        def api_kpis():
            """Get training KPI indicators for all games."""
            return jsonify({
                'mandala': self._compute_kpis('mandala'),
                'lost_cities': self._compute_kpis('lost_cities'),
            })

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
