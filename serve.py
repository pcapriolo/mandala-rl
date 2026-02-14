#!/usr/bin/env python3
"""
Combined production server for Mandala RL games.

Serves both Mandala and Lost Cities from a single Flask app.
Designed for Railway deployment via gunicorn.

Usage:
    # Local dev:
    python serve.py --port 5000

    # Production (Railway):
    gunicorn serve:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from flask import Flask, render_template, jsonify

template_dir = project_root / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# Configuration from environment (with sensible defaults)
MCTS_SIMULATIONS = int(os.environ.get('MCTS_SIMULATIONS', '200'))
MANDALA_CONFIG = os.environ.get('MANDALA_CONFIG', 'configs/default.yaml')
LC_CONFIG = os.environ.get('LC_CONFIG', 'configs/lost_cities.yaml')
MANDALA_CHECKPOINT_DIR = os.environ.get('MANDALA_CHECKPOINT_DIR', 'data/checkpoints')
LC_CHECKPOINT_DIR = os.environ.get('LC_CHECKPOINT_DIR', 'data/lost_cities/checkpoints')
DEPLOY_DIR = os.environ.get('DEPLOY_DIR', 'data/deploy')

# Track which games loaded successfully
loaded_games = {}


def find_checkpoint(deploy_dir, checkpoint_dir):
    """Find best checkpoint: prefer deploy/, fall back to checkpoints/."""
    deploy_path = Path(deploy_dir)
    if deploy_path.exists():
        pts = sorted(deploy_path.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
    cp_path = Path(checkpoint_dir)
    if cp_path.exists():
        pts = sorted(cp_path.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
    return None


# Load Mandala
mandala_cp = find_checkpoint(f"{DEPLOY_DIR}/mandala", MANDALA_CHECKPOINT_DIR)
if mandala_cp:
    try:
        from play_vs_ai_web import create_mandala_blueprint
        bp, server = create_mandala_blueprint(
            checkpoint_path=mandala_cp,
            config_path=MANDALA_CONFIG,
            simulations=MCTS_SIMULATIONS,
            base_url='/mandala',
            checkpoint_dir=MANDALA_CHECKPOINT_DIR
        )
        app.register_blueprint(bp, url_prefix='/mandala')
        loaded_games['mandala'] = {
            'iteration': server.iteration,
            'total_games': server.total_games,
            'checkpoint': Path(server.checkpoint_path).name,
        }
        print(f"[serve] Mandala loaded: iter {server.iteration}")
    except Exception as e:
        print(f"[serve] Failed to load Mandala: {e}")
else:
    print("[serve] No Mandala checkpoint found, skipping")

# Load Lost Cities
lc_cp = find_checkpoint(f"{DEPLOY_DIR}/lost_cities", LC_CHECKPOINT_DIR)
if lc_cp:
    try:
        from play_vs_ai_web_lc import create_lc_blueprint
        bp, server = create_lc_blueprint(
            checkpoint_path=lc_cp,
            config_path=LC_CONFIG,
            simulations=MCTS_SIMULATIONS,
            base_url='/lost-cities',
            checkpoint_dir=LC_CHECKPOINT_DIR
        )
        app.register_blueprint(bp, url_prefix='/lost-cities')
        loaded_games['lost_cities'] = {
            'iteration': server.iteration,
            'total_games': server.total_games,
            'checkpoint': Path(server.checkpoint_path).name,
        }
        print(f"[serve] Lost Cities loaded: iter {server.iteration}")
    except Exception as e:
        print(f"[serve] Failed to load Lost Cities: {e}")
else:
    print("[serve] No Lost Cities checkpoint found, skipping")


@app.route('/')
def landing():
    return render_template('index.html', games=loaded_games)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'games': list(loaded_games.keys()),
        'mcts_simulations': MCTS_SIMULATIONS,
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Combined game server")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '5000')))
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()

    if not loaded_games:
        print("\nNo games loaded! Make sure checkpoints exist in data/deploy/ or data/checkpoints/")
        print("Run: python scripts/create_deploy_checkpoint.py")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"MANDALA RL GAME SERVER")
    print(f"{'='*60}")
    print(f"Games: {', '.join(loaded_games.keys())}")
    print(f"MCTS simulations: {MCTS_SIMULATIONS}")
    print(f"\nhttp://{args.host}:{args.port}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)
