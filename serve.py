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
import threading
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

from flask import Flask, render_template, jsonify

template_dir = project_root / 'templates'
app = Flask(__name__, template_folder=str(template_dir))

# Configuration from environment (with sensible defaults)
MCTS_SIMULATIONS = int(os.environ.get('MCTS_SIMULATIONS', '30'))
MANDALA_CONFIG = os.environ.get('MANDALA_CONFIG', 'configs/default.yaml')
LC_CONFIG = os.environ.get('LC_CONFIG', 'configs/lost_cities.yaml')
MANDALA_CHECKPOINT_DIR = os.environ.get('MANDALA_CHECKPOINT_DIR', 'data/checkpoints')
LC_CHECKPOINT_DIR = os.environ.get('LC_CHECKPOINT_DIR', 'data/lost_cities/checkpoints')
DEPLOY_DIR = os.environ.get('DEPLOY_DIR', 'data/deploy')

# Track which games loaded successfully
loaded_games = {}
_mandala_server = None
_lc_server = None


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
        _mandala_server = server
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
        _lc_server = server
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


# --- Auto-reload: watch training dirs for newer checkpoints ---
AUTO_RELOAD = os.environ.get('AUTO_RELOAD', '1') == '1'
AUTO_RELOAD_INTERVAL = int(os.environ.get('AUTO_RELOAD_INTERVAL', '60'))

def _find_latest_iter(checkpoint_dir):
    """Find highest iteration checkpoint in a directory."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None, None
    best_iter = -1
    best_path = None
    for cp in cp_dir.glob("model_iter_*.pt"):
        try:
            iter_num = int(cp.stem.split('_')[-1])
            if iter_num > best_iter:
                best_iter = iter_num
                best_path = cp
        except (ValueError, IndexError):
            continue
    return best_iter, best_path

def _strip_checkpoint(source_path, deploy_dir):
    """Strip training checkpoint to model_state_dict only, save to deploy dir."""
    import torch
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=False)
    lightweight = {
        'model_state_dict': checkpoint['model_state_dict'],
        'iteration': checkpoint.get('iteration', 'unknown'),
        'total_games': checkpoint.get('total_games', 'unknown'),
    }
    deploy_path = Path(deploy_dir) / 'model.pt'
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lightweight, deploy_path)
    src_mb = source_path.stat().st_size / (1024 * 1024)
    dst_mb = deploy_path.stat().st_size / (1024 * 1024)
    print(f"[auto-reload] Stripped {src_mb:.1f}MB → {dst_mb:.1f}MB: {deploy_path}")
    return deploy_path, lightweight['iteration'], lightweight['total_games']


def _auto_reload_worker():
    """Background thread: checks for newer checkpoints, strips to lightweight, hot-swaps."""
    global _mandala_server, _lc_server
    print(f"[auto-reload] Watching for new checkpoints every {AUTO_RELOAD_INTERVAL}s")
    while True:
        time.sleep(AUTO_RELOAD_INTERVAL)
        try:
            # Check Mandala
            if _mandala_server and MANDALA_CHECKPOINT_DIR:
                latest_iter, latest_path = _find_latest_iter(MANDALA_CHECKPOINT_DIR)
                current_iter = _mandala_server.iteration
                if latest_iter is not None and isinstance(current_iter, int) and latest_iter > current_iter:
                    print(f"[auto-reload] Mandala: iter {current_iter} → {latest_iter}")
                    try:
                        deploy_path, iteration, total_games = _strip_checkpoint(
                            latest_path, f"{DEPLOY_DIR}/mandala"
                        )
                        from play_vs_ai_web import MandalaModelServer
                        new_server = MandalaModelServer(
                            deploy_path, MANDALA_CONFIG, MCTS_SIMULATIONS
                        )
                        _mandala_server.__dict__.update(new_server.__dict__)
                        loaded_games['mandala']['iteration'] = iteration
                        loaded_games['mandala']['total_games'] = total_games
                        loaded_games['mandala']['checkpoint'] = deploy_path.name
                        print(f"[auto-reload] Mandala reloaded: iter {iteration}")
                    except Exception as e:
                        print(f"[auto-reload] Mandala reload failed: {e}")

            # Check Lost Cities
            if _lc_server and LC_CHECKPOINT_DIR:
                latest_iter, latest_path = _find_latest_iter(LC_CHECKPOINT_DIR)
                current_iter = _lc_server.iteration
                if latest_iter is not None and isinstance(current_iter, int) and latest_iter > current_iter:
                    print(f"[auto-reload] Lost Cities: iter {current_iter} → {latest_iter}")
                    try:
                        deploy_path, iteration, total_games = _strip_checkpoint(
                            latest_path, f"{DEPLOY_DIR}/lost_cities"
                        )
                        from play_vs_ai_web_lc import LCModelServer
                        new_server = LCModelServer(
                            deploy_path, LC_CONFIG, MCTS_SIMULATIONS
                        )
                        _lc_server.__dict__.update(new_server.__dict__)
                        loaded_games['lost_cities']['iteration'] = iteration
                        loaded_games['lost_cities']['total_games'] = total_games
                        loaded_games['lost_cities']['checkpoint'] = deploy_path.name
                        print(f"[auto-reload] Lost Cities reloaded: iter {iteration}")
                    except Exception as e:
                        print(f"[auto-reload] Lost Cities reload failed: {e}")
        except Exception as e:
            print(f"[auto-reload] Error: {e}")

if AUTO_RELOAD:
    _reload_thread = threading.Thread(target=_auto_reload_worker, daemon=True)
    _reload_thread.start()


@app.route('/')
def landing():
    return render_template('index.html', games=loaded_games)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'games': list(loaded_games.keys()),
        'mcts_simulations': MCTS_SIMULATIONS,
        'auto_reload': AUTO_RELOAD,
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
