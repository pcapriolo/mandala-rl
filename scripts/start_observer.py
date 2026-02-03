#!/usr/bin/env python3
"""
Start the web-based training observer.

Usage:
    python3 scripts/start_observer.py
    python3 scripts/start_observer.py --port 5000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.viewer.server import TrainingObserver


def main():
    parser = argparse.ArgumentParser(description='Start Mandala training observer web server')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                      help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port to bind to (default: 5000)')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Data directory (default: data)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Create directories if they don't exist
    (data_dir / 'replays').mkdir(parents=True, exist_ok=True)
    (data_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (data_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Start observer
    observer = TrainingObserver(data_dir=data_dir)
    observer.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
