#!/usr/bin/env python3
"""
Standalone Elo evaluation daemon. Watches for checkpoints missing Elo entries
and evaluates them sequentially on CPU. Decoupled from training — runs alongside
it as a separate process, catching up on any missed evaluations.

Usage:
    # Daemon mode (runs forever):
    python scripts/eval_daemon.py --config configs/default.yaml

    # Catch-up mode (run once and exit):
    python scripts/eval_daemon.py --config configs/default.yaml --once

    # Both games on RunPod:
    nohup python scripts/eval_daemon.py --config configs/default.yaml &
    nohup python scripts/eval_daemon.py --config configs/lost_cities.yaml &
"""
import argparse
import re
import time
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.network.model import MandalaNet
from mandala_rl.evaluation.arena import Arena
from mandala_rl.evaluation.elo import EloRating


def find_missing(checkpoint_dir, elo_file):
    """Find iterations that have checkpoints but no Elo entry."""
    # Get all checkpoint iterations
    cp_iters = set()
    for cp in checkpoint_dir.glob('model_iter_*.pt'):
        m = re.search(r'model_iter_(\d+)\.pt$', cp.name)
        if m:
            cp_iters.add(int(m.group(1)))

    if not cp_iters:
        return []

    # Get existing Elo entries
    elo_iters = set()
    if elo_file.exists():
        elo = EloRating()
        elo.load(elo_file)
        for key in elo.ratings:
            m = re.match(r'iter_(\d+)$', key)
            if m:
                elo_iters.add(int(m.group(1)))

    # Missing = has checkpoint AND previous checkpoint exists, but no Elo entry
    missing = []
    for i in sorted(cp_iters):
        if i not in elo_iters and (i - 1) in cp_iters:
            missing.append(i)

    return missing


def run_eval(iteration, checkpoint_dir, elo_file, log_dir, net_cfg, eval_cfg, mcts_cfg, device='cpu'):
    """Run evaluation for a single iteration."""
    current_path = checkpoint_dir / f'model_iter_{iteration}.pt'
    prev_path = checkpoint_dir / f'model_iter_{iteration - 1}.pt'

    if not current_path.exists() or not prev_path.exists():
        print(f"[EVAL] Skipping iter {iteration}: checkpoint missing")
        return False

    def load_model(path):
        m = MandalaNet(
            input_channels=net_cfg['input_channels'],
            num_actions=net_cfg['num_actions'],
            num_res_blocks=net_cfg['num_res_blocks'],
            channels=net_cfg['channels']
        ).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        return m

    num_games = eval_cfg.get('eval_num_games', 20)
    mcts_sims = eval_cfg.get('eval_mcts_simulations', 400)
    print(f"[EVAL] iter_{iteration} vs iter_{iteration - 1} "
          f"({num_games} games, {mcts_sims} sims)")

    current_model = load_model(current_path)
    prev_model = load_model(prev_path)

    # Auto-detect game type from config
    if net_cfg['num_actions'] == 30:
        from mandala_rl.game.engine import MandalaGame
        game = MandalaGame()
    else:
        from lost_cities.game.engine import LostCitiesGame
        game = LostCitiesGame()

    arena = Arena(
        game=game,
        mcts_simulations=mcts_sims,
        c_puct=mcts_cfg['c_puct'],
        device=device
    )
    results = arena.play_match(
        current_model, prev_model,
        num_games=num_games,
        seed=iteration
    )

    # Update Elo
    elo = EloRating(initial_rating=1500.0, k_factor=32.0)
    if elo_file.exists():
        elo.load(elo_file)

    current_id = f"iter_{iteration}"
    prev_id = f"iter_{iteration - 1}"
    total = results['total_games']
    score_current = results['model1_wins'] + 0.5 * results['draws']
    score_prev = results['model2_wins'] + 0.5 * results['draws']

    rating_current = elo.get_rating(current_id)
    rating_prev = elo.get_rating(prev_id)
    expected_current = total / (1.0 + 10 ** ((rating_prev - rating_current) / 400.0))

    elo.ratings[current_id] = rating_current + elo.k_factor * (score_current - expected_current)
    elo.ratings[prev_id] = rating_prev + elo.k_factor * (score_prev - (total - expected_current))

    elo_file.parent.mkdir(parents=True, exist_ok=True)
    elo.save(elo_file)

    win_rate = results['model1_wins'] / total
    print(f"[EVAL] Result: {results['model1_wins']}W/{results['model2_wins']}L/{results['draws']}D "
          f"({win_rate:.0%})")
    print(f"[EVAL] Elo: {current_id}={elo.get_rating(current_id):.0f}, "
          f"{prev_id}={elo.get_rating(prev_id):.0f}")

    # Tensorboard
    if log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar('Evaluation/WinRate', win_rate, iteration)
            writer.add_scalar('Evaluation/CurrentElo', elo.get_rating(current_id), iteration)
            writer.add_scalar('Evaluation/Wins', results['model1_wins'], iteration)
            writer.add_scalar('Evaluation/Losses', results['model2_wins'], iteration)
            writer.close()
        except Exception:
            pass

    return True


def main():
    parser = argparse.ArgumentParser(description='Elo evaluation daemon')
    parser.add_argument('--config', required=True, help='Training config YAML')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between checks')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda, mps')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    net_cfg = config['network']
    eval_cfg = config['evaluation']
    mcts_cfg = config['mcts']

    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir',
                          config.get('checkpoint_dir', 'data/checkpoints')))
    elo_file = Path(config.get('paths', {}).get('elo_file',
                    config.get('elo_file', 'data/elo_ratings.json')))
    log_dir = Path(config.get('paths', {}).get('log_dir',
                   config.get('log_dir', 'data/logs')))

    # Resolve device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    game_name = 'Lost Cities' if net_cfg['num_actions'] != 30 else 'Mandala'
    print(f"[EVAL DAEMON] {game_name} (device: {device})")
    print(f"[EVAL DAEMON] Checkpoints: {checkpoint_dir}")
    print(f"[EVAL DAEMON] Elo file: {elo_file}")
    if not args.once:
        print(f"[EVAL DAEMON] Checking every {args.interval}s (Ctrl+C to stop)")

    while True:
        missing = find_missing(checkpoint_dir, elo_file)
        if missing:
            print(f"[EVAL DAEMON] Missing Elo for iterations: {missing}")
            for iteration in missing:
                run_eval(iteration, checkpoint_dir, elo_file, log_dir,
                         net_cfg, eval_cfg, mcts_cfg, device=device)
        else:
            print(f"[EVAL DAEMON] All checkpoints evaluated")

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
