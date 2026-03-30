#!/usr/bin/env python3
"""
Background evaluation worker. Runs on CPU to avoid GPU contention with training.

Spawned by the trainer after each iteration. Loads two checkpoints, plays an
arena match using Python MCTS, and writes Elo ratings to disk.

Usage (called automatically by trainer, not manually):
    python3 scripts/eval_worker.py --config configs/default.yaml \
        --iteration 10 --checkpoint-dir data/checkpoints \
        --elo-file data/elo_ratings.json --log-dir data/logs
"""
import argparse
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.network.model import MandalaNet
from mandala_rl.evaluation.arena import Arena
from mandala_rl.evaluation.elo import EloRating


def main():
    parser = argparse.ArgumentParser(description='Background Elo evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--elo-file', required=True)
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    net_cfg = config['network']
    eval_cfg = config['evaluation']
    device = 'cpu'

    checkpoint_dir = Path(args.checkpoint_dir)
    current_iter = args.iteration
    prev_iter = current_iter - 1

    current_path = checkpoint_dir / f'model_iter_{current_iter}.pt'
    prev_path = checkpoint_dir / f'model_iter_{prev_iter}.pt'

    if not current_path.exists() or not prev_path.exists():
        missing = current_path if not current_path.exists() else prev_path
        print(f"[EVAL] Skipping: {missing.name} not found")
        return

    # Load models on CPU
    def load_model(path):
        m = MandalaNet(
            input_channels=net_cfg['input_channels'],
            num_actions=net_cfg['num_actions'],
            num_res_blocks=net_cfg['num_res_blocks'],
            channels=net_cfg['channels'],
            belief_size=net_cfg.get('belief_size', 12),
            phase_aware_policy=net_cfg.get('phase_aware_policy', False),
            factored_policy=net_cfg.get('factored_policy', False),
        ).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        return m

    print(f"[EVAL] iter_{current_iter} vs iter_{prev_iter} (CPU, "
          f"{eval_cfg.get('eval_num_games', 20)} games, "
          f"{eval_cfg.get('eval_mcts_simulations', 400)} sims)")

    current_model = load_model(current_path)
    prev_model = load_model(prev_path)

    # Create game engine based on network config
    if net_cfg['num_actions'] == 30:
        from mandala_rl.game.engine import MandalaGame
        game = MandalaGame()
    else:
        from lost_cities.game.engine import LostCitiesGame
        game = LostCitiesGame()
    arena = Arena(
        game=game,
        mcts_simulations=eval_cfg.get('eval_mcts_simulations', 400),
        c_puct=config['mcts']['c_puct'],
        device=device
    )
    results = arena.play_match(
        current_model, prev_model,
        num_games=eval_cfg.get('eval_num_games', 20),
        seed=current_iter
    )

    # Update Elo ratings
    elo_file = Path(args.elo_file)
    elo = EloRating(initial_rating=1500.0, k_factor=32.0)
    if elo_file.exists():
        elo.load(elo_file)

    current_id = f"iter_{current_iter}"
    prev_id = f"iter_{prev_iter}"
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
          f"({win_rate:.0%} win rate)")
    print(f"[EVAL] Elo: {current_id}={elo.get_rating(current_id):.0f}, "
          f"{prev_id}={elo.get_rating(prev_id):.0f}")

    # Write to Tensorboard (separate event file, same log dir)
    if args.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.log_dir)
            writer.add_scalar('Evaluation/WinRate', win_rate, current_iter)
            writer.add_scalar('Evaluation/CurrentElo', elo.get_rating(current_id), current_iter)
            writer.add_scalar('Evaluation/Wins', results['model1_wins'], current_iter)
            writer.add_scalar('Evaluation/Losses', results['model2_wins'], current_iter)
            writer.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
