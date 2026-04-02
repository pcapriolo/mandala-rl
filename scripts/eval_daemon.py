#!/usr/bin/env python3
"""
Tournament-style Elo evaluation daemon.

Instead of comparing iter N vs N-1 (too noisy with 20 games), each checkpoint
plays a mini-tournament against ~20 spread-out prior checkpoints. All
participants' Elo ratings get updated, giving stable cumulative measurements.

Usage:
    # Daemon mode (runs forever):
    python scripts/eval_daemon.py --config configs/default.yaml

    # Single pass (run once and exit):
    python scripts/eval_daemon.py --config configs/default.yaml --once

    # Custom tournament settings:
    python scripts/eval_daemon.py --config configs/default.yaml \
        --tournament-freq 5 --num-opponents 20 --games-per-opponent 5
"""
import argparse
import json
import os
import random
import re
import tempfile
import time
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandala_rl.network.model import MandalaNet
from mandala_rl.evaluation.fast_arena import FastArena
from mandala_rl.evaluation.elo import EloRating
from mandala_rl.evaluation.benchmark_bots import RandomBot, MandalaStrategyBot, LostCitiesStrategyBot, DominionStrategyBot


def load_elo_with_metadata(elo_file):
    """Load Elo JSON preserving the tournament_evaluated set, stats, benchmarks, and tournament log."""
    if elo_file.exists():
        with open(elo_file) as f:
            data = json.load(f)
        elo = EloRating(
            initial_rating=data.get('initial_rating', 1500.0),
            k_factor=data.get('k_factor', 32.0)
        )
        elo.ratings = data.get('ratings', {})
        evaluated = set(data.get('tournament_evaluated', []))
        stats = data.get('stats', {})
        benchmark_stats = data.get('benchmark_stats', {})
        tournament_log = data.get('tournament_log', {})
        return elo, evaluated, stats, benchmark_stats, tournament_log
    return EloRating(initial_rating=1500.0, k_factor=32.0), set(), {}, {}, {}


def save_elo_with_metadata(elo_file, elo, evaluated, stats=None, benchmark_stats=None,
                           tournament_log=None):
    """Save Elo JSON including tournament_evaluated list, per-iter stats, benchmarks, and tournament log.

    Uses atomic write (tempfile + rename) to prevent corruption on crash.
    """
    data = {
        'initial_rating': elo.initial_rating,
        'k_factor': elo.k_factor,
        'ratings': elo.ratings,
        'tournament_evaluated': sorted(evaluated),
        'stats': stats or {},
        'benchmark_stats': benchmark_stats or {},
        'tournament_log': tournament_log or {},
    }
    elo_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(elo_file.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, str(elo_file))
    except Exception:
        os.unlink(tmp_path)
        raise


def get_checkpoint_iters(checkpoint_dir):
    """Get all iteration numbers that have checkpoint files."""
    iters = set()
    for cp in checkpoint_dir.glob('model_iter_*.pt'):
        m = re.search(r'model_iter_(\d+)\.pt$', cp.name)
        if m:
            iters.add(int(m.group(1)))
    return iters


def select_opponents(current_iter, available_iters, num_opponents=20):
    """Pick opponents via stratified random sampling across history.

    Divides prior checkpoints into N equal buckets and randomly picks one
    from each bucket. Always includes the earliest and most recent checkpoint
    for anchoring. This gives good coverage while adding variety across evals.
    """
    candidates = sorted([i for i in available_iters if i < current_iter])
    if not candidates:
        return []
    if len(candidates) <= num_opponents:
        return candidates
    # Stratified random: split into num_opponents buckets, pick one from each
    bucket_size = len(candidates) / num_opponents
    selected = []
    for i in range(num_opponents):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        selected.append(random.choice(candidates[start:end]))
    # Always anchor earliest and most recent
    selected[0] = candidates[0]
    selected[-1] = candidates[-1]
    return sorted(set(selected))


def find_tournament_needed(checkpoint_dir, evaluated, tournament_freq=5, start_iter=1):
    """Find iterations that need tournament evaluation."""
    cp_iters = get_checkpoint_iters(checkpoint_dir)
    needed = []
    for i in sorted(cp_iters):
        if i >= start_iter and i % tournament_freq == 0 and i not in evaluated:
            needed.append(i)
    return needed


def load_model(path, net_cfg, device):
    """Load a model checkpoint."""
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


def run_tournament(iteration, checkpoint_dir, elo, net_cfg, eval_cfg, mcts_cfg,
                   device='cpu', num_opponents=20, games_per_opponent=5,
                   start_iter=1, write_heartbeat=None):
    """Run tournament: all games in parallel via one BatchedMCTS session.

    Returns dict with 'stats' and 'tournament_log' on success, None on failure.
    """
    cp_iters = get_checkpoint_iters(checkpoint_dir)
    available = {i for i in cp_iters if i >= start_iter}
    opponent_iters = select_opponents(iteration, available, num_opponents)

    if not opponent_iters:
        print(f"[EVAL] No opponents for iter {iteration}")
        return None

    current_path = checkpoint_dir / f'model_iter_{iteration}.pt'
    if not current_path.exists():
        print(f"[EVAL] Missing checkpoint: {current_path}")
        return None

    game_type = "mandala" if net_cfg['num_actions'] in (108, 150) else ("dominion" if net_cfg['num_actions'] == 131 else "lost_cities")
    mcts_sims = eval_cfg.get('eval_mcts_simulations', 400)

    # Load all models upfront
    if write_heartbeat:
        write_heartbeat(iteration, 0, len(opponent_iters))

    try:
        current_model = load_model(current_path, net_cfg, device)
    except Exception as e:
        print(f"[EVAL] Corrupt checkpoint {current_path}: {e}")
        return None

    current_id = f"iter_{iteration}"

    # Inherit predecessor's rating if available
    if current_id not in elo.ratings:
        for prev in range(iteration - 1, 0, -1):
            prev_id = f"iter_{prev}"
            if prev_id in elo.ratings:
                elo.ratings[current_id] = elo.ratings[prev_id]
                break

    # Snapshot rating before tournament
    elo_before = elo.get_rating(current_id)

    opponents = []
    for opp_iter in opponent_iters:
        opp_path = checkpoint_dir / f'model_iter_{opp_iter}.pt'
        if not opp_path.exists():
            continue
        try:
            opp_model = load_model(opp_path, net_cfg, device)
        except Exception as e:
            print(f"[EVAL] Corrupt checkpoint {opp_path}, skipping: {e}")
            continue
        opponents.append((opp_model, f"iter_{opp_iter}"))

    if not opponents:
        print(f"[EVAL] No valid opponent checkpoints for iter {iteration}")
        return None

    total_games = len(opponents) * games_per_opponent
    print(f"[EVAL] Tournament: iter {iteration} vs {len(opponents)} opponents "
          f"({games_per_opponent} games each = {total_games} parallel games, "
          f"{mcts_sims} sims)")

    arena = FastArena(
        game_type=game_type,
        mcts_simulations=mcts_sims,
        c_puct=mcts_cfg['c_puct'],
        device=device
    )

    all_results = arena.play_tournament(
        current_model, opponents,
        games_per_opponent=games_per_opponent,
        seed=iteration * 10000,
    )

    # Collect all per-game results, then shuffle before applying Elo.
    # Since Elo updates are sequential (each game changes ratings before the
    # next), the order matters. Shuffling removes opponent-ordering bias.
    total_wins = 0
    total_losses = 0
    total_draws = 0

    all_game_results = []
    for opp_name, r in all_results.items():
        for result in r['game_results']:
            if result == 'current':
                all_game_results.append((current_id, opp_name, current_id))
            elif result == 'opponent':
                all_game_results.append((current_id, opp_name, opp_name))
            else:
                all_game_results.append((current_id, opp_name, None))

        total_wins += r['current_wins']
        total_losses += r['opponent_wins']
        total_draws += r['draws']

        print(f"  vs {opp_name}: {r['current_wins']}W/"
              f"{r['opponent_wins']}L/{r['draws']}D")

    # Shuffle and apply Elo updates in random order
    random.shuffle(all_game_results)
    for p1, p2, winner in all_game_results:
        elo.record_match(p1, p2, winner)

    elo_after = elo.get_rating(current_id)

    agg_total = total_wins + total_losses + total_draws
    win_rate = total_wins / agg_total if agg_total > 0 else 0
    print(f"[EVAL] Tournament done: {total_wins}W/{total_losses}L/{total_draws}D "
          f"({win_rate:.0%})")
    print(f"[EVAL] Elo: {current_id} = {elo_after:.0f} (delta: {elo_after - elo_before:+.1f})")

    if write_heartbeat:
        write_heartbeat(iteration, len(opponents), len(opponents))

    # Build per-participant stats (current + each opponent)
    all_stats = {}
    all_stats[str(iteration)] = {'wins': total_wins, 'losses': total_losses, 'draws': total_draws}
    for opp_name, r in all_results.items():
        m = re.match(r'iter_(\d+)$', opp_name)
        if m:
            opp_iter = m.group(1)
            all_stats[opp_iter] = {
                'wins': r['opponent_wins'],
                'losses': r['current_wins'],
                'draws': r['draws'],
            }

    # Build per-opponent breakdown for tournament log
    opponent_detail = {}
    for opp_name, r in all_results.items():
        opponent_detail[opp_name] = {
            'wins': r['current_wins'],
            'losses': r['opponent_wins'],
            'draws': r['draws'],
        }

    return {
        'stats': all_stats,
        'tournament_log': {
            'timestamp': time.time(),
            'elo_before': round(elo_before, 2),
            'elo_after': round(elo_after, 2),
            'elo_delta': round(elo_after - elo_before, 2),
            'num_opponents': len(opponents),
            'games_played': agg_total,
            'mcts_sims': mcts_sims,
            'opponents': opponent_detail,
            'results_summary': {'wins': total_wins, 'losses': total_losses, 'draws': total_draws},
        },
    }


def run_benchmarks(iteration, checkpoint_dir, net_cfg, mcts_cfg,
                   device='cpu', games_per_benchmark=10):
    """Play benchmark games vs random and strategy bots. No Elo impact."""
    current_path = checkpoint_dir / f'model_iter_{iteration}.pt'
    if not current_path.exists():
        return None

    game_type = "mandala" if net_cfg['num_actions'] in (108, 150) else ("dominion" if net_cfg['num_actions'] == 131 else "lost_cities")
    num_actions = net_cfg['num_actions']
    try:
        current_model = load_model(current_path, net_cfg, device)
    except Exception as e:
        print(f"[EVAL] Corrupt checkpoint for benchmarks {current_path}: {e}")
        return None

    random_bot = RandomBot(num_actions)
    if game_type == 'mandala':
        strategy_bot = MandalaStrategyBot(num_actions)
    elif game_type == 'dominion':
        strategy_bot = DominionStrategyBot(num_actions)
    else:
        strategy_bot = LostCitiesStrategyBot(num_actions)

    arena = FastArena(
        game_type=game_type,
        mcts_simulations=50,
        c_puct=mcts_cfg['c_puct'],
        device=device
    )

    results = arena.play_tournament(
        current_model,
        opponents=[(random_bot, "random"), (strategy_bot, "strategy")],
        games_per_opponent=games_per_benchmark,
        seed=iteration * 10000 + 1,
    )

    bench = {
        'vs_random': {
            'wins': results['random']['current_wins'],
            'losses': results['random']['opponent_wins'],
            'draws': results['random']['draws'],
        },
        'vs_strategy': {
            'wins': results['strategy']['current_wins'],
            'losses': results['strategy']['opponent_wins'],
            'draws': results['strategy']['draws'],
        },
    }
    rw, rl = bench['vs_random']['wins'], bench['vs_random']['losses']
    sw, sl = bench['vs_strategy']['wins'], bench['vs_strategy']['losses']
    print(f"[EVAL] Benchmarks iter {iteration}: "
          f"vs Random {rw}W/{rl}L, vs Strategy {sw}W/{sl}L")
    return bench


def main():
    parser = argparse.ArgumentParser(description='Tournament Elo evaluation daemon')
    parser.add_argument('--config', required=True, help='Training config YAML')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between checks')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda, mps')
    parser.add_argument('--tournament-freq', type=int, default=1,
                        help='Evaluate every N iterations (default: 1)')
    parser.add_argument('--num-opponents', type=int, default=100,
                        help='Prior checkpoints to play against (default: 100)')
    parser.add_argument('--games-per-opponent', type=int, default=1,
                        help='Games per opponent (default: 1)')
    parser.add_argument('--mcts-sims', type=int, default=None,
                        help='Override MCTS simulations (default: from config)')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    net_cfg = config['network']
    eval_cfg = config['evaluation']
    mcts_cfg = config['mcts']

    if args.mcts_sims is not None:
        eval_cfg['eval_mcts_simulations'] = args.mcts_sims

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

    game_name = 'Mandala' if net_cfg['num_actions'] in (108, 150) else ('Dominion' if net_cfg['num_actions'] == 131 else 'Lost Cities')
    start_iter = eval_cfg.get('eval_start_iteration', 1)

    print(f"[EVAL DAEMON] {game_name} — Tournament mode (device: {device})")
    print(f"[EVAL DAEMON] Every {args.tournament_freq} iters, "
          f"{args.num_opponents} opponents, {args.games_per_opponent} games each")
    print(f"[EVAL DAEMON] Checkpoints: {checkpoint_dir}")
    print(f"[EVAL DAEMON] Elo file: {elo_file}")
    if not args.once:
        print(f"[EVAL DAEMON] Polling every {args.interval}s (Ctrl+C to stop)")

    hb_path = checkpoint_dir.parent / 'eval_heartbeat.json'

    def write_heartbeat(evaluating_iter, completed, total):
        try:
            hb_path.write_text(json.dumps({
                'evaluating_iter': evaluating_iter,
                'completed': completed,
                'total': total,
                'timestamp': time.time(),
            }))
        except Exception:
            pass

    while True:
        elo, evaluated, stats, benchmark_stats, tournament_log = load_elo_with_metadata(elo_file)
        needed = find_tournament_needed(
            checkpoint_dir, evaluated,
            tournament_freq=args.tournament_freq,
            start_iter=start_iter
        )

        if needed:
            print(f"[EVAL DAEMON] Tournaments needed: {needed}")
            for idx, iteration in enumerate(needed):
                write_heartbeat(iteration, idx, len(needed))
                result = run_tournament(
                    iteration, checkpoint_dir, elo, net_cfg, eval_cfg, mcts_cfg,
                    device=device,
                    num_opponents=args.num_opponents,
                    games_per_opponent=args.games_per_opponent,
                    start_iter=start_iter,
                    write_heartbeat=write_heartbeat,
                )
                if result is not None:
                    all_participant_stats = result['stats']
                    tournament_log[str(iteration)] = result['tournament_log']
                    evaluated.add(iteration)
                    # Accumulate W/L/D for all participants (current + opponents)
                    for iter_key, s in all_participant_stats.items():
                        if iter_key in stats:
                            stats[iter_key]['wins'] = stats[iter_key].get('wins', 0) + s['wins']
                            stats[iter_key]['losses'] = stats[iter_key].get('losses', 0) + s['losses']
                            stats[iter_key]['draws'] = stats[iter_key].get('draws', 0) + s['draws']
                        else:
                            stats[iter_key] = s

                    # Run benchmark games (vs random + strategy bots)
                    bench = run_benchmarks(
                        iteration, checkpoint_dir, net_cfg, mcts_cfg,
                        device=device,
                    )
                    if bench:
                        benchmark_stats[str(iteration)] = bench

                    save_elo_with_metadata(elo_file, elo, evaluated, stats,
                                           benchmark_stats, tournament_log)

                    # Tensorboard
                    if log_dir:
                        try:
                            from torch.utils.tensorboard import SummaryWriter
                            writer = SummaryWriter(log_dir=str(log_dir))
                            writer.add_scalar('Evaluation/TournamentElo',
                                              elo.get_rating(f'iter_{iteration}'),
                                              iteration)
                            writer.close()
                        except Exception:
                            pass

            write_heartbeat(0, len(needed), len(needed))
        else:
            write_heartbeat(0, 0, 0)
            print(f"[EVAL DAEMON] All tournaments complete")

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
