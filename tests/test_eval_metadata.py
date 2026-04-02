"""Tests for Elo metadata save/load and backward compatibility.

Self-contained: inlines the EloRating class and save/load functions to avoid
importing the full mandala_rl package (which requires torch and mcts_cpp).
"""
import json
import os
import tempfile
from pathlib import Path


class EloRating:
    """Minimal EloRating for testing save/load (mirrors mandala_rl/evaluation/elo.py)."""
    def __init__(self, initial_rating=1500.0, k_factor=32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = {}


def load_elo_with_metadata(elo_file):
    """Mirrors scripts/eval_daemon.py load_elo_with_metadata."""
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
    """Mirrors scripts/eval_daemon.py save_elo_with_metadata."""
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


def test_roundtrip_with_tournament_log(tmp_path):
    """Tournament log survives save/load round-trip."""
    elo_file = tmp_path / 'elo_ratings.json'
    elo = EloRating()
    elo.ratings = {'iter_5': 1520.0, 'iter_10': 1480.0}
    evaluated = {5, 10}
    stats = {'5': {'wins': 3, 'losses': 2, 'draws': 0}}
    benchmark_stats = {'5': {'vs_random': {'wins': 10, 'losses': 0, 'draws': 0}}}
    tournament_log = {
        '10': {
            'timestamp': 1711800000.0,
            'elo_before': 1500.0,
            'elo_after': 1480.0,
            'elo_delta': -20.0,
            'num_opponents': 5,
            'games_played': 5,
            'mcts_sims': 400,
            'opponents': {
                'iter_5': {'wins': 2, 'losses': 3, 'draws': 0},
            },
            'results_summary': {'wins': 2, 'losses': 3, 'draws': 0},
        }
    }

    save_elo_with_metadata(elo_file, elo, evaluated, stats, benchmark_stats, tournament_log)
    loaded_elo, loaded_eval, loaded_stats, loaded_bench, loaded_tlog = load_elo_with_metadata(elo_file)

    assert loaded_elo.ratings == elo.ratings
    assert loaded_eval == evaluated
    assert loaded_stats == stats
    assert loaded_bench == benchmark_stats
    assert loaded_tlog == tournament_log
    assert loaded_tlog['10']['elo_delta'] == -20.0
    assert loaded_tlog['10']['opponents']['iter_5']['wins'] == 2


def test_backward_compat_missing_tournament_log(tmp_path):
    """Old elo files without tournament_log load gracefully."""
    elo_file = tmp_path / 'elo_ratings.json'
    data = {
        'initial_rating': 1500.0,
        'k_factor': 32.0,
        'ratings': {'iter_5': 1530.0},
        'tournament_evaluated': [5],
        'stats': {'5': {'wins': 5, 'losses': 2, 'draws': 1}},
        'benchmark_stats': {},
    }
    elo_file.write_text(json.dumps(data))

    elo, evaluated, stats, bench, tlog = load_elo_with_metadata(elo_file)

    assert elo.ratings == {'iter_5': 1530.0}
    assert evaluated == {5}
    assert tlog == {}


def test_atomic_write_produces_valid_json(tmp_path):
    """Atomic write creates valid JSON and no temp files remain."""
    elo_file = tmp_path / 'elo_ratings.json'
    elo = EloRating()
    elo.ratings = {'iter_1': 1500.0}

    save_elo_with_metadata(elo_file, elo, {1}, tournament_log={'1': {'elo_delta': 0.0}})

    with open(elo_file) as f:
        data = json.load(f)
    assert data['tournament_log']['1']['elo_delta'] == 0.0

    tmp_files = list(tmp_path.glob('*.tmp'))
    assert len(tmp_files) == 0


def test_nonexistent_file_returns_empty(tmp_path):
    """Loading a nonexistent file returns empty defaults including tournament_log."""
    elo_file = tmp_path / 'does_not_exist.json'
    elo, evaluated, stats, bench, tlog = load_elo_with_metadata(elo_file)

    assert elo.ratings == {}
    assert evaluated == set()
    assert stats == {}
    assert bench == {}
    assert tlog == {}
