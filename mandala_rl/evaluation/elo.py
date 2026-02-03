"""
Elo rating system for tracking bot strength over training.
"""
import json
from pathlib import Path
from typing import Dict, Optional


class EloRating:
    """
    Elo rating system for deterministic evaluation ladder.

    Tracks rating of different model checkpoints over time.
    """

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        """
        Args:
            initial_rating: Starting Elo rating
            k_factor: K-factor for rating updates
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}  # model_id -> rating

    def get_rating(self, model_id: str) -> float:
        """Get rating for a model."""
        if model_id not in self.ratings:
            self.ratings[model_id] = self.initial_rating
        return self.ratings[model_id]

    def update_ratings(
        self,
        model1_id: str,
        model2_id: str,
        score1: float,
        score2: float
    ):
        """
        Update ratings after a match.

        Args:
            model1_id: First model identifier
            model2_id: Second model identifier
            score1: Score for model 1 (1.0 = win, 0.5 = draw, 0.0 = loss)
            score2: Score for model 2
        """
        rating1 = self.get_rating(model1_id)
        rating2 = self.get_rating(model2_id)

        # Expected scores
        expected1 = 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))
        expected2 = 1.0 / (1.0 + 10 ** ((rating1 - rating2) / 400.0))

        # Update ratings
        new_rating1 = rating1 + self.k_factor * (score1 - expected1)
        new_rating2 = rating2 + self.k_factor * (score2 - expected2)

        self.ratings[model1_id] = new_rating1
        self.ratings[model2_id] = new_rating2

    def record_match(
        self,
        model1_id: str,
        model2_id: str,
        winner: Optional[str]
    ):
        """
        Record a match result and update ratings.

        Args:
            model1_id: First model identifier
            model2_id: Second model identifier
            winner: Winner model_id, or None for draw
        """
        if winner == model1_id:
            score1, score2 = 1.0, 0.0
        elif winner == model2_id:
            score1, score2 = 0.0, 1.0
        elif winner is None:
            score1, score2 = 0.5, 0.5
        else:
            raise ValueError(f"Invalid winner: {winner}")

        self.update_ratings(model1_id, model2_id, score1, score2)

    def get_leaderboard(self) -> list:
        """
        Get sorted leaderboard of models.

        Returns:
            List of (model_id, rating) tuples sorted by rating descending
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def save(self, filepath: Path):
        """Save ratings to JSON file."""
        data = {
            'initial_rating': self.initial_rating,
            'k_factor': self.k_factor,
            'ratings': self.ratings
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path):
        """Load ratings from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.initial_rating = data['initial_rating']
        self.k_factor = data['k_factor']
        self.ratings = data['ratings']

    def __str__(self):
        """String representation showing leaderboard."""
        lines = ["Elo Leaderboard:", "=" * 50]
        for i, (model_id, rating) in enumerate(self.get_leaderboard(), 1):
            lines.append(f"{i}. {model_id}: {rating:.1f}")
        return "\n".join(lines)
