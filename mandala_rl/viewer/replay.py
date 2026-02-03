"""Game replay serialization and storage."""
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class GameReplay:
    """Stores a complete game for replay."""

    def __init__(self, game_id: str = None):
        self.game_id = game_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.moves: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.final_score = None
        self.winner = None

    def add_move(self, move_num: int, player: int, action_id: int,
                 action_desc: str, state_summary: Dict[str, Any]):
        """Record a move."""
        self.moves.append({
            'move_num': move_num,
            'player': player,
            'action_id': action_id,
            'action_desc': action_desc,
            'state': state_summary
        })

    def set_result(self, score0: int, score1: int, winner: int):
        """Set final game result."""
        self.final_score = (score0, score1)
        self.winner = winner

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'game_id': self.game_id,
            'metadata': self.metadata,
            'moves': self.moves,
            'final_score': self.final_score,
            'winner': self.winner
        }

    def save(self, directory: Path):
        """Save replay to JSON file."""
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"game_{self.game_id}.json"
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'GameReplay':
        """Load replay from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        replay = cls(game_id=data['game_id'])
        replay.moves = data['moves']
        replay.metadata = data['metadata']
        replay.final_score = tuple(data['final_score']) if data['final_score'] else None
        replay.winner = data['winner']
        return replay


def state_to_summary(state) -> Dict[str, Any]:
    """Convert game state to JSON-serializable summary."""
    return {
        'current_player': state.current_player,
        'hands': [
            [c.color for c in state.hands[0]],
            [c.color for c in state.hands[1]]
        ],
        'rivers': [
            [c.color for c in state.rivers[0]],
            [c.color for c in state.rivers[1]]
        ],
        'cups': [len(state.cups[0]), len(state.cups[1])],
        'mountains': [
            [c.color for c in state.mountains[0]],
            [c.color for c in state.mountains[1]]
        ],
        'fields': [
            [[c.color for c in state.fields[0][0]], [c.color for c in state.fields[0][1]]],
            [[c.color for c in state.fields[1][0]], [c.color for c in state.fields[1][1]]]
        ],
        'deck_size': len(state.deck),
        'discard_size': len(state.discard)
    }


def action_id_to_string(action_id: int) -> str:
    """Convert action ID to readable string."""
    colors = ['White', 'Green', 'Purple', 'Yellow', 'Red', 'Orange']

    if action_id < 12:
        color = action_id // 2
        mandala = action_id % 2
        return f"BUILD_MOUNTAIN: {colors[color]} → Mandala {mandala}"
    elif action_id < 24:
        action_id -= 12
        color = action_id // 2
        mandala = action_id % 2
        return f"GROW_FIELD: {colors[color]} → Field {mandala}"
    else:
        color = action_id - 24
        return f"DISCARD: {colors[color]}"
