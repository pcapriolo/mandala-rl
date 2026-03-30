"""
Dominion game state representation.

Multi-step turn structure: each get_next_state() call processes one atomic decision.
A turn consists of multiple sequential actions (play actions, play treasures, buy cards).
The `pending` field tracks card-effect resolution requiring player input.
"""
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .cards import (
    NUM_CARD_TYPES, CARD_BY_ID, GARDENS, PROVINCE, CURSE,
    COPPER, SILVER, GOLD, ESTATE, DUCHY,
)

# Action space constants
NUM_ACTIONS = 131
# Action IDs:
#   0:       END_ACTIONS
#   1:       END_BUYS
#   2:       DONE_SELECTING
#   3-33:    PLAY[card_type]  (play card from hand — 31 card types)
#   34-64:   BUY[card_type]   (buy card from supply — 31 card types)
#   65-95:   SELECT[card_type] (select card for effect resolution — 31 card types)
#   96-126:  GAIN[card_type]   (gain card from supply for effect — 31 card types)
#   127:     REVEAL_MOAT
#   128:     DECLINE_REACTION
#   129:     ACCEPT (e.g., Vassal play action, Library keep)
#   130:     DECLINE (e.g., Vassal skip, Library skip action)

END_ACTIONS = 0
END_BUYS = 1
DONE_SELECTING = 2
PLAY_OFFSET = 3        # PLAY[card_type] = 3 + card_type_id
BUY_OFFSET = 34        # BUY[card_type] = 34 + card_type_id
SELECT_OFFSET = 65     # SELECT[card_type] = 65 + card_type_id
GAIN_OFFSET = 96       # GAIN[card_type] = 96 + card_type_id
REVEAL_MOAT = 127
DECLINE_REACTION = 128
ACCEPT = 129
DECLINE = 130


@dataclass
class PendingDecision:
    """Tracks card effect resolution state."""
    card_id: int              # Which card caused this pending state
    decision_type: str        # 'select_hand', 'select_discard', 'gain', 'react',
                              # 'vassal_play', 'library_draw', 'sentry_card',
                              # 'bandit_trash', 'artisan_topdeck', 'mine_gain',
                              # 'harbinger_topdeck', 'remodel_gain',
                              # 'bureaucrat_topdeck'
    min_select: int = 0       # Minimum cards to select
    max_select: int = 0       # Maximum cards to select
    selected_count: int = 0   # How many selected so far
    # For attacks: who needs to react/respond
    target_player: int = -1
    # For Sentry: revealed cards
    revealed: Optional[List[int]] = None
    # For Bandit: revealed cards from opponent
    bandit_revealed: Optional[List[int]] = None
    # For Throne Room: card to play again, remaining plays
    throne_card: int = -1
    throne_remaining: int = 0
    # For Remodel/Mine: the card that was trashed (to compute gain cost limit)
    trashed_card_cost: int = 0


class PlayerState:
    """One player's state in Dominion."""

    def __init__(self):
        self.deck: List[int] = []       # Card type IDs, top = end
        self.hand: List[int] = []
        self.discard: List[int] = []
        self.in_play: List[int] = []

    def copy(self):
        p = PlayerState()
        p.deck = self.deck.copy()
        p.hand = self.hand.copy()
        p.discard = self.discard.copy()
        p.in_play = self.in_play.copy()
        return p

    def draw(self, n: int) -> int:
        """Draw n cards from deck to hand. Reshuffles discard if needed.
        Returns number of cards actually drawn."""
        drawn = 0
        for _ in range(n):
            if not self.deck:
                if not self.discard:
                    break
                self.deck = self.discard.copy()
                self.discard.clear()
                random.shuffle(self.deck)
            self.hand.append(self.deck.pop())
            drawn += 1
        return drawn

    def draw_to_list(self, n: int) -> List[int]:
        """Draw n cards and return them as a list (not added to hand)."""
        cards = []
        for _ in range(n):
            if not self.deck:
                if not self.discard:
                    break
                self.deck = self.discard.copy()
                self.discard.clear()
                random.shuffle(self.deck)
            cards.append(self.deck.pop())
        return cards

    def total_cards(self) -> int:
        return len(self.deck) + len(self.hand) + len(self.discard) + len(self.in_play)

    def count_vp(self, kingdom_cards: List[int]) -> int:
        """Count total VP including Gardens."""
        all_cards = self.deck + self.hand + self.discard + self.in_play
        vp = 0
        for cid in all_cards:
            card = CARD_BY_ID[cid]
            if card.id == GARDENS.id:
                vp += len(all_cards) // 10
            else:
                vp += card.vp
        return vp

    def has_card_in_hand(self, card_id: int) -> bool:
        return card_id in self.hand

    def count_in_hand(self, card_id: int) -> int:
        return self.hand.count(card_id)

    def remove_from_hand(self, card_id: int) -> bool:
        """Remove one copy of card_id from hand. Returns True if found."""
        try:
            self.hand.remove(card_id)
            return True
        except ValueError:
            return False


class DominionState:
    """Full game state for Dominion."""

    def __init__(self):
        self.current_player: int = 0      # Whose decision it is now
        self.active_player: int = 0       # Whose turn it actually is
        self.players: List[PlayerState] = [PlayerState(), PlayerState()]
        self.supply: Dict[int, int] = {}
        self.trash: List[int] = []
        self.phase: str = 'action'        # 'action', 'buy', 'react', 'select'
        self.actions_remaining: int = 1
        self.buys_remaining: int = 1
        self.coins: int = 0
        self.pending: Optional[PendingDecision] = None
        self.turn_number: int = 1
        self.game_over: bool = False
        self.kingdom_cards: List[int] = []
        self.merchant_silver_bonus: int = 0  # Accumulated Merchant bonuses for this turn
        self.merchant_silver_triggered: bool = False  # Has a Silver been played this turn?
        # Throne Room tracking: card to replay and remaining plays
        self.throne_card: int = -1        # Card being played again by Throne Room
        self.throne_remaining: int = 0    # How many more times to play it
        # For attack resolution tracking
        self._save_phase: str = ''        # Saved phase during attack resolution

    def copy(self) -> 'DominionState':
        s = DominionState()
        s.current_player = self.current_player
        s.active_player = self.active_player
        s.players = [p.copy() for p in self.players]
        s.supply = self.supply.copy()
        s.trash = self.trash.copy()
        s.phase = self.phase
        s.actions_remaining = self.actions_remaining
        s.buys_remaining = self.buys_remaining
        s.coins = self.coins
        if self.pending:
            p = self.pending
            s.pending = PendingDecision(
                card_id=p.card_id, decision_type=p.decision_type,
                min_select=p.min_select, max_select=p.max_select,
                selected_count=p.selected_count, target_player=p.target_player,
                revealed=p.revealed.copy() if p.revealed else None,
                bandit_revealed=p.bandit_revealed.copy() if p.bandit_revealed else None,
                throne_card=p.throne_card, throne_remaining=p.throne_remaining,
                trashed_card_cost=p.trashed_card_cost,
            )
        else:
            s.pending = None
        s.turn_number = self.turn_number
        s.game_over = self.game_over
        s.kingdom_cards = self.kingdom_cards.copy()
        s.merchant_silver_bonus = self.merchant_silver_bonus
        s.merchant_silver_triggered = self.merchant_silver_triggered
        s.throne_card = self.throne_card
        s.throne_remaining = self.throne_remaining
        s._save_phase = self._save_phase
        return s

    def to_tensor(self) -> np.ndarray:
        """Convert state to (280, 8, 8) float32 tensor.

        State is assumed to be in canonical form (current_player = 0).
        Note: Channels 156-279 match the C++ tensor layout but some
        (e.g., buy history 218-279) are zeroed in Python since the
        Python state doesn't track all behavioral data.

        Channels:
          0-30:    My full deck composition (count per card type / 12)
          31-61:   My hand (count per card type / 10)
          62-92:   My in-play area (count per card type / 5)
          93-123:  Supply remaining (count per pile / 12)
          124:     Opponent hand size / 10
          125:     Opponent total cards / 50
          126:     My coins this turn / 12
          127:     My actions remaining / 10
          128:     My buys remaining / 5
          129:     Phase encoding (ordinal: action=0.0, buy=0.33, select=0.67, react=1.0)
          130:     Turn number / 40
          131:     Provinces remaining / 8
          132:     Empty supply piles / 3
          133-137: Pending decision context
          138:     My VP total / 40
          139:     Opponent VP estimate / 40
          140:     Game progress / 1.0
          141:     My total cards / 50
          142:     Pending selected count / max_select
          143:     Action density (action-givers / total action cards)
          144:     Draw density (draw cards / total deck)
          145:     Payload (avg coins per card / 3)
          146:     Green bloat (victory+curse cards / total deck)
          147:     Terminal collision risk (surplus terminals / 5)
          148:     Expected hand value (coins per 5-card hand / 12)
          149:     Opponent expected hand value
          150:     Attack pressure (my attack cards / 3)
          151:     Treasure coins in hand / 12
          152:     Best +cards from playable action in hand / 5
          153:     Best +actions from playable action in hand / 5
          154:     Best +coins from playable action in hand / 5
          155:     Number of action cards in hand / 5
        """
        tensor = np.zeros((280, 8, 8), dtype=np.float32)
        me = self.players[0]
        opp = self.players[1]

        # Ch 0-30: My full deck composition
        all_my = me.deck + me.hand + me.discard + me.in_play
        for cid in all_my:
            tensor[cid] += 1.0 / 12.0

        # Ch 31-61: My hand
        for cid in me.hand:
            tensor[31 + cid] += 1.0 / 10.0

        # Ch 62-92: My in-play
        for cid in me.in_play:
            tensor[62 + cid] += 1.0 / 5.0

        # Ch 93-123: Supply remaining
        for cid, count in self.supply.items():
            if cid < NUM_CARD_TYPES:
                tensor[93 + cid] = count / 12.0

        # Ch 124: Opponent hand size
        tensor[124] = len(opp.hand) / 10.0

        # Ch 125: Opponent total cards
        tensor[125] = opp.total_cards() / 50.0

        # Ch 126: My coins
        tensor[126] = self.coins / 12.0

        # Ch 127: Actions remaining
        tensor[127] = self.actions_remaining / 10.0

        # Ch 128: Buys remaining
        tensor[128] = self.buys_remaining / 5.0

        # Ch 129: Phase encoding
        phase_map = {'action': 0.0, 'buy': 0.33, 'select': 0.67, 'react': 1.0}
        tensor[129] = phase_map.get(self.phase, 0.0)

        # Ch 130: Turn number
        tensor[130] = min(self.turn_number / 40.0, 1.0)

        # Ch 131: Provinces remaining
        tensor[131] = self.supply.get(PROVINCE.id, 0) / 8.0

        # Ch 132: Empty supply piles
        empty = sum(1 for cid, cnt in self.supply.items() if cnt == 0)
        tensor[132] = empty / 3.0

        # Ch 133-137: Pending decision context
        if self.pending:
            p = self.pending
            tensor[133] = p.card_id / float(NUM_CARD_TYPES)
            type_map = {
                'select_hand': 0.1, 'select_discard': 0.2, 'gain': 0.3,
                'react': 0.4, 'vassal_play': 0.5, 'library_draw': 0.6,
                'sentry_card': 0.7, 'bandit_trash': 0.8, 'artisan_topdeck': 0.9,
                'mine_gain': 0.15, 'harbinger_topdeck': 0.25, 'remodel_gain': 0.35,
                'bureaucrat_topdeck': 0.45,
            }
            tensor[134] = type_map.get(p.decision_type, 0.0)
            tensor[135] = p.max_select / 10.0 if p.max_select > 0 else 0.0
            tensor[136] = p.selected_count / 10.0
            tensor[137] = 1.0 if p.target_player >= 0 else 0.0

        # Ch 138: My VP
        tensor[138] = np.clip(me.count_vp(self.kingdom_cards) / 40.0, -0.5, 1.0)

        # Ch 139: Opponent VP estimate
        tensor[139] = np.clip(opp.count_vp(self.kingdom_cards) / 40.0, -0.5, 1.0)

        # Ch 140: Game progress (based on Provinces gone)
        prov_gone = 8 - self.supply.get(PROVINCE.id, 0)
        tensor[140] = prov_gone / 8.0

        # Ch 141: My total cards
        tensor[141] = me.total_cards() / 50.0

        # Ch 142: Pending selected count / max
        if self.pending and self.pending.max_select > 0:
            tensor[142] = self.pending.selected_count / self.pending.max_select

        # Ch 143-150: Derived synergy features
        all_my = me.deck + me.hand + me.discard + me.in_play
        my_total = len(all_my)
        if my_total > 0:
            action_givers = 0
            terminals = 0
            draw_cards = 0
            treasure_coins = 0
            victory_cards = 0
            attack_cards = 0

            for cid in all_my:
                card = CARD_BY_ID[cid]
                if 'Victory' in card.types or 'Curse' in card.types:
                    victory_cards += 1
                if 'Treasure' in card.types:
                    treasure_coins += card.coins
                if 'Action' in card.types:
                    if card.plus_actions > 0:
                        action_givers += 1
                    else:
                        terminals += 1
                    if card.plus_cards > 0:
                        draw_cards += 1
                if 'Attack' in card.types:
                    attack_cards += 1

            total_actions = action_givers + terminals
            if total_actions > 0:
                tensor[143] = action_givers / total_actions
            tensor[144] = draw_cards / my_total
            tensor[145] = treasure_coins / my_total / 3.0
            tensor[146] = victory_cards / my_total
            surplus = max(0, terminals - action_givers - 1)
            tensor[147] = min(1.0, surplus / 5.0)
            expected_hand = treasure_coins * 5.0 / my_total
            tensor[148] = min(1.0, expected_hand / 12.0)
            tensor[150] = min(1.0, attack_cards / 3.0)

        # Ch 149: Opponent expected hand value
        all_opp = opp.deck + opp.hand + opp.discard + opp.in_play
        opp_total = len(all_opp)
        if opp_total > 0:
            opp_treasure_coins = sum(CARD_BY_ID[c].coins for c in all_opp if 'Treasure' in CARD_BY_ID[c].types)
            opp_expected = opp_treasure_coins * 5.0 / opp_total
            tensor[149] = min(1.0, opp_expected / 12.0)

        # Ch 151-155: Action card awareness channels
        # Ch 151: Treasure coins in hand
        hand_treasure_coins = sum(CARD_BY_ID[c].coins for c in me.hand if 'Treasure' in CARD_BY_ID[c].types)
        tensor[151] = min(1.0, hand_treasure_coins / 12.0)

        # Ch 152-154: Best action bonuses (conditional on actions_remaining > 0)
        best_cards = 0
        best_actions = 0
        best_coins = 0
        action_count_in_hand = 0
        for cid in me.hand:
            card = CARD_BY_ID[cid]
            if 'Action' in card.types:
                action_count_in_hand += 1
                if self.actions_remaining > 0:
                    best_cards = max(best_cards, card.plus_cards)
                    best_actions = max(best_actions, card.plus_actions)
                    best_coins = max(best_coins, card.plus_coins)

        tensor[152] = min(1.0, best_cards / 5.0)
        tensor[153] = min(1.0, best_actions / 5.0)
        tensor[154] = min(1.0, best_coins / 5.0)

        # Ch 155: Number of action cards in hand
        tensor[155] = min(1.0, action_count_in_hand / 5.0)

        return tensor

    def get_canonical_form(self) -> 'DominionState':
        """Return state from current_player's perspective (current_player becomes 0)."""
        if self.current_player == 0:
            return self.copy()
        s = self.copy()
        s.players = [s.players[1], s.players[0]]
        s.active_player = 1 - s.active_player
        s.current_player = 0
        if s.pending and s.pending.target_player >= 0:
            s.pending.target_player = 1 - s.pending.target_player
        return s

    def __hash__(self):
        hand_tuples = tuple(
            tuple(sorted(p.hand)) for p in self.players
        )
        return hash((
            hand_tuples,
            self.current_player,
            self.phase,
            self.actions_remaining,
            self.buys_remaining,
            self.coins,
            self.turn_number,
            tuple(sorted(self.supply.items())),
        ))
