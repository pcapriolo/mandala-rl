"""
Dominion game engine implementing complete Base Set 2nd Edition rules.

Action space (131 actions):
  0:       END_ACTIONS
  1:       END_BUYS
  2:       DONE_SELECTING
  3-33:    PLAY[card_type]
  34-64:   BUY[card_type]
  65-95:   SELECT[card_type]
  96-126:  GAIN[card_type]
  127:     REVEAL_MOAT
  128:     DECLINE_REACTION
  129:     ACCEPT
  130:     DECLINE
"""
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .cards import (
    NUM_CARD_TYPES, ALL_CARDS, CARD_BY_ID, KINGDOM_CARDS,
    get_initial_supply, card_name,
    COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, CURSE,
    CELLAR, MOAT, CHAPEL, HARBINGER, MERCHANT, VASSAL, VILLAGE, WORKSHOP,
    BUREAUCRAT, GARDENS, MILITIA, MONEYLENDER, POACHER, REMODEL, SMITHY,
    THRONE_ROOM, ARTISAN, BANDIT, COUNCIL_ROOM, LIBRARY, MARKET, MINE,
    SENTRY, WITCH,
)
from .state import (
    DominionState, PlayerState, PendingDecision,
    NUM_ACTIONS, END_ACTIONS, END_BUYS, DONE_SELECTING,
    PLAY_OFFSET, BUY_OFFSET, SELECT_OFFSET, GAIN_OFFSET,
    REVEAL_MOAT, DECLINE_REACTION, ACCEPT, DECLINE,
)


class DominionGame:
    """Complete Dominion Base Set 2nd Edition game engine."""

    def __init__(self, seed: Optional[int] = None, kingdom_card_ids: Optional[List[int]] = None):
        """
        Args:
            seed: Random seed for reproducibility.
            kingdom_card_ids: Specific 10 kingdom cards to use. If None, randomly selected.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._kingdom_card_ids = kingdom_card_ids

    def get_initial_state(self) -> DominionState:
        """Create a fresh game state with shuffled decks."""
        state = DominionState()

        # Select kingdom cards
        if self._kingdom_card_ids is not None:
            state.kingdom_cards = list(self._kingdom_card_ids)
        else:
            state.kingdom_cards = sorted(random.sample(
                [c.id for c in KINGDOM_CARDS], 10
            ))

        # Setup supply
        state.supply = get_initial_supply(state.kingdom_cards)

        # Setup players: 7 Copper + 3 Estate each
        for p in range(2):
            deck = [COPPER.id] * 7 + [ESTATE.id] * 3
            random.shuffle(deck)
            state.players[p].deck = deck
            state.players[p].draw(5)

        state.current_player = 0
        state.active_player = 0
        state.phase = 'action'
        state.actions_remaining = 1
        state.buys_remaining = 1
        state.coins = 0
        state.turn_number = 1
        return state

    def get_valid_moves(self, state: DominionState) -> np.ndarray:
        """Return binary mask of valid actions."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if state.game_over:
            return mask

        player = state.players[state.current_player]

        # Handle pending decisions first
        if state.pending:
            return self._get_valid_pending(state, mask)

        if state.phase == 'action':
            return self._get_valid_action_phase(state, mask)
        elif state.phase == 'buy':
            return self._get_valid_buy_phase(state, mask)

        return mask

    def _get_valid_pending(self, state: DominionState, mask: np.ndarray) -> np.ndarray:
        """Valid moves during pending decision resolution."""
        p = state.pending
        player = state.players[state.current_player]

        if p.decision_type == 'react':
            # Can reveal Moat or decline
            if MOAT.id in player.hand:
                mask[REVEAL_MOAT] = 1.0
            mask[DECLINE_REACTION] = 1.0

        elif p.decision_type == 'select_hand':
            # Select cards from hand (Cellar discard, Chapel trash, Militia discard)
            for cid in set(player.hand):
                mask[SELECT_OFFSET + cid] = 1.0
            # Can finish selecting if minimum met
            if p.selected_count >= p.min_select:
                mask[DONE_SELECTING] = 1.0

        elif p.decision_type == 'gain':
            # Gain a card from supply (Workshop: cost <= 4)
            cost_limit = p.trashed_card_cost
            for cid, count in state.supply.items():
                if count > 0:
                    card = CARD_BY_ID[cid]
                    if card.cost <= cost_limit:
                        mask[GAIN_OFFSET + cid] = 1.0

        elif p.decision_type == 'remodel_gain':
            # Gain a card costing up to trashed_card_cost + 2
            cost_limit = p.trashed_card_cost + 2
            for cid, count in state.supply.items():
                if count > 0:
                    card = CARD_BY_ID[cid]
                    if card.cost <= cost_limit:
                        mask[GAIN_OFFSET + cid] = 1.0

        elif p.decision_type == 'mine_gain':
            # Gain a Treasure costing up to trashed_card_cost + 3
            cost_limit = p.trashed_card_cost + 3
            for cid, count in state.supply.items():
                if count > 0:
                    card = CARD_BY_ID[cid]
                    if 'Treasure' in card.types and card.cost <= cost_limit:
                        mask[GAIN_OFFSET + cid] = 1.0

        elif p.decision_type == 'artisan_topdeck':
            # Put a card from hand on top of deck
            for cid in set(player.hand):
                mask[SELECT_OFFSET + cid] = 1.0

        elif p.decision_type == 'harbinger_topdeck':
            # May put a card from discard on top of deck, or skip
            for cid in set(player.discard):
                mask[SELECT_OFFSET + cid] = 1.0
            mask[DONE_SELECTING] = 1.0  # Can skip

        elif p.decision_type == 'vassal_play':
            # Accept = play the revealed action, Decline = skip
            mask[ACCEPT] = 1.0
            mask[DECLINE] = 1.0

        elif p.decision_type == 'library_draw':
            # Accept = keep the drawn card, Decline = set aside (skip action card)
            mask[ACCEPT] = 1.0
            mask[DECLINE] = 1.0

        elif p.decision_type == 'sentry_card':
            # For each revealed card: trash, discard, or keep (put back on deck)
            # We use: SELECT = trash, ACCEPT = discard, DECLINE = keep on deck
            mask[SELECT_OFFSET + p.revealed[0]] = 1.0  # Trash it
            mask[ACCEPT] = 1.0   # Discard it
            mask[DECLINE] = 1.0  # Put back on deck

        elif p.decision_type == 'bandit_trash':
            # Opponent must trash one of the revealed non-Copper treasures
            for cid in p.bandit_revealed:
                if cid != COPPER.id and 'Treasure' in CARD_BY_ID[cid].types:
                    mask[SELECT_OFFSET + cid] = 1.0
            # If no valid treasures to trash (shouldn't happen if we got here), done
            if mask.sum() == 0:
                mask[DONE_SELECTING] = 1.0

        elif p.decision_type == 'bureaucrat_topdeck':
            # Opponent must put a Victory card from hand on deck
            has_victory = False
            for cid in set(player.hand):
                if 'Victory' in CARD_BY_ID[cid].types:
                    mask[SELECT_OFFSET + cid] = 1.0
                    has_victory = True
            if not has_victory:
                # Reveal hand (auto-resolve) — treated as DONE
                mask[DONE_SELECTING] = 1.0

        elif p.decision_type == 'select_trash_hand':
            # Select a card from hand to trash (Remodel, Mine, Moneylender)
            if p.card_id == MONEYLENDER.id:
                # Must trash a Copper
                if COPPER.id in player.hand:
                    mask[SELECT_OFFSET + COPPER.id] = 1.0
                else:
                    mask[DONE_SELECTING] = 1.0
            elif p.card_id == MINE.id:
                # Trash a Treasure from hand
                for cid in set(player.hand):
                    if 'Treasure' in CARD_BY_ID[cid].types:
                        mask[SELECT_OFFSET + cid] = 1.0
                if mask.sum() == 0:
                    mask[DONE_SELECTING] = 1.0
            else:
                # Remodel: trash any card from hand
                for cid in set(player.hand):
                    mask[SELECT_OFFSET + cid] = 1.0
                if not player.hand:
                    mask[DONE_SELECTING] = 1.0

        elif p.decision_type == 'artisan_gain':
            # Gain a card costing up to 5 to hand
            for cid, count in state.supply.items():
                if count > 0:
                    card = CARD_BY_ID[cid]
                    if card.cost <= 5:
                        mask[GAIN_OFFSET + cid] = 1.0

        elif p.decision_type == 'throne_select':
            # Select an Action card from hand to play twice
            for cid in set(player.hand):
                if 'Action' in CARD_BY_ID[cid].types:
                    mask[SELECT_OFFSET + cid] = 1.0
            if mask.sum() == 0:
                # No actions in hand — Throne Room fizzles
                mask[DONE_SELECTING] = 1.0

        return mask

    def _get_valid_action_phase(self, state: DominionState, mask: np.ndarray) -> np.ndarray:
        """Valid moves during action phase."""
        player = state.players[state.current_player]

        # Can always end action phase
        mask[END_ACTIONS] = 1.0

        if state.actions_remaining > 0:
            # Can play Action cards from hand
            for cid in set(player.hand):
                card = CARD_BY_ID[cid]
                if 'Action' in card.types:
                    mask[PLAY_OFFSET + cid] = 1.0

        return mask

    def _get_valid_buy_phase(self, state: DominionState, mask: np.ndarray) -> np.ndarray:
        """Valid moves during buy phase."""
        player = state.players[state.current_player]

        # Can always end buy phase
        mask[END_BUYS] = 1.0

        # Can play Treasure cards from hand (free, no action cost)
        for cid in set(player.hand):
            card = CARD_BY_ID[cid]
            if 'Treasure' in card.types:
                mask[PLAY_OFFSET + cid] = 1.0

        # Can buy cards if buys remaining
        if state.buys_remaining > 0:
            for cid, count in state.supply.items():
                if count > 0:
                    card = CARD_BY_ID[cid]
                    if card.cost <= state.coins:
                        mask[BUY_OFFSET + cid] = 1.0

        return mask

    def get_next_state(self, state: DominionState, action: int) -> DominionState:
        """Apply one atomic action and return new state."""
        new_state = state.copy()

        if new_state.pending:
            self._resolve_pending(new_state, action)
        elif action == END_ACTIONS:
            new_state.phase = 'buy'
        elif action == END_BUYS:
            self._do_cleanup(new_state)
        elif action >= PLAY_OFFSET and action < BUY_OFFSET:
            card_id = action - PLAY_OFFSET
            self._play_card(new_state, card_id)
        elif action >= BUY_OFFSET and action < SELECT_OFFSET:
            card_id = action - BUY_OFFSET
            self._buy_card(new_state, card_id)

        # After any resolution, check if Throne Room replay is due
        if (new_state.throne_remaining > 0 and new_state.pending is None
                and new_state.phase == 'action'):
            self._do_throne_replay(new_state)

        return new_state

    def _play_card(self, state: DominionState, card_id: int):
        """Play a card from hand."""
        player = state.players[state.current_player]
        card = CARD_BY_ID[card_id]

        # Remove from hand, add to in_play
        player.remove_from_hand(card_id)
        player.in_play.append(card_id)

        if 'Treasure' in card.types:
            state.coins += card.coins
            # Merchant bonus: +1 coin first time Silver played this turn
            if card_id == SILVER.id and not state.merchant_silver_triggered:
                state.coins += state.merchant_silver_bonus
                state.merchant_silver_triggered = True
            return

        # Action card
        if state.phase == 'action':
            state.actions_remaining -= 1

        # Apply simple bonuses
        state.actions_remaining += card.plus_actions
        state.buys_remaining += card.plus_buys
        state.coins += card.plus_coins
        if card.plus_cards > 0:
            player.draw(card.plus_cards)

        # Card-specific effects
        if card.has_effect:
            self._handle_effect(state, card_id)

    def _handle_effect(self, state: DominionState, card_id: int):
        """Handle card-specific effects after simple bonuses applied."""
        player = state.players[state.active_player]
        opp = state.players[1 - state.active_player]

        if card_id == CELLAR.id:
            # Discard any number of cards, then draw that many
            if player.hand:
                state.pending = PendingDecision(
                    card_id=CELLAR.id, decision_type='select_hand',
                    min_select=0, max_select=len(player.hand),
                )
                state.phase = 'select'

        elif card_id == CHAPEL.id:
            # Trash up to 4 cards from hand
            if player.hand:
                state.pending = PendingDecision(
                    card_id=CHAPEL.id, decision_type='select_hand',
                    min_select=0, max_select=min(4, len(player.hand)),
                )
                state.phase = 'select'

        elif card_id == HARBINGER.id:
            # +1 Card, +1 Action already applied. May put a card from discard on deck.
            if player.discard:
                state.pending = PendingDecision(
                    card_id=HARBINGER.id, decision_type='harbinger_topdeck',
                )
                state.phase = 'select'

        elif card_id == MERCHANT.id:
            # +1 Card, +1 Action already applied.
            # Track: next Silver played gives +1 coin
            state.merchant_silver_bonus += 1

        elif card_id == VASSAL.id:
            # +2 coins already applied. Discard top card; if Action, may play it.
            revealed = player.draw_to_list(1)
            if revealed:
                rcid = revealed[0]
                player.discard.append(rcid)
                if 'Action' in CARD_BY_ID[rcid].types:
                    state.pending = PendingDecision(
                        card_id=VASSAL.id, decision_type='vassal_play',
                        revealed=[rcid],
                    )
                    state.phase = 'select'

        elif card_id == WORKSHOP.id:
            # Gain a card costing up to 4
            state.pending = PendingDecision(
                card_id=WORKSHOP.id, decision_type='gain',
                trashed_card_cost=4,
            )
            state.phase = 'select'

        elif card_id == BUREAUCRAT.id:
            # Gain a Silver on top of deck
            if state.supply.get(SILVER.id, 0) > 0:
                state.supply[SILVER.id] -= 1
                player.deck.append(SILVER.id)
            # Attack: opponent puts Victory from hand on deck
            self._start_attack(state, BUREAUCRAT.id, 'bureaucrat_topdeck')

        elif card_id == MILITIA.id:
            # +2 coins already applied. Attack: opponents discard down to 3.
            self._start_attack(state, MILITIA.id, 'select_hand',
                               min_select=0, max_select=0)

        elif card_id == MONEYLENDER.id:
            # Trash a Copper from hand; if you do, +3 coins
            if COPPER.id in player.hand:
                state.pending = PendingDecision(
                    card_id=MONEYLENDER.id, decision_type='select_trash_hand',
                )
                state.phase = 'select'

        elif card_id == POACHER.id:
            # +1 Card, +1 Action, +1 Coin already applied.
            # Discard a card per empty supply pile
            empty_piles = sum(1 for cid, cnt in state.supply.items() if cnt == 0)
            if empty_piles > 0 and player.hand:
                discard_count = min(empty_piles, len(player.hand))
                state.pending = PendingDecision(
                    card_id=POACHER.id, decision_type='select_hand',
                    min_select=discard_count, max_select=discard_count,
                )
                state.phase = 'select'

        elif card_id == REMODEL.id:
            # Trash a card from hand; gain one costing up to 2 more
            if player.hand:
                state.pending = PendingDecision(
                    card_id=REMODEL.id, decision_type='select_trash_hand',
                )
                state.phase = 'select'

        elif card_id == THRONE_ROOM.id:
            # Choose an Action from hand, play it twice
            has_action = any('Action' in CARD_BY_ID[cid].types for cid in player.hand)
            if has_action:
                state.pending = PendingDecision(
                    card_id=THRONE_ROOM.id, decision_type='throne_select',
                )
                state.phase = 'select'
            # If no actions, Throne Room fizzles (no pending needed)

        elif card_id == ARTISAN.id:
            # Gain a card to hand costing up to 5; put a card from hand on deck
            state.pending = PendingDecision(
                card_id=ARTISAN.id, decision_type='artisan_gain',
            )
            state.phase = 'select'

        elif card_id == BANDIT.id:
            # Gain a Gold
            if state.supply.get(GOLD.id, 0) > 0:
                state.supply[GOLD.id] -= 1
                player.discard.append(GOLD.id)
            # Attack: opponent reveals top 2, trashes a non-Copper treasure
            self._start_attack(state, BANDIT.id, 'bandit_trash')

        elif card_id == COUNCIL_ROOM.id:
            # +4 Cards, +1 Buy already applied. Each other player draws a card.
            opp.draw(1)

        elif card_id == LIBRARY.id:
            # Draw until 7 cards in hand, may skip Actions
            self._library_step(state)

        elif card_id == MINE.id:
            # Trash a Treasure from hand; gain a Treasure to hand costing up to 3 more
            has_treasure = any('Treasure' in CARD_BY_ID[cid].types for cid in player.hand)
            if has_treasure:
                state.pending = PendingDecision(
                    card_id=MINE.id, decision_type='select_trash_hand',
                )
                state.phase = 'select'

        elif card_id == SENTRY.id:
            # +1 Card, +1 Action already applied. Look at top 2 of deck.
            revealed = player.draw_to_list(2)
            if revealed:
                state.pending = PendingDecision(
                    card_id=SENTRY.id, decision_type='sentry_card',
                    revealed=revealed,
                )
                state.phase = 'select'

        elif card_id == WITCH.id:
            # +2 Cards already applied. Attack: opponent gains a Curse.
            self._start_attack(state, WITCH.id, 'witch_curse')

    def _start_attack(self, state: DominionState, card_id: int, decision_type: str,
                       min_select: int = 0, max_select: int = 0):
        """Initiate an attack. Check if opponent has Moat."""
        opp_idx = 1 - state.active_player
        opp = state.players[opp_idx]

        if MOAT.id in opp.hand:
            # Opponent can react
            state._save_phase = state.phase
            state.phase = 'react'
            state.current_player = opp_idx
            state.pending = PendingDecision(
                card_id=card_id, decision_type='react',
                target_player=opp_idx,
                min_select=min_select, max_select=max_select,
            )
        else:
            # No Moat — resolve attack immediately
            self._resolve_attack(state, card_id, opp_idx, min_select, max_select)

    def _resolve_attack(self, state: DominionState, card_id: int, opp_idx: int,
                         min_select: int = 0, max_select: int = 0):
        """Apply the attack effect on the opponent."""
        opp = state.players[opp_idx]

        if card_id == MILITIA.id:
            if len(opp.hand) > 3:
                discard_count = len(opp.hand) - 3
                state.current_player = opp_idx
                state.pending = PendingDecision(
                    card_id=MILITIA.id, decision_type='select_hand',
                    min_select=discard_count, max_select=discard_count,
                    target_player=opp_idx,
                )
                state.phase = 'select'
            # If 3 or fewer, nothing happens

        elif card_id == BUREAUCRAT.id:
            has_victory = any('Victory' in CARD_BY_ID[cid].types for cid in opp.hand)
            if has_victory:
                state.current_player = opp_idx
                state.pending = PendingDecision(
                    card_id=BUREAUCRAT.id, decision_type='bureaucrat_topdeck',
                    target_player=opp_idx,
                )
                state.phase = 'select'
            # If no Victory cards, reveal hand (automatic, nothing to do)

        elif card_id == BANDIT.id:
            revealed = opp.draw_to_list(2)
            trashable = [cid for cid in revealed
                         if cid != COPPER.id and 'Treasure' in CARD_BY_ID[cid].types]
            non_trashable = [cid for cid in revealed if cid not in trashable]

            if len(trashable) == 1:
                # Auto-trash the only option
                state.trash.append(trashable[0])
                opp.discard.extend(non_trashable)
            elif len(trashable) > 1:
                # Opponent must choose which to trash
                state.current_player = opp_idx
                state.pending = PendingDecision(
                    card_id=BANDIT.id, decision_type='bandit_trash',
                    bandit_revealed=revealed,
                    target_player=opp_idx,
                )
                state.phase = 'select'
            else:
                # No trashable treasures — discard all revealed
                opp.discard.extend(revealed)

        elif card_id == WITCH.id:
            if state.supply.get(CURSE.id, 0) > 0:
                state.supply[CURSE.id] -= 1
                opp.discard.append(CURSE.id)

    def _resolve_pending(self, state: DominionState, action: int):
        """Resolve a pending decision."""
        p = state.pending
        player = state.players[state.current_player]

        if p.decision_type == 'react':
            if action == REVEAL_MOAT:
                # Attack blocked — return to attacker
                state.current_player = state.active_player
                state.pending = None
                state.phase = state._save_phase if state._save_phase else 'action'
                state._save_phase = ''
            elif action == DECLINE_REACTION:
                # Proceed with attack
                opp_idx = p.target_player
                card_id = p.card_id
                state.pending = None
                state.phase = state._save_phase if state._save_phase else 'action'
                state._save_phase = ''
                self._resolve_attack(state, card_id, opp_idx, p.min_select, p.max_select)

        elif p.decision_type == 'select_hand':
            if action == DONE_SELECTING:
                self._finish_select_hand(state)
            elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                if p.card_id == CELLAR.id:
                    # Cellar: discard selected, then draw that many after done
                    player.remove_from_hand(card_id)
                    player.discard.append(card_id)
                    p.selected_count += 1
                    if p.selected_count >= p.max_select:
                        self._finish_select_hand(state)
                elif p.card_id == CHAPEL.id:
                    # Chapel: trash selected
                    player.remove_from_hand(card_id)
                    state.trash.append(card_id)
                    p.selected_count += 1
                    if p.selected_count >= p.max_select:
                        self._finish_select_hand(state)
                elif p.card_id == MILITIA.id:
                    # Militia: discard selected
                    player.remove_from_hand(card_id)
                    player.discard.append(card_id)
                    p.selected_count += 1
                    if p.selected_count >= p.min_select:
                        self._finish_select_hand(state)
                elif p.card_id == POACHER.id:
                    # Poacher: discard selected
                    player.remove_from_hand(card_id)
                    player.discard.append(card_id)
                    p.selected_count += 1
                    if p.selected_count >= p.min_select:
                        self._finish_select_hand(state)

        elif p.decision_type == 'select_trash_hand':
            if action == DONE_SELECTING:
                # Fizzle (e.g. no valid card to trash)
                state.pending = None
                state.phase = 'action'
            elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                player.remove_from_hand(card_id)
                state.trash.append(card_id)
                trashed_cost = CARD_BY_ID[card_id].cost

                if p.card_id == MONEYLENDER.id:
                    state.coins += 3
                    state.pending = None
                    state.phase = 'action'
                elif p.card_id == REMODEL.id:
                    state.pending = PendingDecision(
                        card_id=REMODEL.id, decision_type='remodel_gain',
                        trashed_card_cost=trashed_cost,
                    )
                elif p.card_id == MINE.id:
                    state.pending = PendingDecision(
                        card_id=MINE.id, decision_type='mine_gain',
                        trashed_card_cost=trashed_cost,
                    )

        elif p.decision_type in ('gain', 'remodel_gain'):
            if action >= GAIN_OFFSET and action < REVEAL_MOAT:
                card_id = action - GAIN_OFFSET
                state.supply[card_id] -= 1
                player.discard.append(card_id)
                state.pending = None
                state.phase = 'action'
                self._check_game_end(state)

        elif p.decision_type == 'mine_gain':
            if action >= GAIN_OFFSET and action < REVEAL_MOAT:
                card_id = action - GAIN_OFFSET
                state.supply[card_id] -= 1
                player.hand.append(card_id)  # Mine gains to hand
                state.pending = None
                state.phase = 'action'
                self._check_game_end(state)

        elif p.decision_type == 'artisan_gain':
            if action >= GAIN_OFFSET and action < REVEAL_MOAT:
                card_id = action - GAIN_OFFSET
                state.supply[card_id] -= 1
                player.hand.append(card_id)  # Artisan gains to hand
                # Now must put a card from hand on deck
                state.pending = PendingDecision(
                    card_id=ARTISAN.id, decision_type='artisan_topdeck',
                )
                self._check_game_end(state)

        elif p.decision_type == 'artisan_topdeck':
            if action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                player.remove_from_hand(card_id)
                player.deck.append(card_id)
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'harbinger_topdeck':
            if action == DONE_SELECTING:
                state.pending = None
                state.phase = 'action'
            elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                player.discard.remove(card_id)
                player.deck.append(card_id)
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'vassal_play':
            rcid = p.revealed[0]
            if action == ACCEPT:
                # Play the revealed action from discard
                player.discard.remove(rcid)
                player.in_play.append(rcid)
                card = CARD_BY_ID[rcid]
                state.actions_remaining += card.plus_actions
                state.buys_remaining += card.plus_buys
                state.coins += card.plus_coins
                if card.plus_cards > 0:
                    player.draw(card.plus_cards)
                state.pending = None
                state.phase = 'action'
                if card.has_effect:
                    self._handle_effect(state, rcid)
            elif action == DECLINE:
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'library_draw':
            if action == ACCEPT:
                # Keep the card (already in hand via library step)
                state.pending = None
                self._library_step(state)
            elif action == DECLINE:
                # Set aside the action card (move from hand to discard)
                if p.revealed:
                    cid = p.revealed[0]
                    player.remove_from_hand(cid)
                    player.discard.append(cid)
                state.pending = None
                self._library_step(state)

        elif p.decision_type == 'sentry_card':
            rcid = p.revealed[0]
            remaining = p.revealed[1:] if len(p.revealed) > 1 else []

            if action >= SELECT_OFFSET and action < GAIN_OFFSET:
                # Trash this card
                state.trash.append(rcid)
            elif action == ACCEPT:
                # Discard this card
                player.discard.append(rcid)
            elif action == DECLINE:
                # Put back on deck
                player.deck.append(rcid)

            if remaining:
                state.pending = PendingDecision(
                    card_id=SENTRY.id, decision_type='sentry_card',
                    revealed=remaining,
                )
            else:
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'bandit_trash':
            if action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                state.trash.append(card_id)
                # Discard the other revealed cards
                for cid in p.bandit_revealed:
                    if cid == card_id:
                        card_id = -1  # Only remove first match
                        continue
                    player.discard.append(cid)
                state.current_player = state.active_player
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'bureaucrat_topdeck':
            if action == DONE_SELECTING:
                # No Victory cards — reveal hand, nothing happens
                state.current_player = state.active_player
                state.pending = None
                state.phase = 'action'
            elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                player.remove_from_hand(card_id)
                player.deck.append(card_id)
                state.current_player = state.active_player
                state.pending = None
                state.phase = 'action'

        elif p.decision_type == 'throne_select':
            if action == DONE_SELECTING:
                # No action to play — fizzle
                state.pending = None
                state.phase = 'action'
            elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
                card_id = action - SELECT_OFFSET
                # Play this action card (first time)
                player.remove_from_hand(card_id)
                player.in_play.append(card_id)
                card = CARD_BY_ID[card_id]
                # Set up throne tracking for second play
                state.throne_card = card_id
                state.throne_remaining = 1
                # Apply first play
                state.actions_remaining += card.plus_actions
                state.buys_remaining += card.plus_buys
                state.coins += card.plus_coins
                if card.plus_cards > 0:
                    player.draw(card.plus_cards)
                state.pending = None
                state.phase = 'action'
                if card.has_effect:
                    self._handle_effect(state, card_id)
                # If no pending from effect, trigger second play immediately
                if state.pending is None:
                    self._do_throne_replay(state)

    def _finish_select_hand(self, state: DominionState):
        """Finish a select_hand pending decision."""
        p = state.pending
        player = state.players[state.current_player]

        if p.card_id == CELLAR.id:
            # Draw cards equal to number discarded
            player.draw(p.selected_count)
            state.pending = None
            state.phase = 'action'
        elif p.card_id == CHAPEL.id:
            state.pending = None
            state.phase = 'action'
        elif p.card_id == MILITIA.id:
            # Return control to active player
            state.current_player = state.active_player
            state.pending = None
            state.phase = 'action'
        elif p.card_id == POACHER.id:
            state.pending = None
            state.phase = 'action'

    def _library_step(self, state: DominionState):
        """One step of Library: draw to 7, may skip Actions."""
        player = state.players[state.active_player]
        if len(player.hand) >= 7:
            state.pending = None
            state.phase = 'action'
            return

        drawn = player.draw_to_list(1)
        if not drawn:
            # No cards left to draw
            state.pending = None
            state.phase = 'action'
            return

        cid = drawn[0]
        if 'Action' in CARD_BY_ID[cid].types:
            # Offer choice: keep or set aside
            player.hand.append(cid)
            state.pending = PendingDecision(
                card_id=LIBRARY.id, decision_type='library_draw',
                revealed=[cid],
            )
            state.phase = 'select'
        else:
            # Non-action: always keep
            player.hand.append(cid)
            # Continue drawing
            self._library_step(state)

    def _do_throne_replay(self, state: DominionState):
        """Execute the second (or subsequent) play of a Throne Room'd card."""
        if state.throne_remaining <= 0:
            return
        card_id = state.throne_card
        card = CARD_BY_ID[card_id]
        player = state.players[state.active_player]

        state.throne_remaining -= 1
        if state.throne_remaining == 0:
            state.throne_card = -1

        # Apply the card's bonuses again
        state.actions_remaining += card.plus_actions
        state.buys_remaining += card.plus_buys
        state.coins += card.plus_coins
        if card.plus_cards > 0:
            player.draw(card.plus_cards)

        if card.has_effect:
            self._handle_effect(state, card_id)

    def _buy_card(self, state: DominionState, card_id: int):
        """Buy a card from supply."""
        card = CARD_BY_ID[card_id]
        player = state.players[state.current_player]

        state.coins -= card.cost
        state.buys_remaining -= 1
        state.supply[card_id] -= 1
        player.discard.append(card_id)

        self._check_game_end(state)

    def _do_cleanup(self, state: DominionState):
        """Cleanup phase: discard hand + in_play, draw 5, switch turns."""
        player = state.players[state.active_player]

        # Discard hand and in_play
        player.discard.extend(player.hand)
        player.hand.clear()
        player.discard.extend(player.in_play)
        player.in_play.clear()

        # Draw 5
        player.draw(5)

        # Switch to next player's turn
        next_player = 1 - state.active_player
        state.active_player = next_player
        state.current_player = next_player
        state.phase = 'action'
        state.actions_remaining = 1
        state.buys_remaining = 1
        state.coins = 0
        state.merchant_silver_bonus = 0
        state.merchant_silver_triggered = False
        state.throne_card = -1
        state.throne_remaining = 0
        state.pending = None

        if state.active_player == 0:
            state.turn_number += 1

        self._check_game_end(state)

    def _check_game_end(self, state: DominionState):
        """Check if the game is over."""
        # Province pile empty
        if state.supply.get(PROVINCE.id, 0) == 0:
            state.game_over = True
            return

        # 3 supply piles empty
        empty = sum(1 for cnt in state.supply.values() if cnt == 0)
        if empty >= 3:
            state.game_over = True

    def is_terminal(self, state: DominionState) -> bool:
        return state.game_over

    def get_reward(self, state: DominionState, player: int) -> float:
        """Get reward for player in terminal state. 1.0 win, -1.0 loss, 0.0 draw."""
        if not state.game_over:
            return 0.0

        vp0 = state.players[0].count_vp(state.kingdom_cards)
        vp1 = state.players[1].count_vp(state.kingdom_cards)

        if vp0 > vp1:
            return 1.0 if player == 0 else -1.0
        elif vp1 > vp0:
            return 1.0 if player == 1 else -1.0
        return 0.0

    def get_symmetries(self, state: DominionState, policy: np.ndarray) -> List[Tuple]:
        """No useful symmetries for Dominion."""
        return [(state, policy)]

    def action_to_string(self, action: int) -> str:
        """Convert action ID to human-readable string."""
        if action == END_ACTIONS:
            return "End Actions"
        elif action == END_BUYS:
            return "End Buys"
        elif action == DONE_SELECTING:
            return "Done Selecting"
        elif action >= PLAY_OFFSET and action < BUY_OFFSET:
            return f"Play {card_name(action - PLAY_OFFSET)}"
        elif action >= BUY_OFFSET and action < SELECT_OFFSET:
            return f"Buy {card_name(action - BUY_OFFSET)}"
        elif action >= SELECT_OFFSET and action < GAIN_OFFSET:
            return f"Select {card_name(action - SELECT_OFFSET)}"
        elif action >= GAIN_OFFSET and action < REVEAL_MOAT:
            return f"Gain {card_name(action - GAIN_OFFSET)}"
        elif action == REVEAL_MOAT:
            return "Reveal Moat"
        elif action == DECLINE_REACTION:
            return "Decline Reaction"
        elif action == ACCEPT:
            return "Accept"
        elif action == DECLINE:
            return "Decline"
        return f"Unknown({action})"

    def state_to_string(self, state: DominionState) -> str:
        """Human-readable state representation."""
        lines = [f"{'='*50}"]
        lines.append(f"Turn {state.turn_number} | Player {state.active_player}'s turn | "
                     f"Phase: {state.phase} | Deciding: Player {state.current_player}")
        lines.append(f"Actions: {state.actions_remaining} | Coins: {state.coins} | "
                     f"Buys: {state.buys_remaining}")
        lines.append(f"{'='*50}")

        for p in range(2):
            ps = state.players[p]
            vp = ps.count_vp(state.kingdom_cards)
            lines.append(f"\nPlayer {p} (VP: {vp}, Total: {ps.total_cards()}):")
            hand_str = ', '.join(card_name(c) for c in sorted(ps.hand))
            lines.append(f"  Hand ({len(ps.hand)}): {hand_str}")
            if ps.in_play:
                play_str = ', '.join(card_name(c) for c in ps.in_play)
                lines.append(f"  In Play: {play_str}")
            lines.append(f"  Deck: {len(ps.deck)} | Discard: {len(ps.discard)}")

        lines.append(f"\nSupply:")
        for cid in sorted(state.supply.keys()):
            if state.supply[cid] > 0 or cid < 7:
                lines.append(f"  {card_name(cid)}: {state.supply[cid]}")

        if state.pending:
            lines.append(f"\nPending: {state.pending.decision_type} "
                         f"(card: {card_name(state.pending.card_id)})")

        if state.game_over:
            lines.append(f"\nGAME OVER")
            for p in range(2):
                lines.append(f"  Player {p}: {state.players[p].count_vp(state.kingdom_cards)} VP")

        return '\n'.join(lines)

    def state_to_summary(self, state: DominionState) -> Dict[str, Any]:
        """JSON-serializable state summary."""
        def player_summary(p):
            ps = state.players[p]
            return {
                'hand': [card_name(c) for c in sorted(ps.hand)],
                'hand_ids': sorted(ps.hand),
                'in_play': [card_name(c) for c in ps.in_play],
                'in_play_ids': ps.in_play,
                'deck_size': len(ps.deck),
                'discard_size': len(ps.discard),
                'total_cards': ps.total_cards(),
                'vp': ps.count_vp(state.kingdom_cards),
            }
        return {
            'turn_number': state.turn_number,
            'active_player': state.active_player,
            'current_player': state.current_player,
            'phase': state.phase,
            'actions_remaining': state.actions_remaining,
            'buys_remaining': state.buys_remaining,
            'coins': state.coins,
            'players': [player_summary(p) for p in range(2)],
            'supply': {card_name(cid): cnt for cid, cnt in sorted(state.supply.items())},
            'trash': [card_name(c) for c in state.trash],
            'kingdom_cards': [card_name(c) for c in state.kingdom_cards],
            'game_over': state.game_over,
            'pending': state.pending.decision_type if state.pending else None,
        }
