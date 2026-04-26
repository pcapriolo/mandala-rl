"""
Comprehensive tests for Dominion Base Set 2nd Edition implementation.
"""
import random
import numpy as np
import pytest

from dominion.game.cards import (
    NUM_CARD_TYPES, ALL_CARDS, CARD_BY_ID, KINGDOM_CARDS,
    get_initial_supply, card_name,
    COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, CURSE,
    CELLAR, MOAT, CHAPEL, HARBINGER, MERCHANT, VASSAL, VILLAGE, WORKSHOP,
    BUREAUCRAT, GARDENS, MILITIA, MONEYLENDER, POACHER, REMODEL, SMITHY,
    THRONE_ROOM, ARTISAN, BANDIT, COUNCIL_ROOM, LIBRARY, MARKET, MINE,
    SENTRY, WITCH,
)
from dominion.game.state import (
    DominionState, PlayerState, PendingDecision,
    NUM_ACTIONS, END_ACTIONS, END_BUYS, DONE_SELECTING,
    PLAY_OFFSET, BUY_OFFSET, SELECT_OFFSET, GAIN_OFFSET,
    REVEAL_MOAT, DECLINE_REACTION, ACCEPT, DECLINE,
)
from dominion.game.engine import DominionGame


# ── Helper functions ──

def make_game(kingdom_ids=None, seed=42):
    """Create a game with specific kingdom cards."""
    if kingdom_ids is None:
        kingdom_ids = [c.id for c in KINGDOM_CARDS[:10]]
    game = DominionGame(seed=seed, kingdom_card_ids=kingdom_ids)
    return game, game.get_initial_state()


def setup_hand(state, player, hand_ids):
    """Replace a player's hand with specific cards."""
    state.players[player].hand = list(hand_ids)


def play_action(card):
    """Return action ID for playing a card."""
    return PLAY_OFFSET + card.id


def buy_action(card):
    """Return action ID for buying a card."""
    return BUY_OFFSET + card.id


def select_action(card):
    """Return action ID for selecting a card."""
    return SELECT_OFFSET + card.id


def gain_action(card):
    """Return action ID for gaining a card."""
    return GAIN_OFFSET + card.id


# ══════════════════════════════════════════════════════════════
# Card Definitions
# ══════════════════════════════════════════════════════════════

class TestCardDefinitions:
    def test_all_cards_count(self):
        assert len(ALL_CARDS) == NUM_CARD_TYPES

    def test_ids_unique(self):
        ids = [c.id for c in ALL_CARDS]
        assert len(ids) == len(set(ids))

    def test_ids_sequential(self):
        for i, card in enumerate(ALL_CARDS):
            assert card.id == i

    def test_basic_supply(self):
        assert COPPER.cost == 0
        assert SILVER.cost == 3 and SILVER.coins == 2
        assert GOLD.cost == 6 and GOLD.coins == 3
        assert ESTATE.vp == 1
        assert DUCHY.vp == 3
        assert PROVINCE.vp == 6
        assert CURSE.vp == -1

    def test_kingdom_cards_count(self):
        assert len(KINGDOM_CARDS) == 24  # Base 2nd edition

    def test_card_types(self):
        assert 'Treasure' in COPPER.types
        assert 'Victory' in ESTATE.types
        assert 'Action' in VILLAGE.types
        assert 'Attack' in MILITIA.types
        assert 'Reaction' in MOAT.types

    def test_initial_supply_2_player(self):
        kingdom = [c.id for c in KINGDOM_CARDS[:10]]
        supply = get_initial_supply(kingdom)
        assert supply[COPPER.id] == 46
        assert supply[SILVER.id] == 40
        assert supply[GOLD.id] == 30
        assert supply[ESTATE.id] == 8
        assert supply[PROVINCE.id] == 8
        assert supply[CURSE.id] == 10

    def test_supply_victory_kingdom(self):
        kingdom = [GARDENS.id] + [c.id for c in KINGDOM_CARDS if c.id != GARDENS.id][:9]
        supply = get_initial_supply(kingdom)
        assert supply[GARDENS.id] == 8  # Victory kingdom card


# ══════════════════════════════════════════════════════════════
# Initial State
# ══════════════════════════════════════════════════════════════

class TestInitialState:
    def test_initial_hands(self):
        game, state = make_game()
        for p in range(2):
            assert len(state.players[p].hand) == 5

    def test_initial_deck_size(self):
        game, state = make_game()
        for p in range(2):
            total = state.players[p].total_cards()
            assert total == 10  # 7 Copper + 3 Estate

    def test_initial_deck_composition(self):
        game, state = make_game()
        for p in range(2):
            all_cards = state.players[p].deck + state.players[p].hand
            copper_count = all_cards.count(COPPER.id)
            estate_count = all_cards.count(ESTATE.id)
            assert copper_count == 7
            assert estate_count == 3

    def test_initial_phase(self):
        game, state = make_game()
        assert state.phase == 'action'
        assert state.actions_remaining == 1
        assert state.buys_remaining == 1
        assert state.coins == 0

    def test_kingdom_cards_selected(self):
        game, state = make_game()
        assert len(state.kingdom_cards) == 10

    def test_supply_counts(self):
        game, state = make_game()
        assert state.supply[PROVINCE.id] == 8


# ══════════════════════════════════════════════════════════════
# Phase Transitions
# ══════════════════════════════════════════════════════════════

class TestPhaseTransitions:
    def test_end_actions_to_buy(self):
        game, state = make_game()
        state2 = game.get_next_state(state, END_ACTIONS)
        assert state2.phase == 'buy'

    def test_end_buys_cleanup(self):
        game, state = make_game()
        state2 = game.get_next_state(state, END_ACTIONS)
        state3 = game.get_next_state(state2, END_BUYS)
        # Should be next player's turn
        assert state3.current_player == 1
        assert state3.active_player == 1
        assert state3.phase == 'action'
        assert state3.actions_remaining == 1
        assert state3.buys_remaining == 1

    def test_turn_number_increments(self):
        game, state = make_game()
        assert state.turn_number == 1
        # Player 0 passes
        s = game.get_next_state(state, END_ACTIONS)
        s = game.get_next_state(s, END_BUYS)
        assert s.turn_number == 1  # Still turn 1 (P1 hasn't gone)
        # Player 1 passes
        s = game.get_next_state(s, END_ACTIONS)
        s = game.get_next_state(s, END_BUYS)
        assert s.turn_number == 2  # Now turn 2

    def test_cleanup_draws_five(self):
        game, state = make_game()
        # Give player 0 some cards in discard to ensure draw works
        state.players[0].discard = [COPPER.id] * 10
        s = game.get_next_state(state, END_ACTIONS)
        s = game.get_next_state(s, END_BUYS)
        # Player 1 now has 5 cards
        assert len(s.players[1].hand) == 5
        # Player 0 discarded hand and drew 5
        assert len(s.players[0].hand) == 5


# ══════════════════════════════════════════════════════════════
# Treasure Playing
# ══════════════════════════════════════════════════════════════

class TestTreasures:
    def test_play_copper(self):
        game, state = make_game()
        setup_hand(state, 0, [COPPER.id, COPPER.id, ESTATE.id])
        state.phase = 'buy'
        s = game.get_next_state(state, play_action(COPPER))
        assert s.coins == 1
        assert len(s.players[0].hand) == 2

    def test_play_silver(self):
        game, state = make_game()
        setup_hand(state, 0, [SILVER.id])
        state.phase = 'buy'
        s = game.get_next_state(state, play_action(SILVER))
        assert s.coins == 2

    def test_play_gold(self):
        game, state = make_game()
        setup_hand(state, 0, [GOLD.id])
        state.phase = 'buy'
        s = game.get_next_state(state, play_action(GOLD))
        assert s.coins == 3


# ══════════════════════════════════════════════════════════════
# Buying
# ══════════════════════════════════════════════════════════════

class TestBuying:
    def test_buy_copper(self):
        game, state = make_game()
        state.phase = 'buy'
        state.coins = 0
        s = game.get_next_state(state, buy_action(COPPER))
        assert COPPER.id in s.players[0].discard
        assert s.buys_remaining == 0
        assert s.supply[COPPER.id] == state.supply[COPPER.id] - 1

    def test_buy_province(self):
        game, state = make_game()
        state.phase = 'buy'
        state.coins = 8
        s = game.get_next_state(state, buy_action(PROVINCE))
        assert PROVINCE.id in s.players[0].discard
        assert s.coins == 0

    def test_cannot_buy_too_expensive(self):
        game, state = make_game()
        state.phase = 'buy'
        state.coins = 2
        valid = game.get_valid_moves(state)
        assert valid[buy_action(PROVINCE)] == 0.0
        assert valid[buy_action(ESTATE)] == 1.0  # Cost 2

    def test_cannot_buy_empty_pile(self):
        game, state = make_game()
        state.phase = 'buy'
        state.coins = 10
        state.supply[PROVINCE.id] = 0
        valid = game.get_valid_moves(state)
        assert valid[buy_action(PROVINCE)] == 0.0


# ══════════════════════════════════════════════════════════════
# Individual Card Tests
# ══════════════════════════════════════════════════════════════

class TestVillage:
    def test_village(self):
        game, state = make_game()
        setup_hand(state, 0, [VILLAGE.id, COPPER.id, COPPER.id])
        state.players[0].deck = [COPPER.id, SILVER.id]
        s = game.get_next_state(state, play_action(VILLAGE))
        assert s.actions_remaining == 2  # Had 1, used 1, gained 2
        assert len(s.players[0].hand) == 3  # +1 card (drew 1), -1 Village + original 2


class TestSmithy:
    def test_smithy(self):
        game, state = make_game()
        setup_hand(state, 0, [SMITHY.id])
        state.players[0].deck = [COPPER.id] * 5
        s = game.get_next_state(state, play_action(SMITHY))
        assert len(s.players[0].hand) == 3  # Drew 3
        assert s.actions_remaining == 0


class TestMarket:
    def test_market(self):
        game, state = make_game()
        setup_hand(state, 0, [MARKET.id])
        state.players[0].deck = [COPPER.id] * 5
        s = game.get_next_state(state, play_action(MARKET))
        assert s.actions_remaining == 1  # +1 action
        assert s.buys_remaining == 2     # +1 buy
        assert s.coins == 1             # +1 coin
        assert len(s.players[0].hand) == 1  # +1 card


class TestCellar:
    def test_cellar_discard_and_draw(self):
        game, state = make_game()
        setup_hand(state, 0, [CELLAR.id, ESTATE.id, ESTATE.id, COPPER.id])
        state.players[0].deck = [SILVER.id, GOLD.id, COPPER.id]
        s = game.get_next_state(state, play_action(CELLAR))
        assert s.pending is not None
        assert s.pending.decision_type == 'select_hand'
        # Discard 2 estates
        s = game.get_next_state(s, select_action(ESTATE))
        s = game.get_next_state(s, select_action(ESTATE))
        s = game.get_next_state(s, DONE_SELECTING)
        assert s.pending is None
        assert s.phase == 'action'
        # Should have drawn 2 new cards
        assert len(s.players[0].hand) == 3  # Copper + 2 drawn

    def test_cellar_discard_zero(self):
        game, state = make_game()
        setup_hand(state, 0, [CELLAR.id, COPPER.id])
        s = game.get_next_state(state, play_action(CELLAR))
        s = game.get_next_state(s, DONE_SELECTING)
        assert s.pending is None
        assert len(s.players[0].hand) == 1  # Just Copper


class TestChapel:
    def test_chapel_trash(self):
        game, state = make_game()
        setup_hand(state, 0, [CHAPEL.id, COPPER.id, COPPER.id, ESTATE.id])
        s = game.get_next_state(state, play_action(CHAPEL))
        assert s.pending.decision_type == 'select_hand'
        s = game.get_next_state(s, select_action(COPPER))
        s = game.get_next_state(s, select_action(COPPER))
        s = game.get_next_state(s, DONE_SELECTING)
        assert len(s.trash) == 2
        assert len(s.players[0].hand) == 1  # Just Estate

    def test_chapel_trash_zero(self):
        game, state = make_game()
        setup_hand(state, 0, [CHAPEL.id, COPPER.id])
        s = game.get_next_state(state, play_action(CHAPEL))
        s = game.get_next_state(s, DONE_SELECTING)
        assert len(s.trash) == 0


class TestMoat:
    def test_moat_draw(self):
        game, state = make_game()
        setup_hand(state, 0, [MOAT.id])
        state.players[0].deck = [COPPER.id] * 5
        s = game.get_next_state(state, play_action(MOAT))
        assert len(s.players[0].hand) == 2  # Drew 2


class TestMilitia:
    def test_militia_attack(self):
        game, state = make_game()
        setup_hand(state, 0, [MILITIA.id])
        setup_hand(state, 1, [COPPER.id] * 5)
        s = game.get_next_state(state, play_action(MILITIA))
        assert s.coins == 2
        # Opponent must discard to 3
        assert s.current_player == 1
        assert s.pending.decision_type == 'select_hand'
        s = game.get_next_state(s, select_action(COPPER))
        s = game.get_next_state(s, select_action(COPPER))
        assert s.pending is None
        assert len(s.players[1].hand) == 3
        assert s.current_player == 0

    def test_militia_moat_block(self):
        game, state = make_game()
        setup_hand(state, 0, [MILITIA.id])
        setup_hand(state, 1, [MOAT.id, COPPER.id, COPPER.id, COPPER.id, COPPER.id])
        s = game.get_next_state(state, play_action(MILITIA))
        assert s.current_player == 1
        assert s.pending.decision_type == 'react'
        s = game.get_next_state(s, REVEAL_MOAT)
        assert s.pending is None
        assert len(s.players[1].hand) == 5  # No discard

    def test_militia_moat_decline(self):
        game, state = make_game()
        setup_hand(state, 0, [MILITIA.id])
        setup_hand(state, 1, [MOAT.id, COPPER.id, COPPER.id, COPPER.id, COPPER.id])
        s = game.get_next_state(state, play_action(MILITIA))
        s = game.get_next_state(s, DECLINE_REACTION)
        assert s.pending.decision_type == 'select_hand'
        assert s.current_player == 1

    def test_militia_small_hand(self):
        game, state = make_game()
        setup_hand(state, 0, [MILITIA.id])
        setup_hand(state, 1, [COPPER.id, COPPER.id, COPPER.id])
        s = game.get_next_state(state, play_action(MILITIA))
        # Opponent already has 3 cards, nothing happens
        assert s.pending is None
        assert s.current_player == 0


class TestWorkshop:
    def test_workshop_gain(self):
        game, state = make_game()
        setup_hand(state, 0, [WORKSHOP.id])
        s = game.get_next_state(state, play_action(WORKSHOP))
        assert s.pending.decision_type == 'gain'
        s = game.get_next_state(s, gain_action(SILVER))
        assert SILVER.id in s.players[0].discard
        assert s.supply[SILVER.id] == 39

    def test_workshop_cost_limit(self):
        game, state = make_game()
        setup_hand(state, 0, [WORKSHOP.id])
        s = game.get_next_state(state, play_action(WORKSHOP))
        valid = game.get_valid_moves(s)
        assert valid[gain_action(GOLD)] == 0.0   # Cost 6, limit 4
        assert valid[gain_action(SILVER)] == 1.0  # Cost 3


class TestMoneylender:
    def test_moneylender_trash_copper(self):
        game, state = make_game()
        setup_hand(state, 0, [MONEYLENDER.id, COPPER.id])
        s = game.get_next_state(state, play_action(MONEYLENDER))
        assert s.pending.decision_type == 'select_trash_hand'
        s = game.get_next_state(s, select_action(COPPER))
        assert s.coins == 3
        assert COPPER.id in s.trash
        assert COPPER.id not in s.players[0].hand

    def test_moneylender_no_copper(self):
        game, state = make_game()
        setup_hand(state, 0, [MONEYLENDER.id, ESTATE.id])
        s = game.get_next_state(state, play_action(MONEYLENDER))
        assert s.pending is None  # No copper to trash


class TestRemodel:
    def test_remodel(self):
        game, state = make_game()
        setup_hand(state, 0, [REMODEL.id, ESTATE.id])
        s = game.get_next_state(state, play_action(REMODEL))
        assert s.pending.decision_type == 'select_trash_hand'
        s = game.get_next_state(s, select_action(ESTATE))
        assert s.pending.decision_type == 'remodel_gain'
        # Estate costs 2, can gain up to 4
        valid = game.get_valid_moves(s)
        assert valid[gain_action(SILVER)] == 1.0  # Cost 3
        assert valid[gain_action(DUCHY)] == 0.0   # Cost 5
        s = game.get_next_state(s, gain_action(SILVER))
        assert SILVER.id in s.players[0].discard


class TestMine:
    def test_mine_upgrade(self):
        game, state = make_game()
        setup_hand(state, 0, [MINE.id, COPPER.id])
        s = game.get_next_state(state, play_action(MINE))
        s = game.get_next_state(s, select_action(COPPER))
        # Copper costs 0, can gain treasure up to 3
        valid = game.get_valid_moves(s)
        assert valid[gain_action(SILVER)] == 1.0
        assert valid[gain_action(GOLD)] == 0.0  # Cost 6, limit 3
        s = game.get_next_state(s, gain_action(SILVER))
        assert SILVER.id in s.players[0].hand  # Mine gains to hand


class TestHarbinger:
    def test_harbinger_topdeck(self):
        game, state = make_game()
        setup_hand(state, 0, [HARBINGER.id])
        state.players[0].deck = [COPPER.id] * 3
        state.players[0].discard = [GOLD.id, SILVER.id]
        s = game.get_next_state(state, play_action(HARBINGER))
        # +1 card, +1 action applied, now can topdeck from discard
        assert s.pending.decision_type == 'harbinger_topdeck'
        s = game.get_next_state(s, select_action(GOLD))
        assert GOLD.id in s.players[0].deck
        assert GOLD.id not in s.players[0].discard

    def test_harbinger_skip(self):
        game, state = make_game()
        setup_hand(state, 0, [HARBINGER.id])
        state.players[0].deck = [COPPER.id] * 3
        state.players[0].discard = [SILVER.id]
        s = game.get_next_state(state, play_action(HARBINGER))
        s = game.get_next_state(s, DONE_SELECTING)
        assert s.pending is None


class TestMerchant:
    def test_merchant_silver_bonus(self):
        game, state = make_game()
        setup_hand(state, 0, [MERCHANT.id, SILVER.id])
        state.players[0].deck = [COPPER.id] * 5
        s = game.get_next_state(state, play_action(MERCHANT))
        assert s.merchant_silver_bonus == 1
        # Now play Silver in buy phase
        s = game.get_next_state(s, END_ACTIONS)
        s = game.get_next_state(s, play_action(SILVER))
        assert s.coins == 3  # 2 from Silver + 1 from Merchant

    def test_merchant_double(self):
        game, state = make_game()
        setup_hand(state, 0, [MERCHANT.id, MERCHANT.id, SILVER.id])
        state.players[0].deck = [COPPER.id] * 5
        state.actions_remaining = 2
        s = game.get_next_state(state, play_action(MERCHANT))
        s = game.get_next_state(s, play_action(MERCHANT))
        assert s.merchant_silver_bonus == 2
        s = game.get_next_state(s, END_ACTIONS)
        s = game.get_next_state(s, play_action(SILVER))
        assert s.coins == 4  # 2 + 2 bonus


class TestVassal:
    def test_vassal_play_action(self):
        game, state = make_game()
        setup_hand(state, 0, [VASSAL.id])
        # Top of deck = end of list, so VILLAGE is on top
        state.players[0].deck = [COPPER.id, VILLAGE.id]
        s = game.get_next_state(state, play_action(VASSAL))
        assert s.coins == 2
        assert s.pending is not None
        assert s.pending.decision_type == 'vassal_play'
        s = game.get_next_state(s, ACCEPT)
        # Village played: +1 card, +2 actions
        assert s.actions_remaining == 2

    def test_vassal_decline(self):
        game, state = make_game()
        setup_hand(state, 0, [VASSAL.id])
        state.players[0].deck = [VILLAGE.id]  # Village on top
        s = game.get_next_state(state, play_action(VASSAL))
        s = game.get_next_state(s, DECLINE)
        assert s.pending is None
        assert VILLAGE.id in s.players[0].discard

    def test_vassal_non_action(self):
        game, state = make_game()
        setup_hand(state, 0, [VASSAL.id])
        state.players[0].deck = [COPPER.id]  # Copper on top
        s = game.get_next_state(state, play_action(VASSAL))
        # Copper is not an action, no choice offered
        assert s.pending is None
        assert COPPER.id in s.players[0].discard


class TestBureaucrat:
    def test_bureaucrat_gain_silver(self):
        game, state = make_game()
        setup_hand(state, 0, [BUREAUCRAT.id])
        setup_hand(state, 1, [ESTATE.id, COPPER.id, COPPER.id])
        s = game.get_next_state(state, play_action(BUREAUCRAT))
        assert SILVER.id in s.players[0].deck
        assert s.supply[SILVER.id] == 39
        # Opponent must topdeck Victory
        assert s.current_player == 1
        s = game.get_next_state(s, select_action(ESTATE))
        assert ESTATE.id in s.players[1].deck
        assert ESTATE.id not in s.players[1].hand

    def test_bureaucrat_no_victory(self):
        game, state = make_game()
        setup_hand(state, 0, [BUREAUCRAT.id])
        setup_hand(state, 1, [COPPER.id, COPPER.id])
        s = game.get_next_state(state, play_action(BUREAUCRAT))
        # Opponent has no Victory cards → auto-resolved (reveals hand, nothing happens)
        assert s.pending is None
        assert s.current_player == 0


class TestPoacher:
    def test_poacher_empty_piles(self):
        game, state = make_game()
        setup_hand(state, 0, [POACHER.id, COPPER.id, ESTATE.id])
        state.players[0].deck = [COPPER.id] * 5
        # Make 2 supply piles empty
        state.supply[CURSE.id] = 0
        state.supply[ESTATE.id] = 0
        s = game.get_next_state(state, play_action(POACHER))
        # Must discard 2
        assert s.pending.decision_type == 'select_hand'
        assert s.pending.min_select == 2

    def test_poacher_no_empty(self):
        game, state = make_game()
        setup_hand(state, 0, [POACHER.id])
        state.players[0].deck = [COPPER.id] * 5
        s = game.get_next_state(state, play_action(POACHER))
        assert s.pending is None  # No empty piles, no discard


class TestArtisan:
    def test_artisan(self):
        game, state = make_game()
        setup_hand(state, 0, [ARTISAN.id, COPPER.id])
        s = game.get_next_state(state, play_action(ARTISAN))
        assert s.pending.decision_type == 'artisan_gain'
        # Gain a card costing up to 5 to hand
        s = game.get_next_state(s, gain_action(DUCHY))
        assert DUCHY.id in s.players[0].hand
        assert s.pending.decision_type == 'artisan_topdeck'
        # Put a card from hand on deck
        s = game.get_next_state(s, select_action(COPPER))
        assert COPPER.id in s.players[0].deck
        assert COPPER.id not in s.players[0].hand


class TestBandit:
    def test_bandit_gain_gold(self):
        game, state = make_game()
        setup_hand(state, 0, [BANDIT.id])
        setup_hand(state, 1, [COPPER.id])
        state.players[1].deck = [COPPER.id, COPPER.id]
        s = game.get_next_state(state, play_action(BANDIT))
        assert GOLD.id in s.players[0].discard
        # Opponent revealed 2 coppers — no non-copper treasures, discard both
        assert s.pending is None

    def test_bandit_trash_silver(self):
        game, state = make_game()
        setup_hand(state, 0, [BANDIT.id])
        setup_hand(state, 1, [COPPER.id])
        state.players[1].deck = [SILVER.id, ESTATE.id]
        s = game.get_next_state(state, play_action(BANDIT))
        # Silver revealed — auto-trash (only one non-copper treasure)
        assert SILVER.id in s.trash


class TestCouncilRoom:
    def test_council_room(self):
        game, state = make_game()
        setup_hand(state, 0, [COUNCIL_ROOM.id])
        state.players[0].deck = [COPPER.id] * 10
        state.players[1].deck = [COPPER.id] * 5
        opp_hand = len(state.players[1].hand)
        s = game.get_next_state(state, play_action(COUNCIL_ROOM))
        assert len(s.players[0].hand) == 4  # Drew 4
        assert s.buys_remaining == 2
        assert len(s.players[1].hand) == opp_hand + 1  # Opponent drew 1


class TestLibrary:
    def test_library_draw_to_seven(self):
        game, state = make_game()
        setup_hand(state, 0, [LIBRARY.id, COPPER.id])
        state.players[0].deck = [COPPER.id] * 10
        s = game.get_next_state(state, play_action(LIBRARY))
        # Should draw to 7 (had 1 Copper after playing Library, draw 6 non-actions)
        assert len(s.players[0].hand) == 7

    def test_library_skip_action(self):
        game, state = make_game()
        setup_hand(state, 0, [LIBRARY.id, COPPER.id])
        # Put an action on top, then non-actions
        state.players[0].deck = [COPPER.id] * 8 + [VILLAGE.id]
        s = game.get_next_state(state, play_action(LIBRARY))
        # First draw is Village (action) → offer choice
        if s.pending and s.pending.decision_type == 'library_draw':
            s = game.get_next_state(s, DECLINE)  # Skip it
            # Should continue drawing
            assert len(s.players[0].hand) == 7 or s.pending is not None


class TestSentry:
    def test_sentry_trash_both(self):
        game, state = make_game()
        setup_hand(state, 0, [SENTRY.id])
        # Deck top=end: Gold on top (drawn as +1 card), then Copper, then Estate
        state.players[0].deck = [ESTATE.id, COPPER.id, GOLD.id]
        s = game.get_next_state(state, play_action(SENTRY))
        # +1 card (Gold drawn to hand), +1 action, revealed 2: Copper and Estate
        assert s.pending is not None
        assert s.pending.decision_type == 'sentry_card'
        # Trash first revealed card
        first_cid = s.pending.revealed[0]
        s = game.get_next_state(s, SELECT_OFFSET + first_cid)
        if s.pending:
            # Trash second
            second_cid = s.pending.revealed[0]
            s = game.get_next_state(s, SELECT_OFFSET + second_cid)
        assert len(s.trash) == 2


class TestWitch:
    def test_witch_curse(self):
        game, state = make_game()
        setup_hand(state, 0, [WITCH.id])
        state.players[0].deck = [COPPER.id] * 5
        setup_hand(state, 1, [COPPER.id] * 3)
        s = game.get_next_state(state, play_action(WITCH))
        assert len(s.players[0].hand) == 2  # Drew 2
        assert CURSE.id in s.players[1].discard
        assert s.supply[CURSE.id] == 9

    def test_witch_moat_block(self):
        game, state = make_game()
        setup_hand(state, 0, [WITCH.id])
        state.players[0].deck = [COPPER.id] * 5
        setup_hand(state, 1, [MOAT.id, COPPER.id])
        s = game.get_next_state(state, play_action(WITCH))
        assert s.pending.decision_type == 'react'
        s = game.get_next_state(s, REVEAL_MOAT)
        assert CURSE.id not in s.players[1].discard


class TestThroneRoom:
    def test_throne_room_smithy(self):
        game, state = make_game()
        setup_hand(state, 0, [THRONE_ROOM.id, SMITHY.id])
        state.players[0].deck = [COPPER.id] * 10
        s = game.get_next_state(state, play_action(THRONE_ROOM))
        assert s.pending.decision_type == 'throne_select'
        s = game.get_next_state(s, select_action(SMITHY))
        # Smithy played twice: drew 3 + 3 = 6
        assert len(s.players[0].hand) == 6

    def test_throne_room_village(self):
        game, state = make_game()
        setup_hand(state, 0, [THRONE_ROOM.id, VILLAGE.id])
        state.players[0].deck = [COPPER.id] * 10
        s = game.get_next_state(state, play_action(THRONE_ROOM))
        s = game.get_next_state(s, select_action(VILLAGE))
        # Village x2: +2 cards, +4 actions (minus the Throne Room's action cost)
        assert len(s.players[0].hand) == 2
        assert s.actions_remaining == 4  # 1 - 1(TR) + 2 + 2

    def test_throne_room_no_actions(self):
        game, state = make_game()
        setup_hand(state, 0, [THRONE_ROOM.id, COPPER.id])
        s = game.get_next_state(state, play_action(THRONE_ROOM))
        # No action cards in hand → fizzles
        assert s.pending is None or s.pending.decision_type == 'throne_select'
        if s.pending:
            valid = game.get_valid_moves(s)
            assert valid[DONE_SELECTING] == 1.0


# ══════════════════════════════════════════════════════════════
# Game End
# ══════════════════════════════════════════════════════════════

class TestGameEnd:
    def test_provinces_empty(self):
        game, state = make_game()
        state.supply[PROVINCE.id] = 1
        state.phase = 'buy'
        state.coins = 8
        s = game.get_next_state(state, buy_action(PROVINCE))
        assert s.game_over

    def test_three_piles_empty(self):
        game, state = make_game()
        state.supply[COPPER.id] = 1
        state.supply[SILVER.id] = 0
        state.supply[ESTATE.id] = 0
        state.phase = 'buy'
        state.coins = 0
        s = game.get_next_state(state, buy_action(COPPER))
        assert s.game_over  # 3 piles empty

    def test_reward(self):
        game, state = make_game()
        state.game_over = True
        # Player 0: 3 Estates = 3 VP
        state.players[0].hand = [ESTATE.id] * 3
        state.players[0].deck = []
        state.players[0].discard = []
        # Player 1: 1 Province = 6 VP
        state.players[1].hand = [PROVINCE.id]
        state.players[1].deck = []
        state.players[1].discard = []
        assert game.get_reward(state, 0) == -1.0
        assert game.get_reward(state, 1) == 1.0


# ══════════════════════════════════════════════════════════════
# VP Counting
# ══════════════════════════════════════════════════════════════

class TestVPCounting:
    def test_basic_vp(self):
        ps = PlayerState()
        ps.hand = [ESTATE.id, DUCHY.id, PROVINCE.id, CURSE.id]
        assert ps.count_vp([]) == 1 + 3 + 6 - 1

    def test_gardens(self):
        ps = PlayerState()
        ps.hand = [GARDENS.id] + [COPPER.id] * 9
        assert ps.count_vp([GARDENS.id]) == 1  # 10 cards / 10 = 1 VP

        ps.hand = [GARDENS.id] + [COPPER.id] * 19
        assert ps.count_vp([GARDENS.id]) == 2  # 20 / 10 = 2 VP


# ══════════════════════════════════════════════════════════════
# Deck Cycling
# ══════════════════════════════════════════════════════════════

class TestDeckCycling:
    def test_reshuffle(self):
        ps = PlayerState()
        ps.deck = []
        ps.discard = [COPPER.id] * 5
        ps.draw(3)
        assert len(ps.hand) == 3
        assert len(ps.deck) == 2
        assert len(ps.discard) == 0

    def test_reshuffle_not_enough(self):
        ps = PlayerState()
        ps.deck = [COPPER.id]
        ps.discard = [SILVER.id]
        drawn = ps.draw(3)
        assert drawn == 2  # Only 2 cards available
        assert len(ps.hand) == 2


# ══════════════════════════════════════════════════════════════
# Tensor and Canonical Form
# ══════════════════════════════════════════════════════════════

class TestTensorAndCanonical:
    def test_tensor_shape(self):
        game, state = make_game()
        tensor = state.to_tensor()
        # NOTE: this tests the Python reference engine in dominion/game/state.py,
        # which has its own to_tensor() at 280 channels. The C++ engine used for
        # training is at 404 channels per DEVLOG #176; the Python engine has not
        # been updated to match (parallel implementation, used only for local
        # debugging / web UI). Update both together if Python is ever used for
        # training inference.
        assert tensor.shape == (280, 8, 8)
        assert tensor.dtype == np.float32

    def test_canonical_form_player0(self):
        game, state = make_game()
        state.current_player = 0
        canonical = state.get_canonical_form()
        assert canonical.current_player == 0
        assert canonical.players[0].hand == state.players[0].hand

    def test_canonical_form_player1(self):
        game, state = make_game()
        state.current_player = 1
        canonical = state.get_canonical_form()
        assert canonical.current_player == 0
        assert canonical.players[0].hand == state.players[1].hand

    def test_state_copy_independence(self):
        game, state = make_game()
        copy = state.copy()
        copy.players[0].hand.append(GOLD.id)
        assert GOLD.id not in state.players[0].hand
        copy.supply[PROVINCE.id] = 0
        assert state.supply[PROVINCE.id] == 8

    def test_hash(self):
        game, state = make_game()
        h = hash(state)
        assert isinstance(h, int)


# ══════════════════════════════════════════════════════════════
# Valid Moves
# ══════════════════════════════════════════════════════════════

class TestValidMoves:
    def test_action_phase_no_actions(self):
        game, state = make_game()
        setup_hand(state, 0, [COPPER.id, COPPER.id, ESTATE.id])
        valid = game.get_valid_moves(state)
        assert valid[END_ACTIONS] == 1.0
        # No action cards in hand
        for cid in range(NUM_CARD_TYPES):
            if 'Action' in CARD_BY_ID[cid].types:
                assert valid[PLAY_OFFSET + cid] == 0.0

    def test_buy_phase_treasures(self):
        game, state = make_game()
        setup_hand(state, 0, [COPPER.id, SILVER.id, ESTATE.id])
        state.phase = 'buy'
        valid = game.get_valid_moves(state)
        assert valid[play_action(COPPER)] == 1.0
        assert valid[play_action(SILVER)] == 1.0
        assert valid[play_action(ESTATE)] == 0.0  # Not a treasure

    def test_game_over_no_moves(self):
        game, state = make_game()
        state.game_over = True
        valid = game.get_valid_moves(state)
        assert valid.sum() == 0


# ══════════════════════════════════════════════════════════════
# Random Game Completion
# ══════════════════════════════════════════════════════════════

class TestRandomGames:
    def test_random_game_completes(self):
        """Play a random game and verify it completes without crashing."""
        random.seed(42)
        np.random.seed(42)
        game = DominionGame(seed=42)
        state = game.get_initial_state()
        moves = 0
        max_moves = 2000

        while not game.is_terminal(state) and moves < max_moves:
            valid = game.get_valid_moves(state)
            valid_actions = np.where(valid > 0)[0]
            assert len(valid_actions) > 0, f"No valid moves at step {moves}"
            action = random.choice(valid_actions)
            state = game.get_next_state(state, action)
            moves += 1

        assert game.is_terminal(state), f"Game didn't finish in {max_moves} moves"

    def test_multiple_random_games(self):
        """Run 50 random games to verify no crashes."""
        for seed in range(50):
            game = DominionGame(seed=seed)
            state = game.get_initial_state()
            moves = 0
            max_moves = 2000

            while not game.is_terminal(state) and moves < max_moves:
                valid = game.get_valid_moves(state)
                valid_actions = np.where(valid > 0)[0]
                assert len(valid_actions) > 0, f"Game {seed}: no valid moves at step {moves}"
                action = random.choice(valid_actions)
                state = game.get_next_state(state, action)
                moves += 1

            assert game.is_terminal(state), f"Game {seed} didn't finish in {max_moves} moves"


class TestActionToString:
    def test_all_action_strings(self):
        game = DominionGame()
        for i in range(NUM_ACTIONS):
            s = game.action_to_string(i)
            assert isinstance(s, str)
            assert len(s) > 0
