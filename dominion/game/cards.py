"""
Dominion Base Set (2nd Edition) card definitions.

All 33 card types as frozen dataclass instances.
Card type IDs:
  0-6:   Basic supply (Copper, Silver, Gold, Estate, Duchy, Province, Curse)
  7-32:  26 Kingdom cards
"""
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class CardDef:
    """Immutable card type definition."""
    id: int
    name: str
    cost: int
    types: FrozenSet[str]  # {'Action'}, {'Treasure'}, {'Victory'}, {'Action', 'Attack'}, etc.
    # Treasure coins (static value when played)
    coins: int = 0
    # Victory points (static)
    vp: int = 0
    # Action card bonuses
    plus_cards: int = 0
    plus_actions: int = 0
    plus_buys: int = 0
    plus_coins: int = 0
    # Whether this card has an effect handler beyond simple bonuses
    has_effect: bool = False


# ── Basic Supply ──

COPPER   = CardDef(0,  'Copper',   0, frozenset({'Treasure'}), coins=1)
SILVER   = CardDef(1,  'Silver',   3, frozenset({'Treasure'}), coins=2)
GOLD     = CardDef(2,  'Gold',     6, frozenset({'Treasure'}), coins=3)
ESTATE   = CardDef(3,  'Estate',   2, frozenset({'Victory'}), vp=1)
DUCHY    = CardDef(4,  'Duchy',    5, frozenset({'Victory'}), vp=3)
PROVINCE = CardDef(5,  'Province', 8, frozenset({'Victory'}), vp=6)
CURSE    = CardDef(6,  'Curse',    0, frozenset({'Curse'}), vp=-1)

# ── Kingdom Cards (alphabetical, IDs 7-32) ──

CELLAR      = CardDef(7,  'Cellar',      2, frozenset({'Action'}), plus_actions=1, has_effect=True)
MOAT        = CardDef(8,  'Moat',        2, frozenset({'Action', 'Reaction'}), plus_cards=2)
CHAPEL      = CardDef(9,  'Chapel',      2, frozenset({'Action'}), has_effect=True)
HARBINGER   = CardDef(10, 'Harbinger',   3, frozenset({'Action'}), plus_cards=1, plus_actions=1, has_effect=True)
MERCHANT    = CardDef(11, 'Merchant',    3, frozenset({'Action'}), plus_cards=1, plus_actions=1, has_effect=True)
VASSAL      = CardDef(12, 'Vassal',      3, frozenset({'Action'}), plus_coins=2, has_effect=True)
VILLAGE     = CardDef(13, 'Village',     3, frozenset({'Action'}), plus_cards=1, plus_actions=2)
WORKSHOP    = CardDef(14, 'Workshop',    3, frozenset({'Action'}), has_effect=True)
BUREAUCRAT  = CardDef(15, 'Bureaucrat',  4, frozenset({'Action', 'Attack'}), has_effect=True)
GARDENS     = CardDef(16, 'Gardens',     4, frozenset({'Victory'}))
MILITIA     = CardDef(17, 'Militia',     4, frozenset({'Action', 'Attack'}), plus_coins=2, has_effect=True)
MONEYLENDER = CardDef(18, 'Moneylender', 4, frozenset({'Action'}), has_effect=True)
POACHER     = CardDef(19, 'Poacher',     4, frozenset({'Action'}), plus_cards=1, plus_actions=1, plus_coins=1, has_effect=True)
REMODEL     = CardDef(20, 'Remodel',     4, frozenset({'Action'}), has_effect=True)
SMITHY      = CardDef(21, 'Smithy',      4, frozenset({'Action'}), plus_cards=3)
THRONE_ROOM = CardDef(22, 'Throne Room', 4, frozenset({'Action'}), has_effect=True)
ARTISAN     = CardDef(23, 'Artisan',     6, frozenset({'Action'}), has_effect=True)
BANDIT      = CardDef(24, 'Bandit',      5, frozenset({'Action', 'Attack'}), has_effect=True)
COUNCIL_ROOM = CardDef(25, 'Council Room', 5, frozenset({'Action'}), plus_cards=4, plus_buys=1, has_effect=True)
LIBRARY     = CardDef(26, 'Library',     5, frozenset({'Action'}), has_effect=True)
MARKET      = CardDef(27, 'Market',      5, frozenset({'Action'}), plus_cards=1, plus_actions=1, plus_buys=1, plus_coins=1)
MINE        = CardDef(28, 'Mine',        5, frozenset({'Action'}), has_effect=True)
SENTRY      = CardDef(29, 'Sentry',      5, frozenset({'Action'}), plus_cards=1, plus_actions=1, has_effect=True)
WITCH       = CardDef(30, 'Witch',       5, frozenset({'Action', 'Attack'}), plus_cards=2, has_effect=True)

# Note: Base 2nd edition has 26 kingdom cards (IDs 7-32 would be 26 cards)
# Actually we have IDs 7-30 = 24 cards. Let me add the missing two.
# The 2nd edition base set kingdom cards are:
# Cellar, Moat, Chapel, Harbinger, Merchant, Vassal, Village, Workshop,
# Bureaucrat, Gardens, Militia, Moneylender, Poacher, Remodel, Smithy,
# Throne Room, Artisan, Bandit, Council Room, Library, Market, Mine,
# Sentry, Witch
# That's 24 kingdom cards. IDs 7-30 = 24 cards. Let's keep IDs 7-30.

NUM_CARD_TYPES = 31  # 0-30

# ── Lookups ──

ALL_CARDS = [
    COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, CURSE,
    CELLAR, MOAT, CHAPEL, HARBINGER, MERCHANT, VASSAL, VILLAGE, WORKSHOP,
    BUREAUCRAT, GARDENS, MILITIA, MONEYLENDER, POACHER, REMODEL, SMITHY,
    THRONE_ROOM, ARTISAN, BANDIT, COUNCIL_ROOM, LIBRARY, MARKET, MINE,
    SENTRY, WITCH,
]

CARD_BY_ID = {c.id: c for c in ALL_CARDS}
CARD_BY_NAME = {c.name: c for c in ALL_CARDS}

KINGDOM_CARDS = [c for c in ALL_CARDS if c.id >= 7]

BASIC_SUPPLY_IDS = {COPPER.id, SILVER.id, GOLD.id, ESTATE.id, DUCHY.id, PROVINCE.id, CURSE.id}


def get_initial_supply(kingdom_card_ids):
    """Return initial supply counts for a 2-player game.

    Args:
        kingdom_card_ids: list of 10 kingdom card IDs to include

    Returns:
        dict mapping card_type_id -> count
    """
    supply = {
        COPPER.id: 46,    # 60 - 7*2 starting coppers = 46
        SILVER.id: 40,
        GOLD.id: 30,
        ESTATE.id: 8,     # 2-player: 8 each for Victory cards
        DUCHY.id: 8,
        PROVINCE.id: 8,
        CURSE.id: 10,     # 2-player: (num_players - 1) * 10 = 10
    }
    for cid in kingdom_card_ids:
        card = CARD_BY_ID[cid]
        if 'Victory' in card.types:
            supply[cid] = 8   # Victory kingdom cards get 8 copies (2-player)
        else:
            supply[cid] = 10  # All other kingdom cards get 10 copies
    return supply


def card_name(card_id):
    """Get card name from ID."""
    return CARD_BY_ID[card_id].name
