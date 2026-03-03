#!/usr/bin/env python3
"""
Generate Dominion seed data: Smart ActionBigMoney vs pure BigMoney.

The ActionBM heuristic evaluates each kingdom card by a scoring function:
  score = plus_cards*3 + plus_actions*2 + plus_coins*2 + plus_buys + special_bonus

It buys the top 1-2 action cards from whatever kingdom is available,
plays them in chain order (villages first for +actions, then draw cards),
and does BigMoney otherwise. Generates diverse training data across many
different kingdom card combinations.

Games are ActionBM (P0) vs BigMoney (P1), then BigMoney (P0) vs ActionBM (P1),
so the network sees action card strategies from both player perspectives.

Usage:
    python3 scripts/seed_dominion_smart_abm.py --num-games 300 \
        --output /workspace/dominion_data/smart_abm_seed.pkl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from mandala_rl.training.replay_buffer import ReplayBuffer


# Card evaluation scores (pre-computed from card definitions)
# Higher = better to buy. Updated at runtime based on available kingdom cards.
SPECIAL_BONUSES = {
    9:  8,   # Chapel — trashing is incredibly strong (thin deck = consistency)
    18: 5,   # Moneylender — trash Copper for +$3, strong early
    30: 4,   # Witch — +2 cards AND curses opponent (dual value)
    24: 3,   # Bandit — gain Gold + attack
    28: 3,   # Mine — upgrade Copper→Silver→Gold
    22: 2,   # Throne Room — doubles another action
    20: 2,   # Remodel — converts cards (Estate→Silver, etc.)
    15: 1,   # Bureaucrat — gain Silver to deck
}


def score_card(card):
    """Score a kingdom card for buy priority. Higher = buy first."""
    if 'Action' not in card.types:
        return -1  # Don't buy victory/treasure kingdom cards (Gardens)
    base = (card.plus_cards * 3 +
            card.plus_actions * 2 +
            card.plus_coins * 2 +
            card.plus_buys)
    special = SPECIAL_BONUSES.get(card.id, 0)
    # Penalize expensive cards slightly (harder to buy early)
    cost_penalty = max(0, card.cost - 4) * 0.5
    return base + special - cost_penalty


def rank_kingdom_cards(kingdom_card_ids, card_by_id):
    """Rank available kingdom action cards by EV score. Returns sorted list of (score, card_id)."""
    scored = []
    for cid in kingdom_card_ids:
        card = card_by_id[cid]
        s = score_card(card)
        if s > 0:
            scored.append((s, cid))
    scored.sort(reverse=True)
    return scored


def get_play_order(hand_action_ids, card_by_id):
    """Order action cards for play: villages first (non-terminal), then draw, then others."""
    villages = []   # +2 actions (Village, Harbinger, Merchant, Poacher, Sentry)
    draw = []       # Terminal draw (Smithy, Council Room, Witch, Moat, Library)
    economy = []    # +coins, attacks, etc.
    special = []    # Has effects (Chapel, Remodel, Mine, etc.)

    for cid in hand_action_ids:
        card = card_by_id[cid]
        if card.plus_actions >= 2:
            villages.append(cid)
        elif card.plus_actions >= 1:
            # Non-terminal: play before terminals
            villages.append(cid)
        elif card.plus_cards >= 2:
            draw.append(cid)
        elif card.id in (9, 18, 20, 28):  # Chapel, Moneylender, Remodel, Mine
            special.append(cid)
        else:
            economy.append(cid)

    return villages + draw + economy + special


def smart_action_bm(state, valid, ranked_buys, card_by_id, max_action_cards=2,
                    PLAY_OFFSET=3, BUY_OFFSET=34, SELECT_OFFSET=65, GAIN_OFFSET=96,
                    END_ACTIONS=0, END_BUYS=1, DONE_SELECTING=2,
                    REVEAL_MOAT=127, DECLINE_REACTION=128, ACCEPT=129, DECLINE=130,
                    PROVINCE_ID=5, GOLD_ID=2, SILVER_ID=1, DUCHY_ID=4):
    """
    Smart ActionBigMoney heuristic.

    Evaluates available kingdom cards, buys top 1-2, plays in chain order.
    Falls back to BigMoney for treasure buying.
    Handles all pending decisions conservatively.
    """
    # --- Pending decisions: handle conservatively ---
    # Moat reaction: always reveal if we have Moat
    if valid[REVEAL_MOAT]:
        return REVEAL_MOAT

    # Done selecting (Chapel trash, Cellar discard, etc.)
    if valid[DONE_SELECTING]:
        # Chapel: trash Coppers and Estates (up to 4)
        if state.pending and state.pending.decision_type == 'select_hand':
            # Trash Copper and Estate if we can select them
            for trash_target in [0, 3]:  # Copper=0, Estate=3
                if valid[SELECT_OFFSET + trash_target]:
                    return SELECT_OFFSET + trash_target
        # Cellar: discard Victory cards
        if state.pending and state.pending.decision_type == 'select_hand':
            for discard_target in [3, 4, 6]:  # Estate, Duchy, Curse
                if valid[SELECT_OFFSET + discard_target]:
                    return SELECT_OFFSET + discard_target
        return DONE_SELECTING

    # Decline reaction
    if valid[DECLINE_REACTION]:
        return DECLINE_REACTION

    # Accept/Decline for Vassal/Library
    if valid[ACCEPT] and valid[DECLINE]:
        # Vassal: play the revealed action
        if state.pending and state.pending.decision_type == 'vassal_play':
            return ACCEPT
        # Library: skip action cards (keep drawing for treasures)
        if state.pending and state.pending.decision_type == 'library_draw':
            return DECLINE
        return DECLINE

    if valid[ACCEPT]:
        return ACCEPT

    # Gain decisions: gain the best available card
    for gain_prio in [GOLD_ID, SILVER_ID, DUCHY_ID]:
        if valid[GAIN_OFFSET + gain_prio]:
            return GAIN_OFFSET + gain_prio
    # Gain any available card (for Remodel, Workshop, etc.)
    for cid in range(31):
        if valid[GAIN_OFFSET + cid]:
            return GAIN_OFFSET + cid

    # Select decisions (Sentry, Bandit, etc.): pick first valid
    for cid in range(31):
        if valid[SELECT_OFFSET + cid]:
            return SELECT_OFFSET + cid

    # Mine: upgrade treasures (Copper→Silver, Silver→Gold)
    if state.pending and state.pending.decision_type == 'mine_gain':
        if valid[GAIN_OFFSET + GOLD_ID]:
            return GAIN_OFFSET + GOLD_ID
        if valid[GAIN_OFFSET + SILVER_ID]:
            return GAIN_OFFSET + SILVER_ID

    # Bureaucrat topdeck: put a Victory card on top
    if state.pending and state.pending.decision_type == 'bureaucrat_topdeck':
        for cid in [3, 6, 4, 5]:  # Estate, Curse, Duchy, Province (worst first)
            if valid[SELECT_OFFSET + cid]:
                return SELECT_OFFSET + cid

    # Harbinger: topdeck best card from discard
    if state.pending and state.pending.decision_type == 'harbinger_topdeck':
        # Topdeck Gold > Silver > action card, or done
        for cid in [GOLD_ID, SILVER_ID]:
            if valid[SELECT_OFFSET + cid]:
                return SELECT_OFFSET + cid
        return DONE_SELECTING

    # Artisan topdeck: put worst card on top
    if state.pending and state.pending.decision_type == 'artisan_topdeck':
        for cid in [6, 3, 0]:  # Curse, Estate, Copper
            if valid[SELECT_OFFSET + cid]:
                return SELECT_OFFSET + cid
        # Put anything on top
        for cid in range(31):
            if valid[SELECT_OFFSET + cid]:
                return SELECT_OFFSET + cid

    # --- Action phase: play action cards in chain order ---
    player = state.players[state.current_player]
    hand_actions = [cid for cid in player.hand
                    if cid >= 7 and 'Action' in card_by_id[cid].types]

    if hand_actions and valid[END_ACTIONS] and state.actions_remaining > 0:
        # We have actions in hand and actions remaining — play them
        play_order = get_play_order(hand_actions, card_by_id)
        for cid in play_order:
            if valid[PLAY_OFFSET + cid]:
                return PLAY_OFFSET + cid

    # End actions if nothing to play
    if valid[END_ACTIONS]:
        return END_ACTIONS

    # --- Buy phase: play treasures first, then buy ---
    # Python engine doesn't auto-play treasures (C++ only, DEVLOG #54)
    for treasure_id in [GOLD_ID, SILVER_ID, 0]:  # Gold, Silver, Copper
        if valid[PLAY_OFFSET + treasure_id]:
            return PLAY_OFFSET + treasure_id

    if valid[BUY_OFFSET + PROVINCE_ID]:
        return BUY_OFFSET + PROVINCE_ID
    if valid[BUY_OFFSET + GOLD_ID]:
        return BUY_OFFSET + GOLD_ID

    # Buy action cards if we have fewer than max_action_cards
    all_cards = player.deck + player.hand + player.discard + player.in_play
    owned_action_count = sum(1 for c in all_cards
                            if c >= 7 and 'Action' in card_by_id[c].types)

    if owned_action_count < max_action_cards:
        for _, cid in ranked_buys:
            if valid[BUY_OFFSET + cid]:
                return BUY_OFFSET + cid

    # Late game: buy Duchy if Provinces running low
    province_supply = state.supply.get(PROVINCE_ID, 0)
    if province_supply <= 3 and valid[BUY_OFFSET + DUCHY_ID]:
        return BUY_OFFSET + DUCHY_ID

    if valid[BUY_OFFSET + SILVER_ID]:
        return BUY_OFFSET + SILVER_ID

    if valid[END_BUYS]:
        return END_BUYS

    # Fallback
    indices = np.where(valid > 0)[0]
    return int(indices[0]) if len(indices) > 0 else 0


def big_money_action(state, valid,
                     PLAY_OFFSET=3, BUY_OFFSET=34,
                     END_ACTIONS=0, END_BUYS=1, DONE_SELECTING=2,
                     DECLINE_REACTION=128, REVEAL_MOAT=127,
                     ACCEPT=129, DECLINE=130,
                     SELECT_OFFSET=65, GAIN_OFFSET=96,
                     PROVINCE_ID=5, GOLD_ID=2, SILVER_ID=1, DUCHY_ID=4):
    """Pure Big Money: Province > Gold > Silver, always."""
    # Handle pending decisions
    if valid[DONE_SELECTING]:
        return DONE_SELECTING
    if valid[DECLINE_REACTION]:
        return DECLINE_REACTION
    if valid[REVEAL_MOAT]:
        return REVEAL_MOAT
    if valid[ACCEPT]:
        return ACCEPT
    if valid[DECLINE]:
        return DECLINE
    for cid in range(31):
        if valid[GAIN_OFFSET + cid]:
            return GAIN_OFFSET + cid
    for cid in range(31):
        if valid[SELECT_OFFSET + cid]:
            return SELECT_OFFSET + cid

    # End actions immediately (no action cards to play)
    if valid[END_ACTIONS]:
        return END_ACTIONS

    # Play treasures first (Python engine doesn't auto-play)
    for treasure_id in [GOLD_ID, SILVER_ID, 0]:  # Gold, Silver, Copper
        if valid[PLAY_OFFSET + treasure_id]:
            return PLAY_OFFSET + treasure_id

    # Buy: Province > Gold > Duchy (late) > Silver
    if valid[BUY_OFFSET + PROVINCE_ID]:
        return BUY_OFFSET + PROVINCE_ID
    if valid[BUY_OFFSET + GOLD_ID]:
        return BUY_OFFSET + GOLD_ID

    province_supply = state.supply.get(PROVINCE_ID, 0)
    if province_supply <= 3 and valid[BUY_OFFSET + DUCHY_ID]:
        return BUY_OFFSET + DUCHY_ID

    if valid[BUY_OFFSET + SILVER_ID]:
        return BUY_OFFSET + SILVER_ID
    if valid[END_BUYS]:
        return END_BUYS

    indices = np.where(valid > 0)[0]
    return int(indices[0]) if len(indices) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Generate smart ActionBigMoney seed data')
    parser.add_argument('--num-games', type=int, default=300,
                        help='Number of games to generate (half ABM vs BM, half BM vs ABM)')
    parser.add_argument('--output', type=str,
                        default='/workspace/dominion_data/smart_abm_seed.pkl')
    parser.add_argument('--channels', type=int, default=156,
                        help='Pad tensor to this many channels')
    args = parser.parse_args()

    from dominion.game.engine import DominionGame, NUM_ACTIONS
    from dominion.game.cards import CARD_BY_ID

    g = DominionGame()

    examples = []
    t0 = time.time()
    stats = {
        'abm_wins': 0, 'bm_wins': 0, 'draws': 0,
        'action_plays_total': 0, 'games_with_actions': 0,
        'cards_bought': {},
    }

    for game_num in range(args.num_games):
        s = g.get_initial_state()
        game_states = []

        # Rank kingdom cards for this game's setup
        ranked_buys = rank_kingdom_cards(s.kingdom_cards, CARD_BY_ID)

        # Alternate which player is ActionBM
        abm_player = game_num % 2
        action_plays_this_game = 0

        for move_num in range(1000):
            if g.is_terminal(s):
                break
            valid = g.get_valid_moves(s)

            if s.current_player == abm_player:
                action = smart_action_bm(
                    s, valid, ranked_buys, CARD_BY_ID,
                    max_action_cards=2,
                )
            else:
                action = big_money_action(s, valid)

            # Track action card plays
            if 3 <= action < 34:  # PLAY range
                cid = action - 3
                card = CARD_BY_ID.get(cid)
                if card and 'Action' in card.types and cid >= 7:
                    action_plays_this_game += 1

            # Track action card buys
            if 34 <= action < 65:  # BUY range
                cid = action - 34
                if cid >= 7:
                    name = CARD_BY_ID[cid].name
                    stats['cards_bought'][name] = stats['cards_bought'].get(name, 0) + 1

            game_states.append((
                s.get_canonical_form().to_tensor(),
                action,
                s.current_player,
            ))
            s = g.get_next_state(s, action)

        if action_plays_this_game > 0:
            stats['games_with_actions'] += 1
            stats['action_plays_total'] += action_plays_this_game

        # Get outcomes
        r0 = g.get_reward(s, 0)
        r1 = g.get_reward(s, 1)

        # Track win rates
        abm_reward = r0 if abm_player == 0 else r1
        if abm_reward > 0.01:
            stats['abm_wins'] += 1
        elif abm_reward < -0.01:
            stats['bm_wins'] += 1
        else:
            stats['draws'] += 1

        # Build training examples from BOTH players
        for (tensor, action, player) in game_states:
            outcome = r0 if player == 0 else r1
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            policy[action] = 1.0

            # Pad tensor to target channels if needed
            if tensor.shape[0] < args.channels:
                padded = np.zeros((args.channels, tensor.shape[1], tensor.shape[2]),
                                 dtype=tensor.dtype)
                padded[:tensor.shape[0]] = tensor
                tensor = padded

            examples.append((
                tensor,
                policy,
                float(outcome),
                0.0,
                np.zeros(12, dtype=np.float32),
            ))

        if (game_num + 1) % 50 == 0:
            elapsed = time.time() - t0
            total = stats['abm_wins'] + stats['bm_wins'] + stats['draws']
            abm_wr = stats['abm_wins'] / max(total, 1) * 100
            print(f"Games: {game_num + 1}/{args.num_games}  "
                  f"Examples: {len(examples)}  "
                  f"ABM wins: {stats['abm_wins']}/{total} ({abm_wr:.0f}%)  "
                  f"Action plays: {stats['action_plays_total']}  "
                  f"Elapsed: {elapsed:.0f}s")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(max_size=500000)
    buf.add_examples(examples)
    buf.save(Path(args.output))

    total = stats['abm_wins'] + stats['bm_wins'] + stats['draws']
    abm_wr = stats['abm_wins'] / max(total, 1) * 100
    print(f"\n{'='*60}")
    print(f"Done: {len(buf)} examples from {args.num_games} games → {args.output}")
    print(f"ABM vs BM win rate: {stats['abm_wins']}/{total} ({abm_wr:.0f}%)")
    print(f"BM wins: {stats['bm_wins']}, Draws: {stats['draws']}")
    print(f"Games with action plays: {stats['games_with_actions']}/{args.num_games}")
    print(f"Avg action plays/game: {stats['action_plays_total'] / args.num_games:.1f}")
    print(f"\nTop action cards bought:")
    for name, count in sorted(stats['cards_bought'].items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count}")
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
