#!/usr/bin/env python3
"""
Generate Dominion seed training data from ActionBigMoney heuristic self-play.

ActionBigMoney strategy:
  - BUY phase: Province > Gold > [1-2 action cards from kingdom] > Silver
  - ACTION phase: play any action cards in hand by priority order
  - Pending decisions: minimize damage (done_selecting or decline when possible)

This bootstraps the VALUE HEAD with proof that playing action cards leads to wins,
breaking the bootstrap deadlock where the network buys action cards but never plays them.

Usage:
    python3 scripts/seed_dominion_action_bm.py --num-games 300 \
        --output /workspace/dominion_data/action_bm_seed.pkl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from mandala_rl.training.replay_buffer import ReplayBuffer


# Action cards by cost (lower = easier to buy early).
# These are the IDs from the Dominion engine.
# Priority for playing: draw-heavy > action-gen > economic/attack
ACTION_PLAY_PRIORITY_IDS = [
    21,  # Smithy (cost 4): +3 cards — best draw card
    25,  # Council Room (cost 5): +4 cards, +1 buy
    26,  # Library (cost 5): draw to 7
    27,  # Market (cost 5): +1 card, +1 action, +1 buy, +$1
    24,  # Bandit (cost 5): +Gold, attacks
    28,  # Mine (cost 5): upgrade treasure
    30,  # Witch (cost 5): +2 cards, curses opponent
    29,  # Sentry (cost 5): +1 card, look at top 2
    23,  # Artisan (cost 6): gain to hand
    17,  # Militia (cost 4): +$2, attack
    18,  # Moneylender (cost 4): trash copper for +$3
    19,  # Poacher (cost 4): +1 card, +$1
    13,  # Village (cost 3): +2 actions, +1 card
    11,  # Merchant (cost 3): +1 card, +$1 (bonus if Silver played)
    12,  # Vassal (cost 3): +$2, may play top of deck
    22,  # Throne Room (cost 4): play action twice
    20,  # Remodel (cost 4): trash for gain cost+2
    15,  # Bureaucrat (cost 4): gain Silver to deck, attack
    10,  # Harbinger (cost 3): +1 card, +1 action
    14,  # Workshop (cost 3): gain card costing <=4
    16,  # Gardens (cost 4): VP per 10 cards
    7,   # Cellar (cost 2): discard any, draw same
    8,   # Moat (cost 2): +2 cards, reaction
    9,   # Chapel (cost 2): trash up to 4
]

# Action cards worth buying early (cost 4-5, give clear value)
GOOD_ACTION_BUYS = {21, 25, 26, 27, 30, 17, 11, 13}  # Smithy, CouncilRoom, Library, Market, Witch, Militia, Merchant, Village


def action_big_money(state, valid, PLAY_OFFSET, BUY_OFFSET, DONE_SELECTING,
                     END_ACTIONS, END_BUYS, DECLINE_REACTION,
                     PROVINCE_ID, GOLD_ID, SILVER_ID):
    """
    ActionBigMoney heuristic:
    - Buys action cards early (1-2 per deck), then pure BigMoney
    - Plays action cards in priority order before buying
    - Handles pending decisions conservatively
    """
    # Pending decisions: minimize interaction
    if valid[DONE_SELECTING]:
        return DONE_SELECTING
    if valid[DECLINE_REACTION]:
        return DECLINE_REACTION

    # Action phase: play action cards by priority
    for cid in ACTION_PLAY_PRIORITY_IDS:
        if valid[PLAY_OFFSET + cid]:
            return PLAY_OFFSET + cid

    # End action phase if nothing left to play
    if valid[END_ACTIONS]:
        return END_ACTIONS

    # Buy phase: Province > Gold > [action card if < 2 in deck] > Silver
    if valid[BUY_OFFSET + PROVINCE_ID]:
        return BUY_OFFSET + PROVINCE_ID
    if valid[BUY_OFFSET + GOLD_ID]:
        return BUY_OFFSET + GOLD_ID

    # Count action cards already in deck/hand/discard (buy ≤2 action cards total)
    player = state.players[state.current_player]
    all_cards = player.deck + player.hand + player.discard + player.in_play
    action_card_count = sum(1 for c in all_cards if c in GOOD_ACTION_BUYS)

    if action_card_count < 2:
        # Buy best affordable action card from the kingdom
        for cid in ACTION_PLAY_PRIORITY_IDS:
            if cid in state.kingdom_cards and valid[BUY_OFFSET + cid]:
                return BUY_OFFSET + cid

    if valid[BUY_OFFSET + SILVER_ID]:
        return BUY_OFFSET + SILVER_ID

    if valid[END_BUYS]:
        return END_BUYS

    # Fallback: first valid action
    indices = np.where(valid > 0)[0]
    return int(indices[0])


def main():
    parser = argparse.ArgumentParser(description='Generate Dominion ActionBigMoney seed data')
    parser.add_argument('--num-games', type=int, default=300)
    parser.add_argument('--output', type=str,
                        default='/workspace/dominion_data/action_bm_seed.pkl')
    args = parser.parse_args()

    from dominion.game.engine import (
        DominionGame, PLAY_OFFSET, BUY_OFFSET,
        COPPER, SILVER, GOLD, PROVINCE,
        END_ACTIONS, END_BUYS, NUM_ACTIONS,
    )
    from dominion.game.state import DONE_SELECTING, DECLINE_REACTION

    PROVINCE_ID = PROVINCE.id
    GOLD_ID = GOLD.id
    SILVER_ID = SILVER.id

    print(f"Generating {args.num_games} ActionBigMoney games...")

    g = DominionGame()
    examples = []
    t0 = time.time()
    action_plays_total = 0
    games_with_actions = 0

    for game_num in range(args.num_games):
        s = g.get_initial_state()
        game_states = []
        action_plays_this_game = 0

        for _ in range(1000):
            if g.is_terminal(s):
                break
            valid = g.get_valid_moves(s)
            action = action_big_money(
                s, valid, PLAY_OFFSET, BUY_OFFSET, DONE_SELECTING,
                END_ACTIONS, END_BUYS, DECLINE_REACTION,
                PROVINCE_ID, GOLD_ID, SILVER_ID,
            )
            # Track action card plays (not treasures, not buy/end actions)
            if PLAY_OFFSET <= action < BUY_OFFSET:
                cid = action - PLAY_OFFSET
                if cid not in [COPPER.id, SILVER.id, GOLD.id]:
                    action_plays_this_game += 1

            game_states.append((
                s.get_canonical_form().to_tensor(),
                action,
                s.current_player,
            ))
            s = g.get_next_state(s, action)

        if action_plays_this_game > 0:
            games_with_actions += 1
            action_plays_total += action_plays_this_game

        # Get final outcomes
        r0 = g.get_reward(s, 0)
        r1 = g.get_reward(s, 1)

        for (tensor, action, player) in game_states:
            outcome = r0 if player == 0 else r1
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            policy[action] = 1.0
            examples.append((
                tensor,
                policy,
                float(outcome),
                0.0,
                np.zeros(12, dtype=np.float32),
            ))

        if (game_num + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"Games: {game_num + 1}/{args.num_games}  "
                  f"Examples: {len(examples)}  "
                  f"Games w/ action plays: {games_with_actions}  "
                  f"Elapsed: {elapsed:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(max_size=200000)
    buf.add_examples(examples)
    buf.save(Path(args.output))

    avg_action_plays = action_plays_total / args.num_games
    print(f"\nDone: {len(buf)} examples from {args.num_games} games → {args.output}")
    print(f"Games with action card plays: {games_with_actions}/{args.num_games}")
    print(f"Avg action plays per game: {avg_action_plays:.1f}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
