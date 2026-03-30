#!/usr/bin/env python3
"""
Generate Dominion seed training data from Big Money heuristic self-play.

Big Money strategy: Province($8) > Gold($6) > Silver($3), always.
This bootstraps Silver/Gold/Province buying into the replay buffer so
self-play can refine from a non-degenerate starting policy.

Usage:
    python3 scripts/seed_dominion_bigmoney.py --num-games 300 --output /workspace/dominion_data/bm_seed.pkl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from mandala_rl.training.replay_buffer import ReplayBuffer


def big_money_action(valid, PLAY_OFFSET, BUY_OFFSET, COPPER_ID, SILVER_ID, GOLD_ID, PROVINCE_ID, END_ACTIONS, END_BUYS):
    """Return the Big Money action: play treasures, then Province > Gold > Silver."""
    # Play treasures first (Gold > Silver > Copper for coin efficiency)
    for cid in [GOLD_ID, SILVER_ID, COPPER_ID]:
        if valid[PLAY_OFFSET + cid]:
            return PLAY_OFFSET + cid
    # Buy by priority
    for a in [BUY_OFFSET + PROVINCE_ID, BUY_OFFSET + GOLD_ID, BUY_OFFSET + SILVER_ID]:
        if valid[a]:
            return a
    # End action or buy phase
    if valid[END_ACTIONS]:
        return END_ACTIONS
    if valid[END_BUYS]:
        return END_BUYS
    # Fallback: first valid action
    return int(np.where(valid > 0)[0][0])


def main():
    parser = argparse.ArgumentParser(description='Generate Dominion BigMoney seed data')
    parser.add_argument('--num-games', type=int, default=300)
    parser.add_argument('--output', type=str, default='/workspace/dominion_data/bm_seed.pkl')
    args = parser.parse_args()

    from dominion.game.engine import (
        DominionGame, PLAY_OFFSET, BUY_OFFSET,
        COPPER, SILVER, GOLD, PROVINCE,
        END_ACTIONS, END_BUYS, NUM_ACTIONS,
    )

    g = DominionGame()
    COPPER_ID = COPPER.id
    SILVER_ID = SILVER.id
    GOLD_ID = GOLD.id
    PROVINCE_ID = PROVINCE.id

    examples = []
    t0 = time.time()

    for game_num in range(args.num_games):
        s = g.get_initial_state()
        game_states = []  # list of (canonical_tensor, action, current_player)

        # Play game
        for _ in range(1000):
            if g.is_terminal(s):
                break
            valid = g.get_valid_moves(s)
            action = big_money_action(
                valid, PLAY_OFFSET, BUY_OFFSET,
                COPPER_ID, SILVER_ID, GOLD_ID, PROVINCE_ID,
                END_ACTIONS, END_BUYS,
            )
            # Record canonical state + action + player before taking the move
            game_states.append((
                s.get_canonical_form().to_tensor(),
                action,
                s.current_player,
            ))
            s = g.get_next_state(s, action)

        # Get final outcomes using VP margin
        vp0 = s.players[0].count_vp(s.kingdom_cards)
        vp1 = s.players[1].count_vp(s.kingdom_cards)
        margin = vp0 - vp1
        r0 = float(max(-1.0, min(1.0, margin / 5.0)))
        r1 = float(max(-1.0, min(1.0, -margin / 5.0)))

        # Build training examples
        for (tensor, action, player) in game_states:
            outcome = r0 if player == 0 else r1

            # Policy: one-hot on the Big Money action
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            policy[action] = 1.0

            examples.append((
                tensor,
                policy,
                float(outcome),
                0.0,                              # score_loss_target
                np.zeros(12, dtype=np.float32),   # belief (unused)
            ))

        if (game_num + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"Games: {game_num + 1}/{args.num_games}  Examples: {len(examples)}  "
                  f"Elapsed: {elapsed:.0f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    buf = ReplayBuffer(max_size=200000)
    buf.add_examples(examples)
    buf.save(Path(args.output))

    print(f"\nDone: {len(buf)} examples from {args.num_games} games → {args.output}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
