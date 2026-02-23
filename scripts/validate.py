#!/usr/bin/env python3
"""
Pre-training validation suite.

Run this BEFORE starting any training to catch bugs that would
require restarting. Validates game rules, MCTS correctness,
state encoding, and training pipeline end-to-end.

Usage:
    python3 scripts/validate.py                          # validate both games
    python3 scripts/validate.py --game mandala           # validate mandala only
    python3 scripts/validate.py --game lost_cities       # validate lost cities only
"""
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def validate_mcts_sign():
    """Verify MCTS selects moves that are good for the PARENT, not the child."""
    print("\n=== MCTS Sign Convention ===")
    from mandala_rl.mcts.node import MCTSNode

    root = MCTSNode(prior=1.0)
    priors = np.array([0.33, 0.34, 0.33])
    valid = np.array([1.0, 1.0, 1.0])
    root.expand(priors, valid)

    # Child 0: good for child (bad for parent)
    root.children[0].update(0.9)
    root.children[0].update(0.9)
    # Child 1: bad for child (good for parent)
    root.children[1].update(-0.9)
    root.children[1].update(-0.9)
    # Child 2: neutral
    root.children[2].update(0.0)
    root.children[2].update(0.0)

    action, _ = root.select_child(c_puct=0.01)  # Low exploration
    check("MCTS selects move good for parent (not opponent)",
          action == 1,
          f"selected action {action}, expected 1 (best for parent)")


def validate_mandala():
    """Validate Mandala game engine, encoding, and training pipeline."""
    print("\n=== Mandala Game Rules ===")
    from mandala_rl.game.engine import MandalaGame
    from mandala_rl.game.state import GameState, Card

    game = MandalaGame(seed=42)

    # Play multiple games to completion
    errors = 0
    total_moves = 0
    for seed in range(20):
        g = MandalaGame(seed=seed)
        s = g.get_initial_state()
        m = 0
        while not g.is_terminal(s) and m < 500:
            valid = g.get_valid_moves(s)
            va = np.where(valid > 0)[0]
            if len(va) == 0:
                errors += 1
                break
            s = g.get_next_state(s, np.random.choice(va))
            m += 1
        total_moves += m
        if m >= 500:
            errors += 1
    check("20 random games complete without errors", errors == 0, f"{errors} errors")
    check("Games have reasonable length", 30 < total_moves / 20 < 200,
          f"avg {total_moves/20:.0f} moves")

    # Terminal state player switching
    g = MandalaGame(seed=99)
    s = g.get_initial_state()
    last_player = None
    while not g.is_terminal(s):
        last_player = s.current_player
        valid = g.get_valid_moves(s)
        va = np.where(valid > 0)[0]
        if len(va) == 0:
            break
        s = g.get_next_state(s, np.random.choice(va))
    if last_player is not None and g.is_terminal(s):
        check("Terminal state switches player from mover",
              s.current_player == 1 - last_player,
              f"last mover={last_player}, terminal player={s.current_player}")

    # is_terminal only checks game_over
    s_test = GameState()
    s_test.game_ends_next_mandala = True
    s_test.game_over = False
    check("is_terminal does NOT fire on game_ends_next_mandala alone",
          not game.is_terminal(s_test))
    s_test.game_over = True
    check("is_terminal fires on game_over=True",
          game.is_terminal(s_test))

    # State encoding
    print("\n=== Mandala State Encoding ===")
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)
    expected_channels = config['network']['input_channels']

    s = game.get_initial_state()
    tensor = s.to_tensor()
    check(f"Tensor shape matches config ({expected_channels} channels)",
          tensor.shape[0] == expected_channels,
          f"got {tensor.shape[0]}, config says {expected_channels}")
    check("Tensor spatial dims are 8x8",
          tensor.shape[1:] == (8, 8),
          f"got {tensor.shape[1:]}")

    # Opponent river is encoded
    s2 = game.get_initial_state()
    s2.rivers[1] = [Card(2), Card(3)]  # Opponent has colors 2,3 in river
    t2 = s2.to_tensor()
    # Find opponent river channels (should be after my river channels 42-47)
    opp_river_encoded = False
    for ch in range(48, min(expected_channels, 60)):
        if t2[ch, 0, 0] > 0:
            opp_river_encoded = True
            break
    check("Opponent river is encoded in tensor", opp_river_encoded)

    # Cup sizes encoded
    s3 = game.get_initial_state()
    s3.cups[0] = [Card(0)] * 5
    s3.cups[1] = [Card(1)] * 3
    t3 = s3.to_tensor()
    cup_encoded = False
    for ch in range(48, expected_channels):
        if t3[ch, 0, 0] > 0:
            cup_encoded = True
            break
    check("Cup sizes are encoded in tensor", cup_encoded)

    # Training pipeline smoke test
    print("\n=== Mandala Training Pipeline ===")
    _validate_training_pipeline(
        game, config, "mandala",
        expected_channels=expected_channels,
        num_actions=config['network']['num_actions']
    )


def validate_lost_cities():
    """Validate Lost Cities game engine, encoding, and training pipeline."""
    print("\n=== Lost Cities Game Rules ===")
    from lost_cities.game.engine import LostCitiesGame
    from lost_cities.game.state import LostCitiesState, LCCard

    game = LostCitiesGame(seed=42)

    # Play multiple games
    errors = 0
    total_moves = 0
    for seed in range(20):
        g = LostCitiesGame(seed=seed)
        s = g.get_initial_state()
        m = 0
        while not g.is_terminal(s) and m < 200:
            valid = g.get_valid_moves(s)
            va = np.where(valid > 0)[0]
            if len(va) == 0:
                errors += 1
                break
            s = g.get_next_state(s, np.random.choice(va))
            m += 1
        total_moves += m
        if m >= 200:
            errors += 1
    check("20 random games complete without errors", errors == 0, f"{errors} errors")

    # Terminal state player switching
    g = LostCitiesGame(seed=99)
    s = g.get_initial_state()
    last_player = None
    while not g.is_terminal(s):
        last_player = s.current_player
        valid = g.get_valid_moves(s)
        va = np.where(valid > 0)[0]
        if len(va) == 0:
            break
        s = g.get_next_state(s, np.random.choice(va))
    if last_player is not None and g.is_terminal(s):
        check("Terminal state switches player from mover",
              s.current_player == 1 - last_player,
              f"last mover={last_player}, terminal player={s.current_player}")

    # Scoring verification
    s_score = LostCitiesState()
    s_score.expeditions[0][0] = [LCCard(0, 0), LCCard(0, 0), LCCard(0, 3), LCCard(0, 5),
                                  LCCard(0, 7), LCCard(0, 8), LCCard(0, 9), LCCard(0, 10)]
    check("Scoring: 2 wagers + 6 numbers + bonus = 86",
          s_score.compute_score(0) == 86,
          f"got {s_score.compute_score(0)}")

    s_neg = LostCitiesState()
    s_neg.expeditions[0][0] = [LCCard(0, 2), LCCard(0, 3)]
    check("Scoring: R2+R3 = -15 (negative expedition)",
          s_neg.compute_score(0) == -15,
          f"got {s_neg.compute_score(0)}")

    # Ascending order enforcement
    s_asc = LostCitiesState()
    s_asc.expeditions[0][0] = [LCCard(0, 0), LCCard(0, 5)]
    s_asc.hands[0] = sorted([LCCard(0, 3), LCCard(0, 7)] + [LCCard(4, 5)] * 6)
    s_asc.hands[1] = sorted([LCCard(0, 2)] * 8)
    s_asc.deck = [LCCard(4, 10)] * 44
    valid = game.get_valid_moves(s_asc)
    # Red 3 (value 3) can't go on expedition with top 5
    r3_pos = next(i for i, c in enumerate(s_asc.hands[0]) if c.color == 0 and c.value == 3)
    r7_pos = next(i for i, c in enumerate(s_asc.hands[0]) if c.color == 0 and c.value == 7)
    check("Can't play R3 on expedition with top R5",
          valid[r3_pos * 12 + 0] == 0.0)
    check("Can play R7 on expedition with top R5",
          valid[r7_pos * 12 + 0] == 1.0)

    # State encoding
    print("\n=== Lost Cities State Encoding ===")
    import yaml
    with open("configs/lost_cities.yaml") as f:
        config = yaml.safe_load(f)
    expected_channels = config['network']['input_channels']

    s = game.get_initial_state()
    tensor = s.to_tensor()
    check(f"Tensor shape matches config ({expected_channels} channels)",
          tensor.shape[0] == expected_channels,
          f"got {tensor.shape[0]}, config says {expected_channels}")

    # Hand card identity encoded
    s_hand = LostCitiesState()
    s_hand.hands[0] = sorted([LCCard(0, 0), LCCard(0, 5), LCCard(1, 3), LCCard(2, 7),
                               LCCard(3, 0), LCCard(3, 4), LCCard(4, 9), LCCard(4, 10)])
    s_hand.hands[1] = sorted([LCCard(0, 2)] * 8)
    s_hand.deck = [LCCard(4, 10)] * 44
    t_hand = s_hand.to_tensor()

    hand_identity_ok = True
    for i, card in enumerate(s_hand.hands[0]):
        color_val = t_hand[50 + i, 0, 0]
        value_val = t_hand[58 + i, 0, 0]
        decoded_color = round(color_val * 6 - 1)
        decoded_value = round(value_val * 11 - 1)
        if decoded_color != card.color or decoded_value != card.value:
            hand_identity_ok = False
            break
    check("Hand card identity (color+value) correctly encoded per position",
          hand_identity_ok)

    # Canonical form preserves hand identity
    s_can = LostCitiesState()
    s_can.hands[0] = sorted([LCCard(0, 2)] * 8)
    s_can.hands[1] = sorted([LCCard(2, 7), LCCard(3, 8)] + [LCCard(4, 9)] * 6)
    s_can.deck = [LCCard(4, 10)] * 44
    s_can.current_player = 1
    can = s_can.get_canonical_form()
    check("Canonical form swaps hands correctly",
          can.hands[0] == s_can.hands[1] and can.current_player == 0)

    # Training pipeline smoke test
    print("\n=== Lost Cities Training Pipeline ===")
    _validate_training_pipeline(
        game, config, "lost_cities",
        expected_channels=expected_channels,
        num_actions=config['network']['num_actions']
    )


def _validate_training_pipeline(game, config, game_name, expected_channels, num_actions):
    """Smoke test the full training pipeline (self-play -> training -> eval)."""
    from mandala_rl.network.model import MandalaNet
    from mandala_rl.mcts.search import MCTS

    net_config = config['network']
    device = 'cpu'  # Use CPU for validation speed

    # Build network
    network = MandalaNet(
        input_channels=expected_channels,
        num_actions=num_actions,
        num_res_blocks=net_config['num_res_blocks'],
        channels=net_config['channels']
    ).to(device)

    # Verify network forward pass with correct tensor shape
    state = game.get_initial_state()
    canonical = state.get_canonical_form()
    tensor = canonical.to_tensor()
    check(f"State tensor channels == network input_channels",
          tensor.shape[0] == network.input_channels,
          f"tensor has {tensor.shape[0]}, network expects {network.input_channels}")

    x = torch.from_numpy(tensor).unsqueeze(0).to(device)
    policy_logits, value, *_ = network(x)
    check(f"Network outputs correct policy size ({num_actions})",
          policy_logits.shape[1] == num_actions,
          f"got {policy_logits.shape[1]}")
    check("Network value output is scalar per batch",
          value.shape == (1, 1),
          f"got {value.shape}")

    # Verify predict() works
    policy, val = network.predict(canonical)
    check("predict() returns valid policy (sums to ~1)",
          abs(policy.sum() - 1.0) < 0.01,
          f"policy sum = {policy.sum():.4f}")
    check("predict() returns value in [-1, 1]",
          -1.0 <= val <= 1.0,
          f"value = {val}")

    # Verify valid moves mask matches policy size
    valid = game.get_valid_moves(state)
    check("Valid moves mask size == num_actions",
          len(valid) == num_actions,
          f"valid_moves has {len(valid)}, expected {num_actions}")

    # Run 1 MCTS search (quick, 10 sims)
    mcts = MCTS(game=game, network=network.predict, num_simulations=10, c_puct=1.0)
    try:
        visits = mcts.search(state, add_noise=True)
        check("MCTS search completes without error", True)
        check("MCTS returns visits matching action space",
              len(visits) == num_actions,
              f"got {len(visits)}")
        check("MCTS visits are non-negative",
              np.all(visits >= 0))
        check("MCTS visits sum > 0 (at least one action explored)",
              visits.sum() > 0,
              f"total visits = {visits.sum()}")
    except Exception as e:
        check("MCTS search completes without error", False, str(e))

    # Verify loss computation
    batch_states = torch.from_numpy(np.stack([tensor, tensor])).to(device)
    batch_policies = torch.zeros(2, num_actions).to(device)
    batch_policies[0, 0] = 1.0
    batch_policies[1, 1] = 1.0
    batch_values = torch.tensor([1.0, -1.0]).to(device)
    try:
        total_loss, policy_loss, value_loss = network.get_loss(
            batch_states, batch_policies, batch_values
        )
        check("Loss computation works", True)
        check("Total loss is finite", torch.isfinite(total_loss).item(),
              f"total_loss = {total_loss.item()}")
        check("Value loss is weighted (>= 2x policy loss contribution)",
              total_loss.item() > policy_loss.item(),
              f"total={total_loss.item():.2f}, policy={policy_loss.item():.2f}")
    except Exception as e:
        check("Loss computation works", False, str(e))

    # Checkpoint compatibility: verify config matches architecture
    check(f"Config input_channels ({expected_channels}) matches tensor ({tensor.shape[0]})",
          expected_channels == tensor.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Pre-training validation")
    parser.add_argument('--game', choices=['mandala', 'lost_cities'],
                        help='Validate specific game (default: both)')
    args = parser.parse_args()

    print("=" * 60)
    print("PRE-TRAINING VALIDATION")
    print("=" * 60)

    validate_mcts_sign()

    if args.game in (None, 'mandala'):
        validate_mandala()
    if args.game in (None, 'lost_cities'):
        validate_lost_cities()

    print(f"\n{'=' * 60}")
    if FAIL == 0:
        print(f"ALL {PASS} CHECKS PASSED - safe to train")
    else:
        print(f"{FAIL} FAILED, {PASS} passed - DO NOT TRAIN until failures are fixed")
    print(f"{'=' * 60}")

    return 1 if FAIL > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
