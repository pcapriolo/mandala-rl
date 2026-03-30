#!/usr/bin/env python3
"""Migrate a Dominion checkpoint from 218-channel flat-policy to 280-channel phase-aware factored-policy.

Copies trunk weights (conv_input zero-padded for new channels), value/score/belief heads directly.
Policy heads are initialized randomly (no clean mapping from flat→factored).

Usage:
    python scripts/migrate_dominion_checkpoint.py --input <old_checkpoint> --output <new_checkpoint> [--config configs/dominion.yaml]
"""
import argparse
import sys
import os
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandala_rl.network.model import MandalaNet


def migrate(input_path: str, output_path: str, config_path: str):
    print(f"Loading old checkpoint: {input_path}")
    old_ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    old_state = old_ckpt['model_state_dict']

    with open(config_path) as f:
        config = yaml.safe_load(f)

    net_cfg = config['network']
    new_input_channels = net_cfg['input_channels']
    old_input_channels = old_state['conv_input.weight'].shape[1]

    print(f"Old input channels: {old_input_channels}")
    print(f"New input channels: {new_input_channels}")
    print(f"Phase-aware policy: {net_cfg.get('phase_aware_policy', False)}")
    print(f"Factored policy: {net_cfg.get('factored_policy', False)}")

    # Create new model
    new_model = MandalaNet(
        input_channels=net_cfg['input_channels'],
        num_actions=net_cfg['num_actions'],
        num_res_blocks=net_cfg['num_res_blocks'],
        channels=net_cfg['channels'],
        belief_size=net_cfg.get('belief_size', 12),
        phase_aware_policy=net_cfg.get('phase_aware_policy', False),
        factored_policy=net_cfg.get('factored_policy', False),
    )
    new_state = new_model.state_dict()

    copied = 0
    padded = 0
    skipped = 0

    for key in new_state:
        if key in old_state:
            old_tensor = old_state[key]
            new_tensor = new_state[key]

            if old_tensor.shape == new_tensor.shape:
                new_state[key] = old_tensor
                copied += 1
            elif key == 'conv_input.weight' and old_tensor.shape[1] < new_tensor.shape[1]:
                # Zero-pad input conv: (out_ch, old_in, kH, kW) → (out_ch, new_in, kH, kW)
                new_state[key][:, :old_tensor.shape[1]] = old_tensor
                padded += 1
                print(f"  Padded {key}: {old_tensor.shape} → {new_tensor.shape}")
            else:
                # Shape mismatch (e.g., old fc_policy vs new phase heads) — keep random init
                skipped += 1
                print(f"  Skipped {key}: old {old_tensor.shape} != new {new_tensor.shape}")
        else:
            # New key not in old checkpoint (e.g., phase_policy_heads.*) — keep random init
            skipped += 1

    # Check for old keys not in new model
    old_only = set(old_state.keys()) - set(new_state.keys())
    if old_only:
        print(f"  Old keys not in new model (dropped): {sorted(old_only)}")

    new_model.load_state_dict(new_state)

    # Build new checkpoint
    new_ckpt = {
        'iteration': old_ckpt.get('iteration', 0),
        'total_games': old_ckpt.get('total_games', 0),
        'games_in_current_iteration': 0,
        'model_state_dict': new_model.state_dict(),
    }

    torch.save(new_ckpt, output_path)
    print(f"\nMigrated: {copied} copied, {padded} padded, {skipped} randomly initialized")
    print(f"Saved to: {output_path}")

    # Verify
    verify_model = MandalaNet(
        input_channels=net_cfg['input_channels'],
        num_actions=net_cfg['num_actions'],
        num_res_blocks=net_cfg['num_res_blocks'],
        channels=net_cfg['channels'],
        belief_size=net_cfg.get('belief_size', 12),
        phase_aware_policy=net_cfg.get('phase_aware_policy', False),
        factored_policy=net_cfg.get('factored_policy', False),
    )
    verify_model.load_state_dict(new_ckpt['model_state_dict'])

    # Quick forward pass test
    dummy = torch.randn(1, new_input_channels, 8, 8)
    policy, value, score, belief = verify_model(dummy)
    print(f"\nVerification forward pass:")
    print(f"  Policy: {policy.shape} (expected: [1, {net_cfg['num_actions']}])")
    print(f"  Value:  {value.shape} (expected: [1, 1])")
    print(f"  Score:  {score.shape} (expected: [1, 1])")
    print(f"  Belief: {belief.shape} (expected: [1, {net_cfg.get('belief_size', 12)}])")
    assert policy.shape == (1, net_cfg['num_actions'])
    assert value.shape == (1, 1)
    print("Verification passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate Dominion checkpoint to new architecture')
    parser.add_argument('--input', required=True, help='Path to old checkpoint')
    parser.add_argument('--output', required=True, help='Path for new checkpoint')
    parser.add_argument('--config', default='configs/dominion.yaml', help='Config file')
    args = parser.parse_args()
    migrate(args.input, args.output, args.config)
