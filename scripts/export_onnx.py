#!/usr/bin/env python3
"""Export PyTorch model checkpoints to ONNX format for lightweight inference."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from mandala_rl.network.model import MandalaNet


def export_model(checkpoint_path, config_path, output_path):
    """Export a PyTorch checkpoint to ONNX."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    net_cfg = config['network']
    model = MandalaNet(
        input_channels=net_cfg.get('input_channels', 50),
        num_actions=net_cfg.get('num_actions', 256),
        num_res_blocks=net_cfg['num_res_blocks'],
        channels=net_cfg['channels'],
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy = torch.randn(1, net_cfg.get('input_channels', 50), 8, 8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=['state'],
        output_names=['policy', 'value', 'score', 'belief'],
        dynamic_axes={
            'state': {0: 'batch'}, 'policy': {0: 'batch'},
            'value': {0: 'batch'}, 'score': {0: 'batch'}, 'belief': {0: 'batch'},
        },
        opset_version=17,
    )

    # Save metadata alongside
    import json
    meta = {
        'iteration': checkpoint.get('iteration', 'unknown'),
        'total_games': checkpoint.get('total_games', 'unknown'),
        'source': str(checkpoint_path),
        'input_channels': net_cfg.get('input_channels', 50),
        'num_actions': net_cfg.get('num_actions', 256),
    }
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Exported: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    print(f"Metadata: {meta_path}")
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('checkpoint', help='Path to .pt checkpoint')
    parser.add_argument('--config', required=True, help='Config YAML')
    parser.add_argument('--output', required=True, help='Output .onnx path')
    args = parser.parse_args()
    export_model(args.checkpoint, args.config, args.output)
