#!/bin/bash
# Setup script for RunPod / Vast.ai / any CUDA GPU instance.
# Run this on a fresh instance to get training going.
#
# Usage:
#   bash deploy_runpod.sh                          # Fresh start
#   bash deploy_runpod.sh --resume checkpoint.pt   # Resume from checkpoint
set -e

REPO_URL="${REPO_URL:-https://github.com/your-username/mandala-rl.git}"
BRANCH="${BRANCH:-main}"

echo "=== Mandala RL - GPU Training Setup ==="

# Clone if not already present
if [ ! -d "mandala-rl" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" mandala-rl
fi
cd mandala-rl

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Compiles C++ MCTS extension

# Verify setup
python3 -c "
import torch, mcts_cpp
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'C++ MCTS: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start training:"
echo "  python3 scripts/train.py --config configs/default.yaml"
echo ""
echo "Resume from checkpoint:"
echo "  python3 scripts/train.py --config configs/default.yaml --resume data/checkpoints/model_latest.pt"
echo ""
echo "Train Lost Cities:"
echo "  python3 scripts/train.py --config configs/lost_cities.yaml --game lost_cities"
