# Mandala RL

Self-play reinforcement learning system for training a strong Mandala bot using MCTS + neural networks (AlphaZero-style).

## Overview

This project trains a Mandala bot through 100% self-play using:
- Monte Carlo Tree Search (MCTS) for game tree exploration
- Policy/Value neural network for position evaluation
- Self-play data generation
- Iterative training and evaluation
- Deterministic Elo-based evaluation ladder

Optimized for minimal GPU compute on Apple Silicon (MPS backend).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the bot
python scripts/train.py --config configs/default.yaml

# Evaluate against previous versions
python scripts/evaluate.py --checkpoint data/checkpoints/model_latest.pt
```

## Project Structure

```
mandala_rl/
├── game/          # Mandala game engine and rules
├── mcts/          # Monte Carlo Tree Search implementation
├── network/       # Policy/Value neural network
├── selfplay/      # Self-play game generation
├── training/      # Training loop and replay buffer
└── evaluation/    # Elo rating and arena evaluation
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with MPS support
- Apple Silicon Mac (M1/M2/M3)
