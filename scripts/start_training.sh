#!/bin/bash
# Start all training processes on RunPod.
# Run this once after pod restart — it handles everything.
#
# Usage: ./scripts/start_training.sh

set -e
cd /workspace/mandala-rl

# Install deps (fresh container after restart)
echo "Installing dependencies..."
pip install --no-build-isolation -e . 2>&1 | tail -3

# Kill any stale processes
pkill -f 'scripts/train.py' 2>/dev/null || true
pkill -f 'scripts/eval_daemon.py' 2>/dev/null || true
sleep 1

# Fix corrupted model_latest.pt files (0-byte from interrupted saves)
for game_dir in data data/lost_cities; do
    latest="$game_dir/checkpoints/model_latest.pt"
    if [ -f "$latest" ] && [ ! -s "$latest" ]; then
        # Find most recent valid iter checkpoint
        recovery=$(ls -t "$game_dir/checkpoints/model_iter_"*.pt 2>/dev/null | head -1)
        if [ -n "$recovery" ]; then
            echo "Recovering $latest from $recovery"
            cp "$recovery" "$latest"
        else
            echo "WARNING: $latest is corrupt and no iter checkpoint found"
        fi
    fi
done

# Start Mandala training
if [ -f data/checkpoints/model_latest.pt ]; then
    echo "Starting Mandala training (resuming)..."
    nohup python3 scripts/train.py --config configs/default.yaml \
        --resume data/checkpoints/model_latest.pt > /tmp/mandala_train.log 2>&1 &
else
    echo "Starting Mandala training (from scratch)..."
    nohup python3 scripts/train.py --config configs/default.yaml > /tmp/mandala_train.log 2>&1 &
fi
echo "  PID: $!"

# Start Lost Cities training
if [ -f data/lost_cities/checkpoints/model_latest.pt ]; then
    echo "Starting Lost Cities training (resuming)..."
    nohup python3 scripts/train.py --config configs/lost_cities.yaml \
        --resume data/lost_cities/checkpoints/model_latest.pt > /tmp/lc_train.log 2>&1 &
else
    echo "Starting Lost Cities training (from scratch)..."
    nohup python3 scripts/train.py --config configs/lost_cities.yaml > /tmp/lc_train.log 2>&1 &
fi
echo "  PID: $!"

# Start BOTH eval daemons
echo "Starting Mandala eval daemon..."
nohup python3 scripts/eval_daemon.py --config configs/default.yaml > /tmp/eval_daemon.log 2>&1 &
echo "  PID: $!"

echo "Starting Lost Cities eval daemon..."
nohup python3 scripts/eval_daemon.py --config configs/lost_cities.yaml > /tmp/eval_daemon_lc.log 2>&1 &
echo "  PID: $!"

sleep 3
echo ""
echo "=== All processes ==="
ps aux | grep -E 'train\.py|eval_daemon' | grep -v grep
echo ""
echo "=== Memory ==="
free -h | head -2
echo ""
echo "Done. Logs: /tmp/{mandala_train,lc_train,eval_daemon,eval_daemon_lc}.log"
