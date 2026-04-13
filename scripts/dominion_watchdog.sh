#!/bin/bash
SEED="/workspace/dominion_data/bm_seed_280ch_30k.pkl"
CLEAN="/workspace/dominion_data/checkpoints/model_iter_2122_nudged.pt"
LATEST="/workspace/dominion_data/checkpoints/model_latest.pt"
CONFIG="/root/mandala-dom/configs/dominion.yaml"
LOG="/root/train_dom.log"
echo "=== Dominion Watchdog ==="
echo "MANAGED BY RALEIGH WATCHDOG - DO NOT KILL" > /workspace/dominion_data/TRAINING_LOCK
while true; do
    if ! pgrep -f "seed-buffer" > /dev/null 2>&1; then
        echo "[$(date)] Restarting trainer..."
        pkill -f "train.py" 2>/dev/null
        sleep 3
        # Resume from model_latest.pt (dont revert)
        cd /root/mandala-dom
        PYTHONUNBUFFERED=1 nohup python3 scripts/train.py --config "$CONFIG" --resume "$LATEST" --seed-buffer "$SEED" > "$LOG" 2>&1 &
        echo "[$(date)] Started PID $!"
    fi
    sleep 120
done
