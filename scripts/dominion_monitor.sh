#!/bin/bash
# Dominion Training Monitor
# Runs every 10 minutes via launchd. Checks training health and takes corrective actions.
#
# Health checks:
#   1. Is training process running?
#   2. Is iteration progressing? (heartbeat freshness)
#   3. Loss trends (policy/value loss direction)
#   4. Game quality (avg score, provinces bought, game length)
#   5. Disk space
#
# Actions:
#   - Restart training if process died
#   - Alert on stalled progress (no new iteration in 30 min)
#   - Log all metrics for trend analysis

REMOTE="root@38.147.83.30"
KEY="$HOME/.ssh/id_ed25519"
SSH_PORT=26242
SSH="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -i $KEY -p $SSH_PORT"
TRAIN_DIR="/root/mandala-dom"
DATA_DIR="/workspace/dominion_data"
CONFIG="/tmp/dominion_runpod.yaml"
LOCAL_LOG="$HOME/GG/mandala-rl/data/dominion/monitor.jsonl"

mkdir -p "$(dirname "$LOCAL_LOG")"

ts=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "=== Dominion Monitor: $ts ==="

# 1. Check SSH connectivity
if ! $SSH "$REMOTE" "echo ok" >/dev/null 2>&1; then
    echo "ALERT: Cannot reach RunPod ($REMOTE:$SSH_PORT)"
    echo "{\"ts\":\"$ts\",\"status\":\"unreachable\"}" >> "$LOCAL_LOG"
    exit 1
fi

# 2. Check if training process is running
train_pid=$($SSH "$REMOTE" "pgrep -f 'train.py.*dominion' | head -1" 2>/dev/null)
if [ -z "$train_pid" ]; then
    echo "WARNING: Training process not running. Attempting restart..."
    # Ensure config exists
    $SSH "$REMOTE" "test -f $CONFIG || (cd $TRAIN_DIR && sed 's|data/dominion|/workspace/dominion_data|g' configs/dominion.yaml > $CONFIG)" 2>/dev/null
    $SSH "$REMOTE" "cd $TRAIN_DIR && nohup python3 scripts/train.py --config $CONFIG --game dominion > /tmp/dominion_train.log 2>&1 &" 2>/dev/null
    sleep 5
    train_pid=$($SSH "$REMOTE" "pgrep -f 'train.py.*dominion' | head -1" 2>/dev/null)
    if [ -z "$train_pid" ]; then
        echo "ALERT: Failed to restart training!"
        echo "{\"ts\":\"$ts\",\"status\":\"restart_failed\"}" >> "$LOCAL_LOG"
        exit 1
    fi
    echo "Training restarted with PID $train_pid"
    echo "{\"ts\":\"$ts\",\"status\":\"restarted\",\"pid\":$train_pid}" >> "$LOCAL_LOG"
    exit 0
fi

# 3. Pull all metrics in a single SSH session using unique delimiters
metrics=$($SSH "$REMOTE" 'bash -s' << 'CMDS'
echo "<<<HEARTBEAT>>>"
cat /workspace/dominion_data/heartbeat.json 2>/dev/null || echo "{}"
echo ""
echo "<<<LOSSES>>>"
tail -5 /workspace/dominion_data/losses.jsonl 2>/dev/null || echo ""
echo ""
echo "<<<DISK>>>"
df -h /workspace | tail -1
echo "<<<GPU>>>"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A"
echo "<<<TRAINLOG>>>"
tail -3 /tmp/dominion_train.log 2>/dev/null || echo ""
echo "<<<END>>>"
CMDS
) 2>/dev/null

# Helper: extract section between delimiters
extract() {
    echo "$metrics" | sed -n "/<<<$1>>>/,/<<</{/<<<$1>>>/d;/<<</d;p;}"
}

# Parse heartbeat
heartbeat=$(extract HEARTBEAT)
hb_iter=$(echo "$heartbeat" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('iteration',0))" 2>/dev/null || echo 0)
hb_phase=$(echo "$heartbeat" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('phase','unknown'))" 2>/dev/null || echo "unknown")
hb_game=$(echo "$heartbeat" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('game',0))" 2>/dev/null || echo 0)
hb_total=$(echo "$heartbeat" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_games',0))" 2>/dev/null || echo 0)
hb_ts=$(echo "$heartbeat" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('timestamp',0))" 2>/dev/null || echo 0)

# Parse last loss entry (filter empty lines before tail)
losses=$(extract LOSSES | grep -v '^\s*$' | tail -1)
if [ -n "$losses" ] && echo "$losses" | python3 -c "import json,sys; json.loads(sys.stdin.readline())" 2>/dev/null; then
    total_loss=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('total',0):.4f}\")" 2>/dev/null || echo "N/A")
    policy_loss=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('policy',0):.4f}\")" 2>/dev/null || echo "N/A")
    value_loss=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('value',0):.4f}\")" 2>/dev/null || echo "N/A")
    avg_score=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('avg_score',0):.1f}\")" 2>/dev/null || echo "N/A")
    avg_provinces=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('avg_provinces',0):.2f}\")" 2>/dev/null || echo "N/A")
    avg_len=$(echo "$losses" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(f\"{d.get('avg_len',0):.0f}\")" 2>/dev/null || echo "N/A")
else
    total_loss="N/A"; policy_loss="N/A"; value_loss="N/A"
    avg_score="N/A"; avg_provinces="N/A"; avg_len="N/A"
fi

# Parse disk
disk=$(extract DISK | head -1)
disk_avail=$(echo "$disk" | awk '{print $4}')
disk_pct=$(echo "$disk" | awk '{print $5}')

# Parse GPU
gpu=$(extract GPU | head -1)

# Check heartbeat freshness
now=$(date +%s)
hb_age="N/A"
stalled="no"
if [ "$hb_ts" != "0" ]; then
    hb_ts_int=$(printf "%.0f" "$hb_ts" 2>/dev/null || echo 0)
    if [ "$hb_ts_int" -gt 0 ]; then
        hb_age=$(( now - hb_ts_int ))
        if [ "$hb_age" -gt 1800 ]; then
            stalled="yes"
            echo "ALERT: Training stalled — no heartbeat update in ${hb_age}s"
        fi
    fi
fi

# Print summary
echo "  Iteration: $hb_iter | Phase: $hb_phase | Game: $hb_game/$hb_total"
echo "  Loss: total=$total_loss policy=$policy_loss value=$value_loss"
echo "  Quality: avg_score=$avg_score provinces=$avg_provinces avg_len=$avg_len"
echo "  Disk: $disk_avail free ($disk_pct used) | GPU: $gpu"
echo "  Heartbeat age: ${hb_age}s | Stalled: $stalled | PID: $train_pid"

# Log to JSONL
echo "{\"ts\":\"$ts\",\"status\":\"running\",\"iter\":$hb_iter,\"phase\":\"$hb_phase\",\"game\":$hb_game,\"total_games\":$hb_total,\"total_loss\":\"$total_loss\",\"policy_loss\":\"$policy_loss\",\"value_loss\":\"$value_loss\",\"avg_score\":\"$avg_score\",\"avg_provinces\":\"$avg_provinces\",\"avg_len\":\"$avg_len\",\"disk_avail\":\"$disk_avail\",\"disk_pct\":\"$disk_pct\",\"gpu\":\"$gpu\",\"stalled\":\"$stalled\",\"pid\":$train_pid}" >> "$LOCAL_LOG"

# Alerts
if [ "$stalled" = "yes" ]; then
    echo ""
    echo "  >>> STALLED: Consider checking /tmp/dominion_train.log on RunPod"
fi
