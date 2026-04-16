#!/bin/bash
# Sync training data from RunPod to local machine for dashboard
# Usage: ./scripts/sync_runpod.sh [interval_seconds]
#   Default interval: 60s

INTERVAL=${1:-60}
REMOTE="root@38.147.83.30"
PORT=26242
KEY="$HOME/.ssh/id_ed25519"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/data"
REMOTE_DIR="/workspace/mandala-rl/data"

SCP="scp -q -P $PORT -i $KEY"

echo "Syncing from $REMOTE:$PORT every ${INTERVAL}s"
echo "Local: $LOCAL"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    ts=$(date +%H:%M:%S)

    # Mandala
    $SCP "$REMOTE:$REMOTE_DIR/losses.jsonl" "$LOCAL/losses.jsonl" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/elo_ratings.json" "$LOCAL/elo_ratings.json" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/heartbeat.json" "$LOCAL/heartbeat.json" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/eval_heartbeat.json" "$LOCAL/eval_heartbeat.json" 2>/dev/null

    # Lost Cities
    $SCP "$REMOTE:$REMOTE_DIR/lost_cities/losses.jsonl" "$LOCAL/lost_cities/losses.jsonl" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/lost_cities/elo_ratings.json" "$LOCAL/lost_cities/elo_ratings.json" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/lost_cities/heartbeat.json" "$LOCAL/lost_cities/heartbeat.json" 2>/dev/null
    $SCP "$REMOTE:$REMOTE_DIR/lost_cities/eval_heartbeat.json" "$LOCAL/lost_cities/eval_heartbeat.json" 2>/dev/null

    # Dominion
    $SCP "$REMOTE:/workspace/dominion_data/losses.jsonl" "$LOCAL/dominion/losses.jsonl" 2>/dev/null
    $SCP "$REMOTE:/workspace/dominion_data/elo_ratings.json" "$LOCAL/dominion/elo_ratings.json" 2>/dev/null
    $SCP "$REMOTE:/workspace/dominion_data/heartbeat.json" "$LOCAL/dominion/heartbeat.json" 2>/dev/null
    $SCP "$REMOTE:/workspace/dominion_data/eval_heartbeat.json" "$LOCAL/dominion/eval_heartbeat.json" 2>/dev/null

    echo "[$ts] Synced"
    sleep "$INTERVAL"
done
