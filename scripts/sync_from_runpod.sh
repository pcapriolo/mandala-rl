#!/bin/bash
# Sync training data from RunPod to local for dashboard monitoring.
# Uses SSH/SCP directly (no HTTP file server needed).
#
# Usage:
#   ./scripts/sync_from_runpod.sh          # dashboard data only (fast, every 30s)
#   ./scripts/sync_from_runpod.sh 60       # custom interval
#   SYNC_CHECKPOINTS=1 ./scripts/sync_from_runpod.sh   # also sync checkpoint files

REMOTE="root@38.147.83.11"
PORT="17226"
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o ServerAliveInterval=5"
SCP="scp $SSH_OPTS -P $PORT -i $KEY -q"
SSH="ssh $SSH_OPTS -p $PORT -i $KEY"
REMOTE_BASE="/workspace/mandala-rl/data"
LOCAL_BASE="data"
INTERVAL="${1:-30}"

mkdir -p "$LOCAL_BASE/logs" "$LOCAL_BASE/replays" "$LOCAL_BASE/checkpoints"
mkdir -p "$LOCAL_BASE/lost_cities/logs" "$LOCAL_BASE/lost_cities/replays" "$LOCAL_BASE/lost_cities/checkpoints"

sync_elo() {
    local remote_path="$1" local_file="$2"
    local tmp="/tmp/elo_remote_$$.json"
    $SCP "$REMOTE:$remote_path" "$tmp" 2>/dev/null || { rm -f "$tmp"; return 0; }
    local remote_count=$(grep -c "iter_" "$tmp" 2>/dev/null || echo 0)
    local local_count=$(grep -c "iter_" "$local_file" 2>/dev/null || echo 0)
    if [ "$remote_count" -ge "$local_count" ]; then
        mv "$tmp" "$local_file"
    else
        rm -f "$tmp"
    fi
}

echo "Syncing from RunPod ($REMOTE:$PORT) every ${INTERVAL}s..."

while true; do
    # Small dashboard files (heartbeats, losses, elo)
    $SCP "$REMOTE:$REMOTE_BASE/heartbeat.json" "$LOCAL_BASE/heartbeat.json" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/eval_heartbeat.json" "$LOCAL_BASE/eval_heartbeat.json" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/losses.jsonl" "$LOCAL_BASE/losses.jsonl" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/lost_cities/heartbeat.json" "$LOCAL_BASE/lost_cities/heartbeat.json" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/lost_cities/eval_heartbeat.json" "$LOCAL_BASE/lost_cities/eval_heartbeat.json" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/lost_cities/losses.jsonl" "$LOCAL_BASE/lost_cities/losses.jsonl" 2>/dev/null || true

    sync_elo "$REMOTE_BASE/elo_ratings.json" "$LOCAL_BASE/elo_ratings.json"
    sync_elo "$REMOTE_BASE/lost_cities/elo_ratings.json" "$LOCAL_BASE/lost_cities/elo_ratings.json"

    # TensorBoard events
    $SCP "$REMOTE:$REMOTE_BASE/logs/events.*" "$LOCAL_BASE/logs/" 2>/dev/null || true
    $SCP "$REMOTE:$REMOTE_BASE/lost_cities/logs/events.*" "$LOCAL_BASE/lost_cities/logs/" 2>/dev/null || true

    # Checkpoints (optional — large files, only when SYNC_CHECKPOINTS=1)
    if [ "${SYNC_CHECKPOINTS:-0}" = "1" ]; then
        for dir_pair in "checkpoints:checkpoints" "lost_cities/checkpoints:lost_cities/checkpoints"; do
            remote_dir="${dir_pair%%:*}"
            local_dir="${dir_pair##*:}"
            remote_files=$($SSH "$REMOTE" "ls $REMOTE_BASE/$remote_dir/model_iter_*.pt 2>/dev/null" 2>/dev/null) || continue
            for rp in $remote_files; do
                fn=$(basename "$rp")
                [ -f "$LOCAL_BASE/$local_dir/$fn" ] || $SCP "$REMOTE:$rp" "$LOCAL_BASE/$local_dir/$fn" 2>/dev/null || true
            done
        done
    fi

    echo "[$(date +%H:%M:%S)] Synced"
    sleep "$INTERVAL"
done
