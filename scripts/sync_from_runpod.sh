#!/bin/bash
# Sync training data from RunPod to local for dashboard monitoring.
# Uses curl only (no wget dependency). Works on macOS and Linux.
#
# Usage:
#   ./scripts/sync_from_runpod.sh https://<pod-id>-8888.proxy.runpod.net
#
# On RunPod first:
#   cd /workspace/mandala-rl
#   nohup python3 -m http.server 8888 --directory data > fileserver.log 2>&1 &

set -e

RUNPOD_URL="${1:?Usage: $0 <runpod-data-url>}"
INTERVAL="${2:-30}"

mkdir -p data/logs data/replays data/checkpoints

# Fetch directory listing and extract filenames matching a pattern
sync_dir() {
    local url="$1" local_dir="$2" pattern="$3"
    local listing
    listing=$(curl -sf "$url" 2>/dev/null) || return 0
    echo "$listing" | grep -oE "href=\"[^\"]*${pattern}[^\"]*\"" | sed 's/href="//;s/"//' | while read -r filename; do
        if [ ! -f "${local_dir}/${filename}" ]; then
            curl -sf "${url}${filename}" -o "${local_dir}/${filename}" 2>/dev/null || true
        fi
    done
}

echo "Syncing from $RUNPOD_URL every ${INTERVAL}s..."
echo "Press Ctrl+C to stop"

while true; do
    # Elo ratings (1KB, always overwrite — RunPod has full history)
    curl -sf "$RUNPOD_URL/elo_ratings.json" -o data/elo_ratings.json 2>/dev/null || true

    # TensorBoard event files (only download new ones)
    sync_dir "$RUNPOD_URL/logs/" "data/logs" "events"

    # Game replays (only download new ones)
    sync_dir "$RUNPOD_URL/replays/" "data/replays" ".json"

    echo "[$(date +%H:%M:%S)] Synced"
    sleep "$INTERVAL"
done
