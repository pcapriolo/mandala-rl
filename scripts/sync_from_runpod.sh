#!/bin/bash
# Sync training data from RunPod to local for dashboard monitoring.
# Uses direct SSH (supports SCP for binary files).
#
# Usage:
#   ./scripts/sync_from_runpod.sh          # sync everything every 30s
#   ./scripts/sync_from_runpod.sh 60       # custom interval

REMOTE="root@38.147.83.11"
KEY="$HOME/.ssh/id_ed25519"
SSH_PORT=17226
SSH="ssh -o StrictHostKeyChecking=accept-new -i $KEY -p $SSH_PORT"
SCP="scp -o StrictHostKeyChecking=accept-new -i $KEY -P $SSH_PORT"
REMOTE_BASE="/workspace/mandala-rl/data"
LOCAL_BASE="data"
INTERVAL="${1:-30}"

mkdir -p "$LOCAL_BASE/logs" "$LOCAL_BASE/replays" "$LOCAL_BASE/checkpoints"
mkdir -p "$LOCAL_BASE/lost_cities/logs" "$LOCAL_BASE/lost_cities/replays" "$LOCAL_BASE/lost_cities/checkpoints"

download_checkpoint() {
    # Download binary checkpoint via SSH cat pipe (SCP breaks on this connection)
    local remote_path="$1" local_path="$2"
    $SSH -o ServerAliveInterval=5 "$REMOTE" "cat '$remote_path'" > "$local_path" 2>/dev/null
    local rc=$?
    # Verify file is non-empty (failed transfers produce 0-byte files)
    if [ $rc -ne 0 ] || [ ! -s "$local_path" ]; then
        rm -f "$local_path"
        return 1
    fi
    return 0
}

sync_all() {
    # Collect all text data in a single SSH session
    local output
    output=$($SSH "$REMOTE" 'bash -s' << 'CMDS'
echo "===HEARTBEAT_M==="
cat /workspace/mandala-rl/data/heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===EVAL_HB_M==="
cat /workspace/mandala-rl/data/eval_heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===LOSSES_M==="
cat /workspace/mandala-rl/data/losses.jsonl 2>/dev/null || echo ""
echo
echo "===ELO_M==="
echo "{}"
echo "===HEARTBEAT_LC==="
cat /workspace/mandala-rl/data/lost_cities/heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===EVAL_HB_LC==="
cat /workspace/mandala-rl/data/lost_cities/eval_heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===LOSSES_LC==="
cat /workspace/mandala-rl/data/lost_cities/losses.jsonl 2>/dev/null || echo ""
echo
echo "===ELO_LC==="
echo "{}"
echo "===CHECKPOINTS_M==="
for f in /workspace/mandala-rl/data/checkpoints/model_iter_*.pt; do [ -f "$f" ] && basename "$f"; done
echo "===CHECKPOINTS_LC==="
for f in /workspace/mandala-rl/data/lost_cities/checkpoints/model_iter_*.pt; do [ -f "$f" ] && basename "$f"; done
echo "===DONE==="
CMDS
    ) 2>/dev/null

    if [ -z "$output" ]; then
        echo "[$(date +%H:%M:%S)] SSH failed — skipping this cycle"
        return
    fi

    # Extract content between section markers
    extract_section() {
        echo "$output" | sed -n "/^===$1===/,/^===/p" | grep -v '^==='
    }

    extract_section "HEARTBEAT_M" > "$LOCAL_BASE/heartbeat.json"
    extract_section "EVAL_HB_M" > "$LOCAL_BASE/eval_heartbeat.json"
    extract_section "LOSSES_M" > "$LOCAL_BASE/losses.jsonl"
    # Elo files are managed locally by eval_daemon — don't overwrite
    # extract_section "ELO_M" > "$LOCAL_BASE/elo_ratings.json"
    extract_section "HEARTBEAT_LC" > "$LOCAL_BASE/lost_cities/heartbeat.json"
    extract_section "EVAL_HB_LC" > "$LOCAL_BASE/lost_cities/eval_heartbeat.json"
    extract_section "LOSSES_LC" > "$LOCAL_BASE/lost_cities/losses.jsonl"
    # extract_section "ELO_LC" > "$LOCAL_BASE/lost_cities/elo_ratings.json"

    # Extract checkpoint filenames
    local m_checkpoints lc_checkpoints
    m_checkpoints=$(extract_section "CHECKPOINTS_M" | grep -oE 'model_iter_[0-9]+\.pt' | sort -u)
    lc_checkpoints=$(extract_section "CHECKPOINTS_LC" | grep -oE 'model_iter_[0-9]+\.pt' | sort -u)

    local new_m=0 new_lc=0
    for fn in $m_checkpoints; do
        if [ ! -f "$LOCAL_BASE/checkpoints/$fn" ]; then
            echo "  Downloading Mandala $fn..."
            download_checkpoint "$REMOTE_BASE/checkpoints/$fn" "$LOCAL_BASE/checkpoints/$fn"
            if [ $? -eq 0 ]; then
                new_m=$((new_m + 1))
            else
                echo "  Failed to download $fn"
            fi
        fi
    done

    for fn in $lc_checkpoints; do
        if [ ! -f "$LOCAL_BASE/lost_cities/checkpoints/$fn" ]; then
            echo "  Downloading LC $fn..."
            download_checkpoint "$REMOTE_BASE/lost_cities/checkpoints/$fn" "$LOCAL_BASE/lost_cities/checkpoints/$fn"
            if [ $? -eq 0 ]; then
                new_lc=$((new_lc + 1))
            else
                echo "  Failed to download $fn"
            fi
        fi
    done

    local m_total=$(ls "$LOCAL_BASE/checkpoints/model_iter_"*.pt 2>/dev/null | wc -l | tr -d ' ')
    local lc_total=$(ls "$LOCAL_BASE/lost_cities/checkpoints/model_iter_"*.pt 2>/dev/null | wc -l | tr -d ' ')
    local m_remote=$(echo "$m_checkpoints" | grep -c 'model_iter_' 2>/dev/null || echo 0)
    local lc_remote=$(echo "$lc_checkpoints" | grep -c 'model_iter_' 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] Synced (M:${m_total}/${m_remote}ckpts LC:${lc_total}/${lc_remote}ckpts, +${new_m}/+${new_lc} new)"
}

echo "Syncing from RunPod ($REMOTE:$SSH_PORT) every ${INTERVAL}s..."
echo "Checkpoints are always synced — local archive is source of truth."

while true; do
    sync_all
    sleep "$INTERVAL"
done
