#!/bin/bash
# Sync training data from RunPod to local for dashboard monitoring.
# Syncs text data (losses, heartbeats) via SSH and latest checkpoints via SCP.
#
# Usage:
#   ./scripts/sync_from_runpod.sh          # sync every 60s
#   ./scripts/sync_from_runpod.sh 30       # custom interval

# launchd runs with minimal env — set PATH explicitly
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/paulcapriolo}"

REMOTE="root@38.147.83.30"
KEY="$HOME/.ssh/id_ed25519"
SSH_PORT=26242
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -i $KEY -p $SSH_PORT"
SCP="scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -i $KEY -P $SSH_PORT"
REMOTE_BASE="/workspace/mandala-rl/data"
REMOTE_DOM="/workspace/dominion_data"
LOCAL_BASE="data"
INTERVAL="${1:-60}"
PUSH_INTERVAL=300  # Push to git at most every 5 minutes
LAST_PUSH=0
GIT_SSH_COMMAND="ssh -i $KEY -o StrictHostKeyChecking=accept-new"
export GIT_SSH_COMMAND

mkdir -p "$LOCAL_BASE/checkpoints" "$LOCAL_BASE/lost_cities/checkpoints" "$LOCAL_BASE/dominion/checkpoints"

sync_all() {
    # 1. Sync text data in a single SSH session
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
echo "===HEARTBEAT_LC==="
cat /workspace/mandala-rl/data/lost_cities/heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===EVAL_HB_LC==="
cat /workspace/mandala-rl/data/lost_cities/eval_heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===LOSSES_LC==="
cat /workspace/mandala-rl/data/lost_cities/losses.jsonl 2>/dev/null || echo ""
echo
echo "===HEARTBEAT_DOM==="
cat /workspace/dominion_data/heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===EVAL_HB_DOM==="
cat /workspace/dominion_data/eval_heartbeat.json 2>/dev/null || echo "{}"
echo
echo "===LOSSES_DOM==="
cat /workspace/dominion_data/losses.jsonl 2>/dev/null || echo ""
echo
echo "===ELO_DOM==="
cat /workspace/dominion_data/elo_ratings.json 2>/dev/null || echo "{}"
echo
echo "===RUNPOD_HEALTH==="
PID=$(pgrep -f "train.py.*dominion" | head -1)
echo "PID=${PID:-NONE}"
DISK_PCT=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
DISK_AVAIL=$(df -h / | tail -1 | awk '{print $4}')
echo "DISK_PCT=${DISK_PCT}"
echo "DISK_AVAIL=${DISK_AVAIL}"
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "N/A")
echo "GPU_UTIL=${GPU_UTIL}"
echo "GPU_MEM=${GPU_MEM}"
BUF_SIZE=$(du -sh /workspace/dominion_data/checkpoints/buffer_latest.pkl 2>/dev/null | cut -f1 || echo "N/A")
echo "BUF_SIZE=${BUF_SIZE}"
LAST_ERROR=$(grep -E "Error|Traceback|RuntimeError|killed|OOM" /tmp/dominion_train.log 2>/dev/null | tail -1 || echo "")
echo "LAST_ERROR=${LAST_ERROR}"
echo
echo "===DONE==="
CMDS
    ) 2>/dev/null

    if [ -z "$output" ]; then
        echo "[$(date +%H:%M:%S)] SSH failed — skipping this cycle"
        return
    fi

    extract_section() {
        echo "$output" | sed -n "/^===$1===/,/^===/p" | grep -v '^==='
    }

    # Guard append-only files against truncation from SSH drops
    safe_write() {
        local content="$1"
        local dest="$2"
        local new_lines
        new_lines=$(echo "$content" | grep -c .)
        local old_lines=0
        [ -f "$dest" ] && old_lines=$(wc -l < "$dest" | tr -d ' ')
        if [ "$new_lines" -lt "$old_lines" ]; then
            echo "[$(date +%H:%M:%S)] WARN: skipping $dest write ($new_lines < $old_lines lines, likely truncated)"
            return
        fi
        echo "$content" > "$dest"
    }

    extract_section "HEARTBEAT_M" > "$LOCAL_BASE/heartbeat.json"
    extract_section "EVAL_HB_M" > "$LOCAL_BASE/eval_heartbeat.json"
    safe_write "$(extract_section "LOSSES_M")" "$LOCAL_BASE/losses.jsonl"
    extract_section "HEARTBEAT_LC" > "$LOCAL_BASE/lost_cities/heartbeat.json"
    extract_section "EVAL_HB_LC" > "$LOCAL_BASE/lost_cities/eval_heartbeat.json"
    safe_write "$(extract_section "LOSSES_LC")" "$LOCAL_BASE/lost_cities/losses.jsonl"
    extract_section "HEARTBEAT_DOM" > "$LOCAL_BASE/dominion/heartbeat.json"
    extract_section "EVAL_HB_DOM" > "$LOCAL_BASE/dominion/eval_heartbeat.json"
    safe_write "$(extract_section "LOSSES_DOM")" "$LOCAL_BASE/dominion/losses.jsonl"
    extract_section "ELO_DOM" > "$LOCAL_BASE/dominion/elo_ratings.json"

    # 2. Per-iteration Telegram notification
    local ITER_STATE="/tmp/mandala-sync-last-iter.json"
    local dom_losses="$LOCAL_BASE/dominion/losses.jsonl"
    if [ -f "$dom_losses" ]; then
        local current_iter
        current_iter=$(tail -1 "$dom_losses" | python3 -c "import sys,json; print(json.load(sys.stdin).get('iteration',0))" 2>/dev/null || echo 0)
        local last_iter=0
        [ -f "$ITER_STATE" ] && last_iter=$(python3 -c "import sys,json; print(json.load(open('$ITER_STATE')).get('iteration',0))" 2>/dev/null || echo 0)

        if [ "$current_iter" -gt "$last_iter" ] 2>/dev/null; then
            # Load TG credentials
            if [ -f "$HOME/GG/mandala-rl/.env" ]; then
                set -a; source "$HOME/GG/mandala-rl/.env"; set +a
            fi
            if [ -n "$TG_BOT_TOKEN" ] && [ -n "$TG_CHAT_ID" ]; then
                # Parse RunPod health from SSH output
                local health_section
                health_section=$(extract_section "RUNPOD_HEALTH")
                parse_health() { echo "$health_section" | grep "^$1=" | head -1 | cut -d= -f2-; }
                local h_pid=$(parse_health PID)
                local h_disk_pct=$(parse_health DISK_PCT)
                local h_disk_avail=$(parse_health DISK_AVAIL)
                local h_gpu_util=$(parse_health GPU_UTIL)
                local h_gpu_mem=$(parse_health GPU_MEM)
                local h_buf_size=$(parse_health BUF_SIZE)
                local h_last_error=$(parse_health LAST_ERROR)

                # Get latest, prev, and baseline loss lines
                local latest_loss=$(tail -1 "$dom_losses")
                local prev_loss=$(tail -2 "$dom_losses" | head -1)
                local baseline_loss=$(grep '"iteration": 250' "$dom_losses" | tail -1)

                SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
                python3 "$SCRIPT_DIR/tg_notify.py" \
                    "$TG_BOT_TOKEN" "$TG_CHAT_ID" \
                    "$h_pid" "$h_disk_pct" "$h_disk_avail" "$h_gpu_util" "$h_gpu_mem" "$h_buf_size" \
                    "" "$h_last_error" \
                    "$latest_loss" "$prev_loss" "$baseline_loss" \
                    "" "" "" "" 2>/dev/null

                echo "[$(date +%H:%M:%S)] Telegram sent (iter $current_iter)"
            fi
            # Update state with current line
            tail -1 "$dom_losses" > "$ITER_STATE"
        fi
    fi

    # 3. Sync latest checkpoints via SCP
    local new=0
    for game_dir in "" "lost_cities/" "dominion/"; do
        if [ "$game_dir" = "dominion/" ]; then
            local remote_ckpt="/workspace/dominion_data/checkpoints/model_latest.pt"
        else
            local remote_ckpt="$REMOTE_BASE/${game_dir}checkpoints/model_latest.pt"
        fi
        local local_ckpt="$LOCAL_BASE/${game_dir}checkpoints/model_latest.pt"
        local label="${game_dir:-mandala/}"

        # Compare remote iteration with local
        local remote_iter
        remote_iter=$($SSH "$REMOTE" "python3 -c \"import torch; print(torch.load('$remote_ckpt', map_location='cpu', weights_only=False)['iteration'])\"" 2>/dev/null)
        local local_iter=0
        if [ -f "$local_ckpt" ]; then
            local_iter=$(python3 -c "import torch; print(torch.load('$local_ckpt', map_location='cpu', weights_only=False)['iteration'])" 2>/dev/null || echo 0)
        fi

        if [ -n "$remote_iter" ] && [ "$remote_iter" != "$local_iter" ]; then
            $SCP "$REMOTE:$remote_ckpt" "$local_ckpt" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "  ${label}latest: iter $local_iter → $remote_iter"
                new=$((new + 1))
            fi
        fi
    done

    local m_lines=$(wc -l < "$LOCAL_BASE/losses.jsonl" 2>/dev/null | tr -d ' ')
    local lc_lines=$(wc -l < "$LOCAL_BASE/lost_cities/losses.jsonl" 2>/dev/null | tr -d ' ')
    local dom_lines=$(wc -l < "$LOCAL_BASE/dominion/losses.jsonl" 2>/dev/null | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Synced (M:${m_lines} LC:${lc_lines} DOM:${dom_lines} loss entries, ${new} checkpoints updated)"

    # Config drift detection (warn only, never auto-fix)
    local remote_dom_config local_dom_config
    remote_dom_config=$($SSH $REMOTE "cat /root/mandala-dom/configs/dominion.yaml" 2>/dev/null) || true
    local_dom_config=$(cat "$HOME/GG/mandala-rl/configs/dominion.yaml" 2>/dev/null) || true
    if [ -n "$remote_dom_config" ] && [ -n "$local_dom_config" ] && [ "$remote_dom_config" != "$local_dom_config" ]; then
        echo "[$(date +%H:%M:%S)] CONFIG DRIFT: RunPod dominion.yaml differs from repo. Run: ./scripts/sync_config_to_runpod.sh"
    fi

    # Auto-push data files to git for Railway deployment (throttled)
    local now
    now=$(date +%s)
    if [ $((now - LAST_PUSH)) -ge $PUSH_INTERVAL ]; then
        cd "$HOME/GG/mandala-rl"
        local data_files="data/dominion/losses.jsonl data/dominion/heartbeat.json data/dominion/elo_ratings.json"
        if ! git diff --quiet $data_files 2>/dev/null; then
            local dom_iter
            dom_iter=$(tail -1 "$LOCAL_BASE/dominion/losses.jsonl" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('iteration','?'))" 2>/dev/null || echo "$dom_lines")
            git add $data_files 2>/dev/null
            git commit -m "data: update dominion training data (iter $dom_iter)" 2>/dev/null
            # Rebase on remote first to handle PRs merged from other workspaces
            git pull --rebase origin main 2>/dev/null
            if git push origin main 2>&1; then
                LAST_PUSH=$now
                echo "[$(date +%H:%M:%S)] Pushed training data to git (iter $dom_iter)"
            else
                echo "[$(date +%H:%M:%S)] Git push failed"
            fi
        fi
    fi
}

echo "Syncing from RunPod ($REMOTE:$SSH_PORT) every ${INTERVAL}s..."
echo "Syncs: losses.jsonl, heartbeats, model_latest.pt (when iter changes)"

while true; do
    sync_all
    sleep "$INTERVAL"
done
