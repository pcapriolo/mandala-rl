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
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -i $KEY -p $SSH_PORT"
SCP="scp -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -i $KEY -P $SSH_PORT"
REMOTE_BASE="/workspace/mandala-rl/data"
REMOTE_DOM="/workspace/dominion_data"
LOCAL_BASE="data"
INTERVAL="${1:-60}"

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

    extract_section "HEARTBEAT_M" > "$LOCAL_BASE/heartbeat.json"
    extract_section "EVAL_HB_M" > "$LOCAL_BASE/eval_heartbeat.json"
    extract_section "LOSSES_M" > "$LOCAL_BASE/losses.jsonl"
    extract_section "HEARTBEAT_LC" > "$LOCAL_BASE/lost_cities/heartbeat.json"
    extract_section "EVAL_HB_LC" > "$LOCAL_BASE/lost_cities/eval_heartbeat.json"
    extract_section "LOSSES_LC" > "$LOCAL_BASE/lost_cities/losses.jsonl"
    extract_section "HEARTBEAT_DOM" > "$LOCAL_BASE/dominion/heartbeat.json"
    extract_section "EVAL_HB_DOM" > "$LOCAL_BASE/dominion/eval_heartbeat.json"
    extract_section "LOSSES_DOM" > "$LOCAL_BASE/dominion/losses.jsonl"
    extract_section "ELO_DOM" > "$LOCAL_BASE/dominion/elo_ratings.json"

    # 2. Sync latest checkpoints via SCP
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

    # Auto-push losses.jsonl to git for Railway deployment
    cd "$HOME/GG/mandala-rl"
    if ! git diff --quiet data/dominion/losses.jsonl 2>/dev/null; then
        git add data/dominion/losses.jsonl
        git commit -m "data: update dominion losses (iter $dom_lines)"
        git push origin main 2>/dev/null && echo "[$(date +%H:%M:%S)] Pushed losses.jsonl (iter $dom_lines)" || echo "[$(date +%H:%M:%S)] Push failed"
    fi
}

echo "Syncing from RunPod ($REMOTE:$SSH_PORT) every ${INTERVAL}s..."
echo "Syncs: losses.jsonl, heartbeats, model_latest.pt (when iter changes)"

while true; do
    sync_all
    sleep "$INTERVAL"
done
