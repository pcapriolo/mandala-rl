#!/bin/bash
# Monk Hourly Wake-Up
# Runs every hour via launchd. Pulls training metrics, checks inbox, posts CEO update.
#
# Flow:
#   1. Pull latest metrics from monitor.jsonl
#   2. Check GG_Monk_Inbox.md for [NEW] items (flag for human attention)
#   3. Post hourly report to GG_CEO_Inbox.md
#   4. Append machine-readable entry to monk_hourly.jsonl

# launchd runs with minimal env — set PATH explicitly
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/paulcapriolo}"

GG_DIR="$HOME/GG"
REPO_DIR="$GG_DIR/mandala-rl"
CEO_INBOX="$GG_DIR/GG_CEO_Inbox.md"
MONK_INBOX="$GG_DIR/GG_Monk_Inbox.md"
MONITOR_LOG="$REPO_DIR/data/dominion/monitor.jsonl"
HOURLY_LOG="$REPO_DIR/data/monk_hourly.jsonl"

mkdir -p "$(dirname "$HOURLY_LOG")"

ts=$(date +"%Y-%m-%d %H:%M")
ts_full=$(date +"%Y-%m-%d %H:%M:%S")

echo "=== Monk Hourly: $ts ==="

# --- 1. Pull latest training metrics ---

latest=""
if [ -f "$MONITOR_LOG" ]; then
    latest=$(tail -1 "$MONITOR_LOG" 2>/dev/null)
fi

if [ -z "$latest" ]; then
    status="no_data"
    iter="N/A"; phase="N/A"; total_games="N/A"
    total_loss="N/A"; policy_loss="N/A"; value_loss="N/A"
    avg_score="N/A"; avg_provinces="N/A"; avg_len="N/A"
    disk_avail="N/A"; disk_pct="N/A"; gpu="N/A"
    pid="N/A"; stalled="N/A"
else
    # Parse JSON fields with python3
    field() { echo "$latest" | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(d.get('$1','N/A'))" 2>/dev/null || echo "N/A"; }
    status=$(field status)
    iter=$(field iter)
    phase=$(field phase)
    total_games=$(field total_games)
    total_loss=$(field total_loss)
    policy_loss=$(field policy_loss)
    value_loss=$(field value_loss)
    avg_score=$(field avg_score)
    avg_provinces=$(field avg_provinces)
    avg_len=$(field avg_len)
    disk_avail=$(field disk_avail)
    disk_pct=$(field disk_pct)
    gpu=$(field gpu)
    pid=$(field pid)
    stalled=$(field stalled)
fi

# Compute iteration rate from monitor log (iters in last hour)
iter_rate="N/A"
if [ -f "$MONITOR_LOG" ] && [ "$iter" != "N/A" ]; then
    one_hour_ago=$(date -v-1H +"%Y-%m-%d %H:%M" 2>/dev/null || date -d "1 hour ago" +"%Y-%m-%d %H:%M" 2>/dev/null)
    if [ -n "$one_hour_ago" ]; then
        old_iter=$(grep -F "$one_hour_ago" "$MONITOR_LOG" 2>/dev/null | head -1 | python3 -c "import json,sys; d=json.loads(sys.stdin.readline()); print(d.get('iter',0))" 2>/dev/null)
        if [ -n "$old_iter" ] && [ "$old_iter" != "0" ] && [ "$iter" != "N/A" ]; then
            iter_rate=$(( iter - old_iter ))
        fi
    fi
fi

# Check for pending inbox items
inbox_new=0
if [ -f "$MONK_INBOX" ]; then
    inbox_new=$(grep -c '^### \[NEW\]' "$MONK_INBOX" 2>/dev/null || echo 0)
fi

echo "  Status: $status | Iter: $iter | Games: $total_games"
echo "  Loss: total=$total_loss policy=$policy_loss value=$value_loss"
echo "  Quality: score=$avg_score provinces=$avg_provinces len=$avg_len"
echo "  Disk: $disk_avail ($disk_pct) | GPU: $gpu | Stalled: $stalled"
echo "  Inbox [NEW]: $inbox_new"

# --- 2. Health assessment ---

health="OK"
alerts=""

if [ "$status" = "unreachable" ] || [ "$status" = "restart_failed" ]; then
    health="CRITICAL"
    alerts="${alerts}\n- RunPod unreachable or restart failed"
fi

if [ "$stalled" = "yes" ]; then
    health="WARNING"
    alerts="${alerts}\n- Training stalled (no heartbeat update in >30 min)"
fi

if [ "$disk_pct" != "N/A" ]; then
    pct_num=$(echo "$disk_pct" | tr -d '%')
    if [ "$pct_num" -gt 85 ] 2>/dev/null; then
        health="WARNING"
        alerts="${alerts}\n- Disk usage high: $disk_pct"
    fi
fi

# --- 3. Post CEO update ---

# Build the report
report="### [NEW] $ts — Hourly Dominion Training Report
Priority: LOW

**Health: $health**

| Metric | Value |
|--------|-------|
| Iteration | $iter |
| Phase | $phase |
| Total Games | $total_games |
| Policy Loss | $policy_loss |
| Value Loss | $value_loss |
| Total Loss | $total_loss |
| Avg Score | $avg_score |
| Avg Provinces | $avg_provinces |
| Avg Game Length | $avg_len |
| GPU | $gpu |
| Disk | $disk_avail free ($disk_pct used) |
| PID | $pid |
| Stalled | $stalled |"

if [ -n "$alerts" ]; then
    report="$report

**Alerts:**$(echo -e "$alerts")"
fi

if [ "$inbox_new" -gt 0 ]; then
    report="$report

**Note:** $inbox_new unread message(s) in Monk Inbox awaiting processing."
fi

# Append to CEO inbox (before the closing, or at end)
echo "" >> "$CEO_INBOX"
echo "$report" >> "$CEO_INBOX"

echo "  CEO inbox updated."

# --- 4. Machine-readable log ---

echo "{\"ts\":\"$ts_full\",\"health\":\"$health\",\"status\":\"$status\",\"iter\":\"$iter\",\"phase\":\"$phase\",\"total_games\":\"$total_games\",\"total_loss\":\"$total_loss\",\"policy_loss\":\"$policy_loss\",\"value_loss\":\"$value_loss\",\"avg_score\":\"$avg_score\",\"avg_provinces\":\"$avg_provinces\",\"avg_len\":\"$avg_len\",\"disk_avail\":\"$disk_avail\",\"disk_pct\":\"$disk_pct\",\"gpu\":\"$gpu\",\"stalled\":\"$stalled\",\"pid\":\"$pid\",\"inbox_new\":$inbox_new}" >> "$HOURLY_LOG"

echo "  Logged to $HOURLY_LOG"
echo "=== Monk Hourly Complete ==="
