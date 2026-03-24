#!/bin/bash
# Monk Wake-Up (every 2 hours)
# Runs via launchd. Collects training metrics, invokes Claude (sonnet) to:
#   - Assess training health across all channels (losses, gameplay, infra)
#   - Process CEO inbox instructions
#   - Fix/improve training if needed (code changes, deploy, restart)
#   - Post concise CEO update with metric trends
#   - Log with "watching" notes for continuity between runs
# Hard timeout: 10 minutes.

# launchd runs with minimal env — set PATH explicitly
export PATH="/Users/paulcapriolo/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/paulcapriolo}"

REPO_DIR="$HOME/GG/mandala-rl"

# Load secrets from .env (not tracked in git)
if [ -f "$REPO_DIR/.env" ]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi
LOSSES="$REPO_DIR/data/dominion/losses.jsonl"
MONITOR="$REPO_DIR/data/dominion/monitor.jsonl"
HOURLY_LOG="$REPO_DIR/data/monk_hourly.jsonl"
MONK_INBOX="$HOME/GG/GG_Monk_Inbox.md"
PROMPT_FILE="/tmp/monk_hourly_prompt.md"

ts=$(date +"%Y-%m-%d %H:%M:%S")
echo "=== Monk Hourly: $ts ==="

mkdir -p "$(dirname "$HOURLY_LOG")"

# --- Check Telegram for CEO replies ---
TG_BOT="${TG_BOT_TOKEN:?Set TG_BOT_TOKEN env var}"
TG_CHAT="${TG_CHAT_ID:?Set TG_CHAT_ID env var}"
TG_OFFSET_FILE="/tmp/monk_tg_offset"

LAST_OFFSET=0
[ -f "$TG_OFFSET_FILE" ] && LAST_OFFSET=$(cat "$TG_OFFSET_FILE")

python3 << TGEOF
import json, urllib.request, datetime

bot = "${TG_BOT}"
chat = "${TG_CHAT}"
offset = ${LAST_OFFSET}
inbox = "${MONK_INBOX}"

url = f"https://api.telegram.org/bot{bot}/getUpdates?offset={offset + 1}&timeout=1"
try:
    r = json.loads(urllib.request.urlopen(url, timeout=5).read())
    if not r.get("ok") or not r.get("result"):
        raise SystemExit(0)

    max_id = offset
    for u in r["result"]:
        uid = u["update_id"]
        if uid > max_id:
            max_id = uid
        msg = u.get("message", {})
        if str(msg.get("chat", {}).get("id")) != chat:
            continue
        text = msg.get("text", "").strip()
        if not text:
            continue
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f'\n### [NEW] {ts} — CEO via Telegram\nPriority: HIGH\n\n{text}\n'
        with open(inbox, "a") as f:
            f.write(entry)
        print(f"  Telegram inbox: {text[:80]}")

    with open("${TG_OFFSET_FILE}", "w") as f:
        f.write(str(max_id))
except Exception as e:
    print(f"  Telegram poll error: {e}")
TGEOF

# --- Sync fresh data from RunPod (non-fatal if unavailable) ---
echo "  Syncing data from RunPod..."
mkdir -p "$REPO_DIR/data/dominion"
scp -q -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P 26242 \
    root@38.147.83.30:/workspace/dominion_data/losses.jsonl \
    "$REPO_DIR/data/dominion/losses.jsonl" 2>/dev/null \
    && echo "  losses.jsonl synced" || echo "  WARNING: RunPod sync failed (pod may be restarting)"
scp -q -o ConnectTimeout=10 -i ~/.ssh/id_ed25519 -P 26242 \
    root@38.147.83.30:/workspace/dominion_data/monitor.jsonl \
    "$REPO_DIR/data/dominion/monitor.jsonl" 2>/dev/null || true

# --- Collect raw data ---

RECENT_LOSSES=""
if [ -f "$LOSSES" ]; then
    RECENT_LOSSES=$(grep -v '^\s*$' "$LOSSES" | tail -10)
fi

LATEST_MONITOR=""
if [ -f "$MONITOR" ]; then
    LATEST_MONITOR=$(tail -1 "$MONITOR")
fi

INBOX_NEW=0
if [ -f "$MONK_INBOX" ]; then
    INBOX_NEW=$(grep -c '^### \[NEW\]' "$MONK_INBOX" 2>/dev/null)
    INBOX_NEW=${INBOX_NEW:-0}
fi

PRIOR_LOGS=""
if [ -f "$HOURLY_LOG" ]; then
    PRIOR_LOGS=$(tail -10 "$HOURLY_LOG")
fi

RECENT_DEVLOG=""
DEVLOG="$REPO_DIR/DEVLOG.md"
if [ -f "$DEVLOG" ]; then
    # Last 5 entries only (grep ## DEVLOG headings, take last 5, read from first match)
    FIRST_LINE=$(grep -n '^## ' "$DEVLOG" | tail -5 | head -1 | cut -d: -f1)
    if [ -n "$FIRST_LINE" ]; then
        RECENT_DEVLOG=$(tail -n +"$FIRST_LINE" "$DEVLOG")
    fi
fi

# --- Write prompt to temp file (avoids heredoc escaping issues with JSON) ---

cat > "$PROMPT_FILE" << 'STATIC_EOF'
You are the Monk, waking up for your scheduled check. Be fast and focused. You have a 10-minute hard timeout.

## Step 1: Read your prior log
Your last hourly log entry (below) has a "watching" field — this is what past-you flagged for this run. Check those items first.

## Step 2: Assess training health
Compare the last 10 iterations (losses.jsonl below) against your prior check. Evaluate across all channels:

**Loss heads:**
- policy_loss: should decrease over time (network learning action structure)
- value_loss: must be nonzero (>0.001). Zero = value head blind, training stuck
- score_loss: auxiliary signal, should track value_loss

**Gameplay quality:**
- avg_provinces: THE key milestone. >0 means bot discovered province buying
- avg_treasures: Silver/Gold buying (prerequisite for provinces)
- avg_estates / avg_duchies: VP card awareness
- avg_buys: total economic activity per player
- avg_curses: should be ~0 (buying curses = degenerate)
- action_rate: are action cards being played?

**Game dynamics:**
- avg_len / avg_turns: 200 = all games hitting move cap (no natural endings)
- draw_rate: high draw rate + low value_loss = value head can't differentiate
- p0_wr: first-player win rate (should be ~0.5, strong bias = bug)

**Infrastructure:**
- Is training process alive? (check monitor.jsonl status + pid)
- Disk usage trending dangerously?
- GPU utilization reasonable?

Flag anything that is **stuck** (no change across 5+ iterations) or **regressing** (metric going wrong direction).

## Step 3: Process CEO inbox
Read ~/GG/GG_Monk_Inbox.md for [NEW] messages. Act on instructions, mark [ACKNOWLEDGED] or [DONE] with `> Monk:` reply.

**IMPORTANT:** For each [NEW] CEO message you process, also append your reply to /tmp/monk_ceo_replies.txt (one per line, plain text). This file is sent to the CEO via Telegram so they see your response on their phone. Keep each reply concise (1-2 sentences). Clear the file first by writing (not appending) your first reply, then append subsequent ones.

## Step 4: Take action if needed
If the CEO gave instructions, or if training is clearly stuck/broken, fix it:
- Make code changes, scp to RunPod, rebuild (`cd /root/mandala-dom && pip install -e .`), restart training
- Write a DEVLOG entry in ~/GG/mandala-rl/DEVLOG.md for any substantive change
- RunPod SSH: `ssh root@38.147.83.30 -p 26242 -i ~/.ssh/id_ed25519`
- RunPod repo: /root/mandala-dom, data: /workspace/dominion_data
- Config: Use `configs/dominion.yaml` directly (NEVER copy/sed to /tmp or /workspace — single source of truth)
- Don't change game rules to fix degenerate behavior. Fix training signal instead.

**NON-NEGOTIABLE CONFIG GUARD — DO NOT VIOLATE:**
- `disabled_basic_supply` MUST be `[0, 3, 4, 6, 16]` (Phase 0: Silver/Gold/Province ONLY).
- Estate (3) and Duchy (4) MUST remain disabled. DO NOT enable them. DO NOT change to Phase 1.
- If you see disabled_basic_supply as `[0, 6, 16]`, that is WRONG — fix it back to `[0, 3, 4, 6, 16]`.
- Only the CEO can authorize phase transitions. You are NOT the CEO.
- DO NOT modify disabled_basic_supply, max_action_cards, or big_money_force_rate for ANY reason.

If nothing needs fixing, skip this step entirely.

## Step 5: Post CEO update
Append a [NEW] message to ~/GG/GG_CEO_Inbox.md. Format:
```
### [NEW] YYYY-MM-DD HH:MM — Monk Check: <1-line summary>
Priority: LOW/MEDIUM/HIGH

**Iter X→Y** (since last check)
| Metric | Last Check | Now | Trend |
| ... | ... | ... | ... |

<2-3 sentence assessment. What changed, what matters, what to watch.>
```

## Step 6: Log for next run
Append ONE JSON line to ~/GG/mandala-rl/data/monk_hourly.jsonl:
```json
{"ts":"...","health":"OK|WARNING|CRITICAL","iter":N,"policy_loss":...,"value_loss":...,"avg_provinces":...,"avg_treasures":...,"avg_buys":...,"draw_rate":...,"avg_len":...,"action":"what you did or 'observation only'","watching":"specific things for next run to check"}
```
The "watching" field is your note to future-you. Be specific: "value_loss should be >0.01 by iter 85" not "watch value loss".
STATIC_EOF

# Append dynamic data
cat >> "$PROMPT_FILE" << DYNAMIC_EOF

## Current state ($(date +"%Y-%m-%d %H:%M"))

### Last 10 training iterations (losses.jsonl):
\`\`\`
${RECENT_LOSSES}
\`\`\`

### Latest infrastructure snapshot (monitor.jsonl):
\`\`\`
${LATEST_MONITOR}
\`\`\`

### Monk Inbox [NEW] messages: ${INBOX_NEW}

### Your last 10 hourly logs (monk_hourly.jsonl) — this is YOUR prior history:
\`\`\`
${PRIOR_LOGS}
\`\`\`

### Recent DEVLOG entries (last 5):
\`\`\`
${RECENT_DEVLOG}
\`\`\`
DYNAMIC_EOF

# --- Clear CEO replies file before Claude runs ---
> /tmp/monk_ceo_replies.txt

# --- Invoke Claude with 10-minute timeout ---
echo "  Invoking Claude for analysis..."
TIMEOUT=600  # 10 minutes

cd "$REPO_DIR"
cat "$PROMPT_FILE" | claude -p "Execute the Monk wake-up. All context and instructions are in stdin." \
    --model sonnet \
    --allowedTools "Read Edit Write Bash Glob Grep" \
    >> /tmp/monk-hourly.log 2>&1 &
CLAUDE_PID=$!

# Wait up to TIMEOUT seconds, then kill
ELAPSED=0
while kill -0 $CLAUDE_PID 2>/dev/null; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "  Claude TIMED OUT after ${TIMEOUT}s — killing PID $CLAUDE_PID"
        kill $CLAUDE_PID 2>/dev/null
        sleep 5
        kill -9 $CLAUDE_PID 2>/dev/null
        break
    fi
done
wait $CLAUDE_PID 2>/dev/null
EXIT_CODE=$?
echo "  Claude exited with code $EXIT_CODE at $(date +"%Y-%m-%d %H:%M:%S")"

# --- Telegram notification ---
TG_MSG="/tmp/monk_tg_msg.txt"

python3 << 'PYEOF'
import json
from html import escape

LOSSES = "/Users/paulcapriolo/GG/mandala-rl/data/dominion/losses.jsonl"
MONK_LOG = "/Users/paulcapriolo/GG/mandala-rl/data/monk_hourly.jsonl"
OUT = "/tmp/monk_tg_msg.txt"

try:
    # Metrics from losses.jsonl (always has all fields)
    with open(LOSSES) as f:
        loss_lines = [l.strip() for l in f if l.strip()]
    m = json.loads(loss_lines[-1])

    # Monk state from monk_hourly.jsonl (action/watching)
    monk = {}
    try:
        with open(MONK_LOG) as f:
            monk_lines = [l.strip() for l in f if l.strip()]
        monk = json.loads(monk_lines[-1])
    except Exception:
        pass

    health = monk.get("health", "OK")
    icon = {"OK": "\u2705", "WARNING": "\u26a0\ufe0f", "CRITICAL": "\U0001f6a8"}.get(health, "\u2753")
    it = m.get("iteration", "?")

    def f(v):
        """Format a number nicely."""
        try:
            v = float(v)
            if v == 0: return "0"
            if v < 0.01: return f"{v:.4f}"
            return f"{v:.2f}"
        except (TypeError, ValueError):
            return "0"

    def pct(v):
        try: return f"{float(v)*100:.0f}%"
        except (TypeError, ValueError): return "0%"

    msg = f"""{icon} <b>Monk \u2014 Iter {it}</b>

<pre>Policy    {f(m.get('policy'))}    Value   {f(m.get('value'))}
Draws     {pct(m.get('draw_rate'))}     Len     {f(m.get('avg_len'))}

Provinces {f(m.get('avg_provinces'))}    Duchies  {f(m.get('avg_duchies', 0))}
Estates   {f(m.get('avg_estates', 0))}    Curses   {f(m.get('avg_curses', 0))}
Silver/Au {f(m.get('avg_treasures'))}    Actions  {f(m.get('avg_action_buys', 0))}
Buys/plyr {f(m.get('avg_buys'))}    Score    {f(m.get('avg_score'))}
ActRate   {pct(m.get('action_rate'))}     ActUtil  {pct(m.get('action_utilization'))}</pre>"""

    # CEO replies (monk's responses to inbox messages)
    try:
        with open("/tmp/monk_ceo_replies.txt") as f:
            replies = f.read().strip()
        if replies:
            msg += f"\n\n\U0001f4ac <b>CEO Replies:</b>\n{escape(replies)}"
    except Exception:
        pass

    action = monk.get("action", "")
    if action and action not in ("none", "observation only", ""):
        msg += f"\n\n\U0001f527 <b>Changed:</b>\n{escape(action)}"

    watching = monk.get("watching", "")
    if watching:
        msg += f"\n\n\U0001f441 <b>Watching:</b>\n{escape(watching)}"

    msg += "\n\n<i>Reply to send instructions to the Monk</i>"

    with open(OUT, "w") as f:
        f.write(msg)
except Exception as e:
    with open(OUT, "w") as f:
        f.write(f"Monk check complete (error: {e})")
PYEOF

curl -s -X POST "https://api.telegram.org/bot${TG_BOT}/sendMessage" \
    -d chat_id="$TG_CHAT" \
    -d parse_mode=HTML \
    --data-urlencode text@"$TG_MSG" > /dev/null 2>&1

echo "=== Monk Hourly Complete ==="
