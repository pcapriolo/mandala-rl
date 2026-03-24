#!/usr/bin/env python3
"""Monk Daemon — conversational Telegram bot + 15-min scheduled reports.

Uses Claude (via claude CLI) to have natural conversations about training.
Claude gets full context: RunPod status, training metrics, trends, project knowledge.
Can take actions on RunPod (restart, kill, logs, disk cleanup) when asked naturally.
Scheduled status reports every 15 minutes with proactive infra actions.
"""

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.parse

# Force unbuffered output for launchd
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Config ──────────────────────────────────────────────────
REPO_DIR = os.path.expanduser("~/GG/mandala-rl")
ENV_FILE = os.path.join(REPO_DIR, ".env")
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH_PORT = "26242"
SSH_HOST = "root@38.147.83.30"
RUNPOD_REPO = "/root/mandala-dom"
CLAUDE_CLI = os.path.expanduser("~/.local/bin/claude")
STATE_FILE = "/tmp/monk-daemon-state.json"
HISTORY_FILE = "/tmp/monk-daemon-history.json"
REPORT_INTERVAL = 900   # 15 minutes
POLL_TIMEOUT = 30       # Telegram long-poll seconds
MAX_HISTORY = 20        # conversation turns to keep


def load_env():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
    return os.environ["TG_BOT_TOKEN"], os.environ["TG_CHAT_ID"]


# ── Telegram API ────────────────────────────────────────────

def tg_send(bot, chat, text):
    # Telegram max message length is 4096
    if len(text) > 4000:
        text = text[:4000] + "\n\n<i>(truncated)</i>"
    data = urllib.parse.urlencode({
        "chat_id": chat, "parse_mode": "HTML", "text": text
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{bot}/sendMessage", data
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        log(f"tg_send OK ({len(text)} chars)")
        return True
    except urllib.error.HTTPError as e:
        # If HTML parse fails, retry without parse_mode
        if e.code == 400:
            log(f"tg_send HTML failed, retrying plain text")
            data = urllib.parse.urlencode({
                "chat_id": chat, "text": text
            }).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{bot}/sendMessage", data
            )
            try:
                urllib.request.urlopen(req, timeout=10)
                log(f"tg_send OK plain ({len(text)} chars)")
                return True
            except Exception as e2:
                log(f"tg_send plain failed: {e2}")
                return False
        log(f"tg_send failed: {e}")
        return False
    except Exception as e:
        log(f"tg_send failed: {e}")
        return False


def tg_get_updates(bot, offset, timeout=30):
    url = (f"https://api.telegram.org/bot{bot}/getUpdates"
           f"?offset={offset}&timeout={timeout}")
    try:
        resp = urllib.request.urlopen(url, timeout=timeout + 10)
        return json.loads(resp.read())
    except Exception as e:
        log(f"getUpdates failed: {e}")
        return None


# ── SSH helpers ─────────────────────────────────────────────

def ssh_run(cmd, timeout=20):
    try:
        start = time.time()
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", "-i", SSH_KEY, "-p", SSH_PORT, SSH_HOST, cmd],
            capture_output=True, text=True, timeout=timeout
        )
        log(f"SSH {time.time()-start:.1f}s rc={r.returncode}")
        return r.stdout
    except subprocess.TimeoutExpired:
        log("SSH TIMEOUT")
        return "SSH_TIMEOUT"
    except Exception as e:
        log(f"SSH FAILED: {e}")
        return f"SSH_FAILED: {e}"


COLLECT_SCRIPT = r"""
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
CPU_PCT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | tr -d '%us,' || echo "N/A")
RAM_USED=$(free -h | awk '/^Mem:/{print $3}')
RAM_TOTAL=$(free -h | awk '/^Mem:/{print $2}')
echo "CPU_PCT=${CPU_PCT}"
echo "RAM_USED=${RAM_USED}"
echo "RAM_TOTAL=${RAM_TOTAL}"
LOSSES=/workspace/dominion_data/losses.jsonl
if [ -f "$LOSSES" ]; then
    echo "LATEST_LOSS=$(tail -1 $LOSSES)"
    echo "PREV_LOSS=$(tail -2 $LOSSES | head -1)"
    BASELINE_LINE=$(grep '"iteration": 250' $LOSSES | tail -1)
    echo "BASELINE=${BASELINE_LINE}"
    TOTAL=$(wc -l < $LOSSES | tr -d ' ')
    echo "TOTAL_ITERS=${TOTAL}"
    FR_BASELINE=$(python3 -c "
import json
lines = open('$LOSSES').readlines()
if lines:
    cur_fr = json.loads(lines[-1]).get('force_rate')
    prev = None
    for line in lines:
        d = json.loads(line)
        if d.get('force_rate') == cur_fr:
            break
        prev = line.strip()
    if prev:
        print(prev)
" 2>/dev/null)
    echo "FR_BASELINE=${FR_BASELINE}"
fi
CKPT_DIR=/root/mandala-dom/data/dominion/checkpoints
CKPT_COUNT=$(ls "$CKPT_DIR"/model_iter_*.pt 2>/dev/null | wc -l | tr -d " ")
GAME_CKPT_COUNT=$(ls "$CKPT_DIR"/model_latest_game*.pt 2>/dev/null | wc -l | tr -d " ")
BUF_SIZE=$(du -sh "$CKPT_DIR"/buffer_latest.pkl 2>/dev/null | cut -f1 || echo "N/A")
TMP_FILES=$(ls "$CKPT_DIR"/*.tmp 2>/dev/null | wc -l | tr -d " ")
echo "CKPT_COUNT=${CKPT_COUNT}"
echo "GAME_CKPT_COUNT=${GAME_CKPT_COUNT}"
echo "BUF_SIZE=${BUF_SIZE}"
echo "TMP_FILES=${TMP_FILES}"
LAST_ERROR=$(grep -E "Error|Traceback|RuntimeError|killed|OOM" /tmp/dominion_train.log 2>/dev/null | tail -1 || echo "")
echo "LAST_ERROR=${LAST_ERROR}"
BUF_EXAMPLES=$(grep -o 'buffer: [0-9]*' /tmp/dominion_train.log 2>/dev/null | tail -1 | grep -o '[0-9]*' || echo "0")
BUF_CAPACITY=$(grep replay_buffer_size /root/mandala-dom/configs/dominion.yaml 2>/dev/null | grep -o '[0-9]*' | head -1 || echo "100000")
echo "BUF_EXAMPLES=${BUF_EXAMPLES}"
echo "BUF_CAPACITY=${BUF_CAPACITY}"
"""


def collect_status():
    raw = ssh_run(COLLECT_SCRIPT, timeout=25)
    if not raw or raw.startswith("SSH_"):
        return {"_ssh_failed": True}
    d = {"_ssh_failed": False}
    for line in raw.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            d[k] = v
    return d


def format_status_context(s):
    """Format status data as plain text context for Claude."""
    if s.get("_ssh_failed"):
        return "SSH to RunPod FAILED — pod may be down."

    pid = s.get("PID", "NONE")
    lines = []
    lines.append(f"Process: {'RUNNING (PID ' + pid + ')' if pid not in ('NONE','') else 'DEAD'}")
    lines.append(f"Disk: {s.get('DISK_PCT','?')}% used ({s.get('DISK_AVAIL','?')} free)")
    lines.append(f"GPU: {s.get('GPU_UTIL','?')}% util, {s.get('GPU_MEM','?')}")
    lines.append(f"Buffer on disk: {s.get('BUF_SIZE','?')}")

    buf_ex = s.get("BUF_EXAMPLES", "0")
    buf_cap = s.get("BUF_CAPACITY", "100000")
    try:
        fill = int(buf_ex) / int(buf_cap) * 100
        lines.append(f"Replay buffer: {int(buf_ex)//1000}K/{int(buf_cap)//1000}K ({fill:.0f}% full)")
    except:
        pass

    latest_raw = s.get("LATEST_LOSS", "")
    prev_raw = s.get("PREV_LOSS", "")
    baseline_raw = s.get("BASELINE", "")

    if latest_raw:
        m = json.loads(latest_raw)
        lines.append(f"\nCurrent iteration: {m.get('iteration','?')}")
        lines.append(f"Policy loss: {m.get('policy','?')}")
        lines.append(f"Value loss: {m.get('value','?')}")
        lines.append(f"Score loss: {m.get('score','?')}")
        lines.append(f"Belief loss: {m.get('belief','?')}")
        lines.append(f"Avg provinces: {m.get('avg_provinces','?')}")
        lines.append(f"Avg duchies: {m.get('avg_duchies','?')}")
        lines.append(f"Avg score: {m.get('avg_score','?')}")
        lines.append(f"Avg treasures: {m.get('avg_treasures','?')}")
        lines.append(f"Avg action buys: {m.get('avg_action_buys','?')}")
        lines.append(f"Avg buys: {m.get('avg_buys','?')}")
        lines.append(f"Avg curses: {m.get('avg_curses','?')}")
        lines.append(f"Action rate: {m.get('action_rate','?')}")
        lines.append(f"Action utilization: {m.get('action_utilization','?')}")
        lines.append(f"Avg coins at buy: {m.get('avg_coins_at_buy','?')}")
        lines.append(f"Avg coins wasted: {m.get('avg_coins_wasted','?')}")
        lines.append(f"Avg turns: {m.get('avg_turns','?')}")
        lines.append(f"Draw rate: {m.get('draw_rate','?')}")
        lines.append(f"P0 win rate: {m.get('p0_wr','?')}")
        top = m.get("top_buys", {})
        if top:
            lines.append(f"Top buys: {', '.join(f'{k} ({v:.0%})' for k,v in list(top.items())[:5])}")

        if prev_raw:
            p = json.loads(prev_raw)
            lines.append(f"\nPrevious iteration: {p.get('iteration','?')}")
            lines.append(f"Prev policy: {p.get('policy','?')}, value: {p.get('value','?')}, belief: {p.get('belief','?')}")
            lines.append(f"Prev provinces: {p.get('avg_provinces','?')}, score: {p.get('avg_score','?')}")

        if baseline_raw:
            b = json.loads(baseline_raw)
            lines.append(f"\nBaseline (iter {b.get('iteration','250')} — 218ch migration):")
            lines.append(f"Baseline policy: {b.get('policy','?')}, value: {b.get('value','?')}, belief: {b.get('belief','?')}")
            lines.append(f"Baseline provinces: {b.get('avg_provinces','?')}, score: {b.get('avg_score','?')}")
            lines.append(f"Baseline coins/buy: {b.get('avg_coins_at_buy','?')}, turns: {b.get('avg_turns','?')}")

    err = s.get("LAST_ERROR", "")
    if err:
        lines.append(f"\nLast error in log: {err[:200]}")

    return "\n".join(lines)


# ── Formatting for scheduled reports ────────────────────────

def f(v, dec=2):
    try:
        v = float(v)
        if v == 0: return "0"
        if v < 0.01: return f"{v:.4f}"
        return f"{v:.{dec}f}"
    except: return "-"

def pct(v):
    try: return f"{float(v)*100:.0f}%"
    except: return "-"

def trend(cur, prev):
    try:
        c, p = float(cur), float(prev)
        if p == 0: return "\u2197\ufe0f" if c > 0 else "\u2192"
        if c > p * 1.05: return "\u2197\ufe0f"
        if c < p * 0.95: return "\u2198\ufe0f"
        return "\u2192"
    except: return ""

def delta(cur, base, invert=False):
    try:
        c, b = float(cur), float(base)
        if b == 0:
            return f"0 \u2192 {f(c)}" + (" \U0001f195" if c > 0 else "")
        pct_change = ((c - b) / abs(b)) * 100
        arrow = "\u2198\ufe0f" if pct_change < -5 else ("\u2197\ufe0f" if pct_change > 5 else "\u2192")
        if invert:
            arrow = "\u2197\ufe0f" if pct_change < -5 else ("\u2198\ufe0f" if pct_change > 5 else "\u2192")
        sign = "+" if pct_change > 0 else ""
        return f"{f(b)} \u2192 {f(c)} {arrow}{sign}{pct_change:.0f}%"
    except:
        return f"- \u2192 {f(cur)}"


def build_status_msg(s, actions=""):
    if s.get("_ssh_failed"):
        return "\U0001f6a8 <b>Monk Monitor</b>\n\nSSH to RunPod failed."

    pid = s.get("PID", "NONE")
    latest_raw = s.get("LATEST_LOSS", "")
    prev_raw = s.get("PREV_LOSS", "")
    baseline_raw = s.get("BASELINE", "")

    if not latest_raw:
        return "\U0001f6a8 <b>Monk Monitor</b>\n\nNo training data."

    m = json.loads(latest_raw)
    p = json.loads(prev_raw) if prev_raw else {}
    b = json.loads(baseline_raw) if baseline_raw else {}
    it = m.get("iteration", "?")

    icon = "\u2705"
    concerns = []
    if pid in ("NONE", ""):
        icon = "\U0001f6a8"; concerns.append("Training process dead")
    try:
        if int(s.get("DISK_PCT", "0")) > 85:
            icon = "\u26a0\ufe0f"; concerns.append(f"Disk {s['DISK_PCT']}% full")
    except: pass
    if m.get("belief", 0) == 0 and it != "?":
        concerns.append("Belief loss is 0")
    if m.get("avg_provinces", 0) and float(m.get("avg_provinces", 1)) < 1.0:
        concerns.append(f"Provinces low ({f(m.get('avg_provinces', 0))})")
    try:
        if float(m.get("value", 1)) < 0.01:
            concerns.append("Value loss near 0")
    except: pass

    proc = f"\u2705 PID {pid}" if pid not in ("NONE", "") else "\u274c DEAD"
    buf_ex = s.get("BUF_EXAMPLES", "0")
    buf_cap = s.get("BUF_CAPACITY", "100000")
    try:
        buf_str = f"{int(buf_ex)//1000}K/{int(buf_cap)//1000}K ({int(buf_ex)/int(buf_cap)*100:.0f}%)"
    except:
        buf_str = "?"

    fr = m.get('force_rate')
    fr_str = f"  (fr={f(fr)})" if fr is not None else ""

    # Parse FR baseline early so Gameplay trends can use it
    fr_baseline_raw = s.get("FR_BASELINE", "")
    try:
        fb = json.loads(fr_baseline_raw) if fr_baseline_raw else {}
    except:
        fb = {}
    # Trend source: FR baseline if available, else prev iter
    t = fb if fb else p

    msg = f"""{icon} <b>Monk \u2014 Iter {it}</b>{fr_str}

<b>RunPod</b>
<pre>Process  {proc}
Disk     {s.get('DISK_PCT','?')}% ({s.get('DISK_AVAIL','?')} free)
GPU      {s.get('GPU_UTIL','?')}% / {s.get('GPU_MEM','?')}
CPU      {s.get('CPU_PCT','?')}%
RAM      {s.get('RAM_USED','?')} / {s.get('RAM_TOTAL','?')}
Buffer   {s.get('BUF_SIZE','?')} \u2014 {buf_str}</pre>

<b>Losses</b> <i>(vs prev iter)</i>
<pre>Policy  {f(m.get('policy'))} {trend(m.get('policy'), p.get('policy'))}   Value  {f(m.get('value'))} {trend(m.get('value'), p.get('value'))}
Score   {f(m.get('score'))}      Belief {f(m.get('belief'),4)} {trend(m.get('belief'), p.get('belief'))}</pre>

<b>Gameplay</b> <i>(vs fr={fb.get('force_rate','?')} iter {fb.get('iteration','?')})</i>
<pre>Provnce {delta(m.get('avg_provinces',0), t.get('avg_provinces',0))}
Duchy   {delta(m.get('avg_duchies',0), t.get('avg_duchies',0))}
Copper  {delta(m.get('avg_copper',0), t.get('avg_copper',0))}
Silver  {delta(m.get('avg_silver',0), t.get('avg_silver',0))}
Gold    {delta(m.get('avg_gold',0), t.get('avg_gold',0))}
Score   {delta(m.get('avg_score',0), t.get('avg_score',0))}
$/buy   {delta(m.get('avg_coins_at_buy',0), t.get('avg_coins_at_buy',0))}
Waste   {delta(m.get('avg_coins_wasted',0), t.get('avg_coins_wasted',0), invert=True)}
Turns   {delta(m.get('avg_turns',0), t.get('avg_turns',0), invert=True)}
ActBuy  {delta(m.get('avg_action_buys',0), t.get('avg_action_buys',0))}
Buys    {delta(m.get('avg_buys',0), t.get('avg_buys',0))}
Curses  {f(m.get('avg_curses'))}   Draws {pct(m.get('draw_rate'))}
ActRate {pct(m.get('action_rate'))}    Util  {pct(m.get('action_utilization'))}</pre>"""

    top = m.get("top_buys", {})
    top_t = t.get("top_buys", {})
    if top:
        msg += "\n<b>Action cards</b>\n<pre>"
        for card, avg in top.items():
            msg += f"{card:8s} {delta(avg, top_t.get(card, 0))}\n"
        msg += "</pre>"

    if b:
        msg += f"\n\n\U0001f4c8 <b>Since 218ch (iter {b.get('iteration', 250)})</b>\n<pre>"
        msg += f"Belief  {delta(m.get('belief',0), b.get('belief',0))}\n"
        msg += f"Policy  {delta(m.get('policy',0), b.get('policy',0), invert=True)}\n"
        msg += f"Value   {delta(m.get('value',0), b.get('value',0))}\n"
        msg += f"Provnce {delta(m.get('avg_provinces',0), b.get('avg_provinces',0))}\n"
        msg += f"Score   {delta(m.get('avg_score',0), b.get('avg_score',0))}\n"
        msg += f"$/buy   {delta(m.get('avg_coins_at_buy',0), b.get('avg_coins_at_buy',0))}\n"
        msg += f"Turns   {delta(m.get('avg_turns',0), b.get('avg_turns',0), invert=True)}"
        msg += "</pre>"

    if fb:
        try:
            fb_it = fb.get("iteration", "?")
            fb_fr = fb.get("force_rate", "?")
            msg += f"\n\n\U0001f4ca <b>Since fr={fb_fr} (iter {fb_it})</b>\n<pre>"
            msg += f"Policy  {delta(m.get('policy',0), fb.get('policy',0), invert=True)}\n"
            msg += f"Value   {delta(m.get('value',0), fb.get('value',0))}\n"
            msg += f"Provnce {delta(m.get('avg_provinces',0), fb.get('avg_provinces',0))}\n"
            msg += f"Score   {delta(m.get('avg_score',0), fb.get('avg_score',0))}\n"
            msg += f"Copper  {delta(m.get('avg_copper',0), fb.get('avg_copper',0))}\n"
            msg += f"Gold    {delta(m.get('avg_gold',0), fb.get('avg_gold',0))}\n"
            msg += f"Turns   {delta(m.get('avg_turns',0), fb.get('avg_turns',0), invert=True)}"
            msg += "</pre>"
        except:
            pass

    if concerns:
        msg += "\n\n\u26a0\ufe0f " + " | ".join(concerns)
    err = s.get("LAST_ERROR", "")
    if err and len(err) > 5:
        msg += f"\n\n\u274c <b>Last error:</b>\n<code>{err[:120]}</code>"

    if actions:
        msg += f"\n\n\U0001f527 {actions}"

    msg += "\n\n<i>Reply: help | restart | kill | logs | status</i>"
    return msg


# ── Eval ────────────────────────────────────────────────────

EVAL_YAML = os.path.join(REPO_DIR, "scripts", "monk-daemon-eval.yaml")
EVAL_REPORTS_DIR = os.path.join(REPO_DIR, "data", "eval_reports")


def run_eval():
    """Run the eval suite against monk_daemon.py, writing a timestamped report."""
    if not os.path.exists(EVAL_YAML):
        log("Eval YAML not found, skipping")
        return
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M")
    report_path = os.path.join(EVAL_REPORTS_DIR, f"monk-daemon-eval-report-{ts}.md")
    prompt = (
        f"/eval Evaluate {EVAL_YAML} against {os.path.join(REPO_DIR, 'scripts', 'monk_daemon.py')} "
        f"and write the full report to {report_path}"
    )
    try:
        subprocess.run(
            [CLAUDE_CLI, "-p", prompt, "--allowedTools", "Read Write Glob Grep"],
            cwd=REPO_DIR, capture_output=True, text=True,
        )
        log(f"Eval complete → {report_path}")
    except Exception as e:
        log(f"Eval error: {e}")


# ── Proactive infra actions ─────────────────────────────────

def run_proactive_actions(s):
    if s.get("_ssh_failed"):
        return ""
    actions = []
    tmp = s.get("TMP_FILES", "0")
    if tmp.isdigit() and int(tmp) > 0:
        ssh_run(f"rm -f {RUNPOD_REPO}/data/dominion/checkpoints/*.tmp")
        actions.append(f"Cleaned {tmp} .tmp")

    gc = s.get("GAME_CKPT_COUNT", "0")
    if gc.isdigit() and int(gc) > 5:
        ssh_run(f"cd {RUNPOD_REPO}/data/dominion/checkpoints && ls -t model_latest_game*.pt | tail -n +6 | xargs rm -f")
        actions.append("Pruned game ckpts")

    cc = s.get("CKPT_COUNT", "0")
    if cc.isdigit() and int(cc) > 10:
        ssh_run(f"cd {RUNPOD_REPO}/data/dominion/checkpoints && ls model_iter_*.pt | sort -t_ -k3 -n | head -n -10 | xargs rm -f")
        actions.append("Pruned iter ckpts")

    if s.get("PID") == "NONE" and s.get("LATEST_LOSS"):
        ssh_run(f"cd {RUNPOD_REPO} && nohup python scripts/train.py --config configs/dominion.yaml --resume /workspace/dominion_data/checkpoints/model_latest.pt > /tmp/dominion_train.log 2>&1 &")
        actions.append("RESTARTED training")
        s["PID"] = "restarted"

    return ". ".join(actions)


# ── Claude conversation ─────────────────────────────────────

SYSTEM_PROMPT = """You are the Monk — an AI training monitor for a Dominion (card game) reinforcement learning bot. You report to the CEO (Paul) via Telegram.

## Your Identity
You are a calm, knowledgeable monk-developer with deep understanding of RL training. You speak concisely — Telegram messages should be short and readable on mobile. No markdown (Telegram doesn't render it). Use plain text or minimal HTML (<b>, <i>, <code>, <pre>) only when helpful.

## Project Context
- AlphaZero-style self-play training for Dominion (2-player card game)
- C++ MCTS engine (800 sims) + PyTorch neural network (10 res blocks, 128 channels)
- 218 tensor channels (expanded from 156 at iter 250 — added opponent deck visibility + belief head)
- Belief head: predicts opponent hand composition (31 card types). Was producing all-zeros before iter 250, now learning.
- Running on RunPod GPU pod. Training ~8-10 min/iter.
- Replay buffer: 100K examples max. Disk is 20GB total, careful about space.
- Key insight: the bot needs to learn province buying strategy, economy building, and when to buy action cards vs treasures.

## What Good Dominion Play Looks Like
- Provinces > 2.5 is decent (buying victory cards)
- Belief loss should be > 0 and increasing (learning to predict opponent)
- Action utilization > 20% means the bot is actually playing action cards it buys
- Avg score > 20 is good for 2-player
- Coins/buy > 5 means decent economy
- Lower turns = more efficient games

## Your Rules
- NEVER change training config (learning rate, buffer size, MCTS sims, etc.) without CEO approval
- You CAN take infrastructure actions: restart dead training, clean disk, prune checkpoints
- Be honest about what you see — if something looks bad, say so
- Keep responses concise — 2-5 sentences typical, more only if asked for detail
- When the CEO asks you to do something on RunPod, include the action tag (see below)

## Actions
If the CEO asks you to take an action, include EXACTLY ONE of these tags on its own line at the end of your response:
ACTION:restart — Kill and restart training
ACTION:kill — Kill training without restart
ACTION:logs — Fetch and send last 30 lines of training log
ACTION:disk — Fetch and send disk breakdown
ACTION:status — Send a full formatted status report

Only include an action tag if the user clearly wants that action. For conversational messages, just respond naturally.

## Current RunPod Status
{status_context}
"""


def call_claude(system_prompt, conversation):
    """Call Claude via CLI with conversation context."""
    # Build the prompt: system context + recent conversation
    prompt_parts = []
    for role, text in conversation:
        if role == "user":
            prompt_parts.append(f"CEO: {text}")
        else:
            prompt_parts.append(f"Monk: {text}")

    full_prompt = "\n".join(prompt_parts)

    try:
        log(f"Claude call ({len(full_prompt)} chars prompt)...")
        start = time.time()
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # Prevent nested session error
        r = subprocess.run(
            [CLAUDE_CLI, "-p", full_prompt,
             "--model", "haiku",
             "--system-prompt", system_prompt],
            capture_output=True, text=True, timeout=60, env=env
        )
        elapsed = time.time() - start
        log(f"Claude done in {elapsed:.1f}s, rc={r.returncode}")
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        else:
            log(f"Claude error: {r.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        log("Claude TIMEOUT")
        return None
    except Exception as e:
        log(f"Claude FAILED: {e}")
        return None


# ── Main daemon ─────────────────────────────────────────────

class MonkDaemon:
    def __init__(self):
        self.bot, self.chat = load_env()
        self.last_update_id = 0
        self.last_report = 0
        self.conversation = []  # list of (role, text) tuples
        self._kill_hold_until = 0
        self._last_status = None
        self._last_status_time = 0
        self.load_state()
        log(f"Monk Daemon started. Chat: {self.chat}")

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as fh:
                    d = json.load(fh)
                    self.last_update_id = d.get("last_update_id", 0)
                    self.last_report = d.get("last_report", 0)
            except: pass
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE) as fh:
                    self.conversation = json.load(fh)
            except: pass

    def save_state(self):
        with open(STATE_FILE, "w") as fh:
            json.dump({
                "last_update_id": self.last_update_id,
                "last_report": self.last_report
            }, fh)
        with open(HISTORY_FILE, "w") as fh:
            json.dump(self.conversation[-MAX_HISTORY:], fh)

    def send(self, text):
        return tg_send(self.bot, self.chat, text)

    def get_status(self, force=False):
        """Get RunPod status, cached for 30s."""
        if not force and self._last_status and time.time() - self._last_status_time < 30:
            return self._last_status
        self._last_status = collect_status()
        self._last_status_time = time.time()
        return self._last_status

    def handle_message(self, text):
        """Handle a CEO message via Claude conversation."""
        log(f"MSG: {text[:80]}")

        # Collect fresh status for context
        s = self.get_status(force=True)
        status_ctx = format_status_context(s)
        system = SYSTEM_PROMPT.replace("{status_context}", status_ctx)

        # Add user message to history
        self.conversation.append(("user", text))

        # Call Claude
        response = call_claude(system, self.conversation[-MAX_HISTORY:])

        if not response:
            self.send("Sorry, I couldn't process that right now. Try again in a moment.")
            return

        # Parse action tags
        action = None
        clean_lines = []
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("ACTION:"):
                action = stripped.split(":", 1)[1].strip().lower()
            else:
                clean_lines.append(line)
        clean_response = "\n".join(clean_lines).strip()

        # Send the conversational response
        if clean_response:
            self.send(clean_response)
            self.conversation.append(("assistant", clean_response))

        # Execute action if requested
        if action:
            self.execute_action(action, s)

        self.save_state()

    def execute_action(self, action, s=None):
        """Execute a RunPod action."""
        log(f"ACTION: {action}")

        if action == "restart":
            pid_out = ssh_run("pgrep -f 'train.py.*dominion' | head -1")
            pid = pid_out.strip()
            if pid and not pid.startswith("SSH_"):
                ssh_run(f"kill {pid}")
                time.sleep(3)
            ssh_run(
                f"cd {RUNPOD_REPO} && nohup python scripts/train.py "
                f"--config configs/dominion.yaml "
                f"--resume /workspace/dominion_data/checkpoints/model_latest.pt "
                f"> /tmp/dominion_train.log 2>&1 &"
            )
            time.sleep(2)
            new_pid = ssh_run("pgrep -f 'train.py.*dominion' | head -1").strip()
            self.send(f"\u2705 Training restarted{' (PID ' + new_pid + ')' if new_pid else ''}")

        elif action == "kill":
            pid_out = ssh_run("pgrep -f 'train.py.*dominion' | head -1")
            pid = pid_out.strip()
            if pid and not pid.startswith("SSH_"):
                ssh_run(f"kill {pid}")
                self.send(f"\u274c Training killed (was PID {pid})")
                self._kill_hold_until = time.time() + REPORT_INTERVAL
            else:
                self.send("\u2705 No training process running.")

        elif action == "logs":
            out = ssh_run("tail -30 /tmp/dominion_train.log 2>/dev/null || echo 'No log'")
            if out and not out.startswith("SSH_"):
                self.send(f"<pre>{out[:3500]}</pre>")
            else:
                self.send(f"Could not fetch logs: {out[:100]}")

        elif action == "disk":
            out = ssh_run(
                "df -h / && echo && "
                f"du -sh {RUNPOD_REPO}/data/dominion/checkpoints/*.pt 2>/dev/null | sort -rh | head -5 && echo && "
                f"du -sh {RUNPOD_REPO}/data/dominion/checkpoints/buffer_latest.pkl 2>/dev/null"
            )
            if out and not out.startswith("SSH_"):
                self.send(f"<pre>{out[:3500]}</pre>")

        elif action == "status":
            if not s:
                s = self.get_status(force=True)
            msg = build_status_msg(s)
            self.send(msg)

    def send_scheduled_report(self):
        """15-minute scheduled report with proactive actions."""
        log("Scheduled report...")
        s = self.get_status(force=True)
        actions = ""
        if time.time() >= self._kill_hold_until:
            actions = run_proactive_actions(s)
        msg = build_status_msg(s, actions=actions)
        self.send(msg)
        self.last_report = time.time()
        self.save_state()
        run_eval()

    # ── Main loop ───────────────────────────────────────────

    def run(self):
        self.send_scheduled_report()

        while True:
            try:
                data = tg_get_updates(
                    self.bot, self.last_update_id + 1, timeout=POLL_TIMEOUT
                )
                if data and data.get("result"):
                    for update in data["result"]:
                        uid = update["update_id"]
                        if uid > self.last_update_id:
                            self.last_update_id = uid
                        msg = update.get("message", {})
                        if (str(msg.get("chat", {}).get("id")) == self.chat
                                and msg.get("text")):
                            self.handle_message(msg["text"].strip())
                    self.save_state()

                if time.time() - self.last_report >= REPORT_INTERVAL:
                    self.send_scheduled_report()

            except KeyboardInterrupt:
                log("Monk Daemon stopped.")
                break
            except Exception as e:
                log(f"Error: {e}")
                traceback.print_exc()
                time.sleep(10)


if __name__ == "__main__":
    daemon = MonkDaemon()
    daemon.run()
