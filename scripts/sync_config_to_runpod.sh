#!/bin/bash
# Push config from local repo to RunPod.
# Only syncs the YAML config file. Does NOT restart training.
# Hot-reloadable params take effect at next iteration.
# Non-hot-reloadable params (network arch, lr_milestones, etc.) require manual restart.
#
# Usage:
#   ./scripts/sync_config_to_runpod.sh              # dominion (default)
#   ./scripts/sync_config_to_runpod.sh dominion
#   ./scripts/sync_config_to_runpod.sh default       # mandala

set -euo pipefail

GAME=${1:-dominion}
CONFIG="configs/${GAME}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found. Run from repo root."
    exit 1
fi

# SSH config (matches sync_from_runpod.sh)
REMOTE="root@38.147.83.30"
KEY="$HOME/.ssh/id_ed25519"
SSH_PORT=26242
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $KEY -p $SSH_PORT"

REMOTE_CONFIG="/root/mandala-dom/$CONFIG"

# Fetch remote config
REMOTE_YAML=$(ssh $SSH_OPTS $REMOTE "cat $REMOTE_CONFIG" 2>/dev/null) || {
    echo "Error: Could not read $REMOTE_CONFIG from RunPod."
    exit 1
}
LOCAL_YAML=$(cat "$CONFIG")

if [ "$LOCAL_YAML" = "$REMOTE_YAML" ]; then
    echo "No differences. Already in sync."
    exit 0
fi

echo "=== Config diff (RunPod → Local) ==="
diff <(echo "$REMOTE_YAML") <(echo "$LOCAL_YAML") || true

# Warn about non-hot-reloadable changes
# Hot-reloadable: dirichlet_epsilon, dirichlet_alpha, c_puct, temperature,
#   temperature_threshold, explore_epsilon, action_explore_boost,
#   action_buy_force_rate, action_play_force_rate, big_money_force_rate,
#   draw_penalty, max_turns
# Everything else requires restart.
RESTART_KEYS="input_channels|num_actions|num_res_blocks|channels|belief_size|phase_aware|factored|learning_rate|lr_milestones|lr_gamma|replay_buffer|min_buffer|entropy_weight|policy_weight|epochs_per|batch_size|province_supply|opponent_diversity|opponent_iter|disabled_basic|max_action_cards|num_simulations"
RESTART_CHANGES=$(diff <(echo "$REMOTE_YAML") <(echo "$LOCAL_YAML") | grep '^[<>]' | grep -E "$RESTART_KEYS" || true)

if [ -n "$RESTART_CHANGES" ]; then
    echo ""
    echo "WARNING: These changes require a TRAINING RESTART to take effect:"
    echo "$RESTART_CHANGES"
    echo "(They will sit dormant in the config file until the process restarts.)"
    echo ""
fi

read -p "Push local config to RunPod? [y/N] " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Aborted."
    exit 1
fi

scp $SSH_OPTS "$CONFIG" "$REMOTE:$REMOTE_CONFIG"
echo "Config pushed to RunPod:$REMOTE_CONFIG"
