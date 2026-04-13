#!/bin/bash
# Sync Dominion training data from RunPod to local for dashboard
SSH_KEY=~/.ssh/id_ed25519
HOST=root@38.147.83.30
PORT=26242
LOCAL_DIR=/Users/paulcapriolo/conductor/workspaces/mandala-rl/raleigh/data/dominion
REMOTE_DIR=/workspace/dominion_data

scp -q -P $PORT -i $SSH_KEY $HOST:$REMOTE_DIR/losses.jsonl $LOCAL_DIR/losses.jsonl 2>/dev/null
scp -q -P $PORT -i $SSH_KEY $HOST:$REMOTE_DIR/heartbeat.json $LOCAL_DIR/heartbeat.json 2>/dev/null
scp -q -P $PORT -i $SSH_KEY $HOST:$REMOTE_DIR/elo_ratings.json $LOCAL_DIR/elo_ratings.json 2>/dev/null
