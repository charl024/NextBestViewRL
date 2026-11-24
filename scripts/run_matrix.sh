#!/usr/bin/env bash
# Start a tmux session for each training condition.

REPO="$(cd "$(dirname "$0")/.." && pwd)"

for cond in full hist1 nodirs; do
    session="nbv_$cond"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session $session already exists, skipping"
        continue
    fi
    tmux new-session -d -s "$session" -c "$REPO" "bash scripts/nbv_${cond}_train.sh; exec bash"
    echo "started $session"
done

tmux ls
