#!/usr/bin/env bash
# Usage: scripts/sequence_figures.sh [run_dir]
cd "$(dirname "$0")/.." || exit 1

RUN="${1:-SLM-Lab/data/nbv_full_2026_11_24_093508}"

for split in test_seen test_unseen; do
    uv run --project SLM-Lab python src/sequence_figure.py --cache data/cache/$split --run "$RUN" \
        --objects $(ls data/cache/$split | sed 's/\.npz$//')
done
