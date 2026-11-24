#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1

RUNS=(--runs "full=SLM-Lab/data/nbv_full_*" --runs "hist1=SLM-Lab/data/nbv_hist1_*" --runs "nodirs=SLM-Lab/data/nbv_nodirs_*")

for split in test_seen test_unseen; do
    uv run --project SLM-Lab python src/evaluate.py --cache data/cache/$split --k 4 "${RUNS[@]}" --device cuda
done

THRESHOLD=3.82
FLOOR=3.25

uv run --project SLM-Lab python src/evaluate.py --cache data/cache/test_seen --k 3 --optimal

uv run --project SLM-Lab python src/metrics.py "${RUNS[@]}" --threshold ${THRESHOLD} --floor ${FLOOR}

uv run --project SLM-Lab python src/figures.py "${RUNS[@]}" --random ${FLOOR} --farthest ${THRESHOLD} --greedy 4.91 \
    --view-run SLM-Lab/data/nbv_full_2026_11_24_093508 --device cuda
