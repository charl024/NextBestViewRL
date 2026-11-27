#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1

RUNS=(--runs "full=SLM-Lab/data/nbv_full_*" --runs "hist1=SLM-Lab/data/nbv_hist1_*" --runs "nodirs=SLM-Lab/data/nbv_nodirs_*")

# Baseline returns on the training meshes (saved to results/eval_train_k4.json):
# FLOOR = random, THRESHOLD = farthest-point, GREEDY = greedy
uv run --project SLM-Lab python src/evaluate.py --cache data/cache/train --k 4
eval "$(uv run --project SLM-Lab python src/train_returns.py results/eval_train_k4.json --shell)"
[[ -n "${THRESHOLD:-}" ]] || { echo "could not read train-mesh returns" >&2; exit 1; }
echo "train-mesh returns: floor ${FLOOR}, threshold ${THRESHOLD}, greedy ${GREEDY}"

for split in test_seen test_unseen; do
    uv run --project SLM-Lab python src/evaluate.py --cache data/cache/$split --k 4 "${RUNS[@]}" --device cuda
done

uv run --project SLM-Lab python src/evaluate.py --cache data/cache/test_seen --k 3 --optimal

uv run --project SLM-Lab python src/metrics.py "${RUNS[@]}" --threshold ${THRESHOLD} --floor ${FLOOR}

uv run --project SLM-Lab python src/figures.py "${RUNS[@]}" --random ${FLOOR} --farthest ${THRESHOLD} --greedy ${GREEDY} \
    --view-run SLM-Lab/data/nbv_full_2026_11_24_093508 --device cuda
