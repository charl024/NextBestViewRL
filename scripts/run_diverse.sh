#!/usr/bin/env bash
# Train and evaluate on the varied training classes: data, splits, precompute,
# nbv_div_{full,hist1,nodirs} x 5 seeds (conditions in parallel), evaluation, metrics,
# figures, and the furniture-trained vs. varied-trained agents on the unseen classes.
# Finished runs are skipped.
#
# Usage, from the repo root with the precompute environment active:
#   tmux new -s nbv_div 'bash scripts/run_diverse.sh 2>&1 | tee runs/run_diverse.log; exec bash'
#
# Environment overrides:
#   PY       python for the data and precompute steps (default: python)
#   WORKERS  precompute processes (default: 8)
#   DEVICE   evaluation device (default: cuda)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
SEEDS=(11 22 33 44 55)
CONDS=(full hist1 nodirs)
LOGS=runs/diverse
OUT=results/diverse
SLM=(uv run --project SLM-Lab python)
mkdir -p "$LOGS" "$OUT"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 0. Checks ---------------------------------------------------------------
"$PY" -c "import open3d" || { echo "PY=$PY cannot import open3d" >&2; exit 1; }
[[ -e SLM-Lab/slm_lab/spec/nbv/nbv.yaml ]] || { echo "SLM-Lab/slm_lab/spec/nbv/nbv.yaml link missing" >&2; exit 1; }
command -v uv > /dev/null || { echo "uv not found" >&2; exit 1; }

# ---- 1. Data, splits, precompute ----------------------------------------------
log "data setup"
./data_setup.sh
log "splits"
"$PY" src/splits.py
log "precompute"
"$PY" src/precompute.py --workers "$WORKERS"

# ---- 2. Training --------------------------------------------------------------
run_done() {    # spec seed -> 0 if a finished, non-dev run exists
    local spec=$1 seed=$2 d
    for d in SLM-Lab/data/"${spec}"_*/; do
        [[ -f "$d/model/${spec}_t0_s0_net_model.pt" ]] || continue
        grep -Eq "\"random_seed\": ${seed}[,}[:space:]]" "$d/${spec}_t0_spec.json" || continue
        grep -Eq '"max_frame": 20000[,}[:space:]]' "$d/${spec}_t0_spec.json" && continue
        return 0
    done
    return 1
}

train() {
    local spec="nbv_div_$1" seed
    for seed in "${SEEDS[@]}"; do
        if run_done "$spec" "$seed"; then
            log "$spec seed $seed already done, skipping"
            continue
        fi
        log "$spec seed $seed start"
        (cd SLM-Lab && uv run python ../run_slm.py run -s seed="$seed" nbv/nbv.yaml "$spec" train)
        log "$spec seed $seed end"
    done
}

log "training: ${CONDS[*]} in parallel, seeds ${SEEDS[*]} in turn; logs in $LOGS/"
pids=()
for c in "${CONDS[@]}"; do
    train "$c" > "$LOGS/train_$c.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

missing=()
for c in "${CONDS[@]}"; do
    for s in "${SEEDS[@]}"; do run_done "nbv_div_$c" "$s" || missing+=("nbv_div_$c/s$s"); done
done
if (( fail || ${#missing[@]} )); then
    echo "training incomplete (${missing[*]:-a job exited with an error}); see $LOGS/" >&2
    exit 1
fi
log "all ${#CONDS[@]} x ${#SEEDS[@]} runs done"

# ---- 3. Evaluation --------------------------------------------------------------
RUNS=()
for c in "${CONDS[@]}"; do RUNS+=(--runs "$c=SLM-Lab/data/nbv_div_${c}_*"); done

log "baseline returns on the training meshes"
"${SLM[@]}" src/evaluate.py --cache data/cache/train_div --k 4 --out "$OUT"
eval "$("${SLM[@]}" src/train_returns.py "$OUT/eval_train_div_k4.json" --shell)"
[[ -n "${THRESHOLD:-}" ]] || { echo "could not read train-mesh returns" >&2; exit 1; }
log "train-mesh returns: floor $FLOOR, threshold $THRESHOLD, greedy $GREEDY"

log "fixed sequences"
"${SLM[@]}" src/fixed_sequence.py --cache data/cache/train_div --k 4 --out "$OUT/fixed_seq_k4.npy"

for split in test_seen_div test_unseen; do
    log "evaluate $split"
    "${SLM[@]}" src/evaluate.py --cache "data/cache/$split" --k 4 "${RUNS[@]}" \
        --fixed-seq "$OUT/fixed_seq_k4.npy" --device "$DEVICE" --out "$OUT" | tee "$OUT/eval_${split}_k4.txt"
done

log "learning-curve metrics"
"${SLM[@]}" src/metrics.py "${RUNS[@]}" --threshold "$THRESHOLD" --floor "$FLOOR" --out "$OUT" | tee "$OUT/metrics.txt"

log "figures"
VIEW_RUNS=(SLM-Lab/data/nbv_div_full_*/)
"${SLM[@]}" src/figures.py "${RUNS[@]}" --random "$FLOOR" --farthest "$THRESHOLD" --greedy "$GREEDY" \
    --eval-dir "$OUT" --seen-split test_seen_div --view-run "${VIEW_RUNS[0]%/}" --device "$DEVICE" --out "$OUT/figures"

log "furniture vs. varied training on the unseen classes"
"${SLM[@]}" src/evaluate.py --cache data/cache/test_unseen --k 4 \
    --runs "furniture=SLM-Lab/data/nbv_full_*" --runs "varied=SLM-Lab/data/nbv_div_full_*" \
    --device "$DEVICE" --out "$OUT/compare" | tee "$OUT/compare_unseen.txt"

log "done: results in $OUT/"
