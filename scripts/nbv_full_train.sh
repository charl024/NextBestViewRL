cd SLM-Lab

for seed in 11 22 33 44 55; do
    uv run python ../run_slm.py run -s seed=$seed nbv/nbv.yaml nbv_full train;
done
