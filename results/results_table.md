# Results

k = 4 views per object, 200 objects per test split, one fixed start view per object (same for every method).
Agent rows are mean +/- std across 5 seeds (11, 22, 33, 44, 55). Baselines are single values.
fixed_seq uses the same views for every object with the same start view, fit on the training meshes (`results/fixed_seq_k4.npy`).

Columns:
- **Score**: final normalized area score, F(S) / F_all
- **Obj SD**: std of the final score across objects (agents: averaged over seeds)
- **AUC**: mean of the score curve over views 1..k
- **% greedy**: per object, (method - start) / (greedy - start), averaged
- **Binary**: fraction of visible patches seen at all

## Test seen (ModelNet10 classes, held-out objects)

| Method | Score | Obj SD | AUC | % greedy | Binary |
|---|---|---|---|---|---|
| random | 0.549 | 0.080 | 0.406 | 66.6 | 0.817 |
| farthest | 0.604 | 0.085 | 0.442 | 78.2 | 0.870 |
| fixed_seq | 0.656 | 0.062 | 0.479 | 87.5 | 0.894 |
| top_k_area | 0.645 | 0.090 | 0.466 | 84.5 | 0.885 |
| greedy | 0.719 | 0.070 | 0.519 | 100.0 | 0.910 |
| full | 0.705 +/- 0.004 | 0.074 | 0.480 +/- 0.004 | 97.2 +/- 0.8 | 0.893 +/- 0.002 |
| hist1 | 0.701 +/- 0.001 | 0.074 | 0.478 +/- 0.003 | 96.4 +/- 0.1 | 0.894 +/- 0.002 |
| nodirs | 0.698 +/- 0.005 | 0.076 | 0.471 +/- 0.007 | 95.8 +/- 0.9 | 0.892 +/- 0.002 |

## Test unseen (10 classes not used in training)

| Method | Score | Obj SD | AUC | % greedy | Binary |
|---|---|---|---|---|---|
| random | 0.534 | 0.090 | 0.400 | 66.0 | 0.835 |
| farthest | 0.596 | 0.082 | 0.436 | 79.6 | 0.881 |
| fixed_seq | 0.611 | 0.083 | 0.452 | 82.0 | 0.889 |
| top_k_area | 0.634 | 0.099 | 0.457 | 86.1 | 0.884 |
| greedy | 0.697 | 0.080 | 0.506 | 100.0 | 0.918 |
| full | 0.630 +/- 0.007 | 0.105 | 0.444 +/- 0.003 | 85.8 +/- 1.5 | 0.882 +/- 0.003 |
| hist1 | 0.629 +/- 0.005 | 0.104 | 0.445 +/- 0.003 | 85.7 +/- 1.0 | 0.884 +/- 0.003 |
| nodirs | 0.626 +/- 0.007 | 0.099 | 0.438 +/- 0.006 | 85.0 +/- 1.5 | 0.881 +/- 0.006 |

## Per-seed final score

| Condition | Split | s11 | s22 | s33 | s44 | s55 | Mean +/- SD |
|---|---|---|---|---|---|---|---|
| full | test seen | 0.708 | 0.698 | 0.709 | 0.705 | 0.706 | 0.705 +/- 0.004 |
| hist1 | test seen | 0.702 | 0.702 | 0.702 | 0.701 | 0.701 | 0.701 +/- 0.001 |
| nodirs | test seen | 0.702 | 0.691 | 0.699 | 0.702 | 0.696 | 0.698 +/- 0.005 |
| full | test unseen | 0.637 | 0.621 | 0.625 | 0.627 | 0.638 | 0.630 +/- 0.007 |
| hist1 | test unseen | 0.637 | 0.629 | 0.626 | 0.627 | 0.627 | 0.629 +/- 0.005 |
| nodirs | test unseen | 0.630 | 0.623 | 0.636 | 0.621 | 0.619 | 0.626 +/- 0.007 |

## Greedy vs. optimal (k = 3, test seen, baselines only)

| Method | Score | Obj SD | AUC | % greedy | Binary |
|---|---|---|---|---|---|
| random | 0.468 | 0.087 | 0.351 | 59.3 | 0.748 |
| farthest | 0.528 | 0.089 | 0.388 | 74.5 | 0.828 |
| top_k_area | 0.572 | 0.116 | 0.407 | 81.6 | 0.813 |
| greedy | 0.646 | 0.080 | 0.453 | 100.0 | 0.878 |
| optimal | 0.653 | 0.077 | 0.440 | 101.8 | 0.879 |

## Learning curves (training meshes, eval return = 10 x score gain)

Baseline returns on all 500 training meshes (k = 4, same fixed start views; from `results/eval_train_k4.json` via `src/train_returns.py`):

| Baseline | Score | Return |
|---|---|---|
| random | 0.547 | 3.248 |
| farthest | 0.604 | 3.823 |
| top_k_area | 0.633 | 4.108 |
| greedy | 0.713 | 4.910 |

Threshold = farthest-point return (3.82); floor = random return (3.25); greedy return (4.91) is the reference line in the learning-curve figure.

| Condition | AUC | Final return | Frames to threshold (per seed) | Failed | Minutes per run |
|---|---|---|---|---|---|
| full | 4.624 +/- 0.025 | 4.747 +/- 0.087 | 120k, 180k, 140k, 120k, 160k | 0/5 | 31.8 - 32.2 |
| hist1 | 4.616 +/- 0.018 | 4.724 +/- 0.090 | 140k, 180k, 140k, 160k, 120k | 0/5 | 31.9 - 32.3 |
| nodirs | 4.571 +/- 0.054 | 4.697 +/- 0.107 | 180k, 200k, 140k, 180k, 220k | 0/5 | 31.9 - 32.3 |

## Compute

15 runs x 5M frames on one NVIDIA RTX 5090, 3 runs in parallel (about 2,500 fps each), about 32 minutes per run, about 2 h 40 min for the full matrix.
