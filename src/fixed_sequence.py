"""Fit the fixed-sequence baseline on the training meshes.

    python src/fixed_sequence.py --cache data/cache/train --k 4 --out results/fixed_seq_k4.npy
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from score import load_cache


def fit(files, k, n_views=100):
    ''' Greedy on the mean normalized score over files, for each start view.
        Returns (ndarray (n_views, k) int64, row s = the views for start view s) '''
    seq = np.zeros((n_views, k), np.int64)
    seq[:, 0] = np.arange(n_views)
    rows = np.arange(n_views)[:, None]
    for t in range(1, k):
        total = np.zeros((n_views, n_views))           # [start, candidate]
        for f in tqdm(files, desc=f"view {t + 1}/{k}"):
            o = load_cache(f)
            P = o["P"]
            best = P[seq[:, :t]].max(axis=1)            # (start, M)
            for s in range(n_views):
                total[s] += np.maximum(best[s], P).sum(axis=1) / o["F_all"]
        total[rows, seq[:, :t]] = -np.inf
        seq[:, t] = total.argmax(axis=1)
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/train")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="results/fixed_seq_k4.npy")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.cache).glob("*.npz"))
    assert files, f"no .npz files in {a.cache}"
    seq = fit(files, a.k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, seq)
    print(f"fit on {len(files)} objects from {a.cache}, wrote {a.out}")


if __name__ == "__main__":
    main()
