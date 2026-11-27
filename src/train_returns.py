"""Baseline returns on the training meshes, from evaluate.py output.

    python src/train_returns.py results/eval_train_k4.json
"""
import argparse
import json

import numpy as np

REWARD_SCALE = 10.0
METHODS = ("random", "farthest", "top_k_area", "greedy")


def returns(path, reward_scale=REWARD_SCALE):
    ''' Mean of reward_scale x (final score - start score) per baseline. '''
    with open(path) as fh:
        res = json.load(fh)
    return {m: reward_scale * float(np.mean([c[-1] - c[0] for c in res[m]["curves"]]))
            for m in METHODS if m in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/eval_train_k4.json")
    ap.add_argument("--shell", action="store_true")     # print as FLOOR=, THRESHOLD=, GREEDY=
    a = ap.parse_args()

    r = returns(a.path)
    if a.shell:
        print(f"FLOOR={r['random']:.2f}")
        print(f"THRESHOLD={r['farthest']:.2f}")
        print(f"GREEDY={r['greedy']:.2f}")
    else:
        for m, v in r.items():
            print(f"{m:12s} {v:.3f}")


if __name__ == "__main__":
    main()
