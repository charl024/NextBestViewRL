from pathlib import Path

import numpy as np

from evaluate import evaluate

files = sorted(str(p) for p in Path("data/cache/train").glob("*.npz"))[:200]
for k in range(2, 9):
    r = evaluate(files, k)
    print(k, {m: round(float(np.mean(r[m]["final"])), 3) for m in ("random", "farthest", "greedy")})
