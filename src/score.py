"""Set area score, rewards, metrics."""
import numpy as np

def load_cache(path):
    ''' Load one object's tables '''
    with np.load(path) as z:
        P = z["P"].astype(np.float32)
        return {"P": P, "depth": z["depth"].astype(np.float32), "A": z["A"], "views": z["views"], "F_all": float(P.max(axis=0).sum())}

def set_score(P, views):
    ''' F(S) = sum over patches of the best projected area among the views in S. '''
    return float(P[list(views)].max(axis=0).sum())

def gain(P, best, a):
    ''' F(S + a) - F(S), where best = per-patch max over S. '''
    return float(np.maximum(P[a] - best, 0.0).sum())

def normalized(P, views, F_all):
    return set_score(P, views) / F_all

def binary_coverage(P, views):
    ''' Fraction of reachable patches seen at all (the Scan-RL style metric). '''
    reachable = P.max(axis=0) > 0
    seen = P[list(views)].max(axis=0) > 0
    return float(seen[reachable].mean())

def curve(P, views, F_all):
    ''' Normalized score after each view, in order. '''
    best = np.zeros(P.shape[1], np.float32)
    out = []
    for v in views:
        best = np.maximum(best, P[v])
        out.append(float(best.sum()) / F_all)
    return out
