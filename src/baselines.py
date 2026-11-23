"""Baselines, for comparison"""

import numpy as np

def random_views(P, views, start, k, rng):
    ''' Get a set of k views, randomly '''
    rest = [v for v in range(len(P)) if v != start]
    return [start] + list(rng.choice(rest, size=k-1, replace=False))

def farthest_point(P, views, start, k, rng=None):
    ''' Next views are one with largest angle to nearest chosen view '''
    d = views / np.linalg.norm(views, axis=1, keepdims=True)
    chosen = [start]
    for _ in range(k - 1):
        closest = (d @ d[chosen].T).max(axis=1)
        closest[chosen] = np.inf
        chosen.append(int(closest.argmin()))
    return chosen

def top_k_area(P, views, start, k, rng=None, A=None):
    ''' The single-view salient-view metric applied k-1 times. '''
    A = P.sum(axis=1) if A is None else A
    order = [int(v) for v in np.argsort(-A) if v != start]
    return [start] + order[:k - 1]

def greedy(P, views, start, k, rng=None):
    ''' Greedy choice algorithm, choices are based on knowing the table already '''
    best = P[start].copy()
    chosen = [start]
    for _ in range(k - 1):
        g = np.maximum(P - best, 0.0).sum(axis=1)
        g[chosen] = -1.0
        a = int(g.argmax())
        chosen.append(a)
        best = np.maximum(best, P[a])
    return chosen

def optimal_k3(P, views, start, k=3, rng=None):
    ''' Brute force search, goes through all pairs '''
    base = P[start]
    best_val, best_pair = -1.0, None
    for i in range(len(P)):
        if i == start:
            continue
        Bi = np.maximum(base, P[i])
        vals = np.maximum(Bi, P).sum(axis=1)
        vals[[start, i]] = -1.0
        vals[:i] = -1.0
        j = int(vals.argmax())
        if vals[j] > best_val:
            best_val, best_pair = vals[j], (i, j)
    return [start, *best_pair]

BASELINES = {"random": random_views, "farthest": farthest_point, "top_k_area": top_k_area, "greedy": greedy}
