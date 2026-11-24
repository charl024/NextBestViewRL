"""Evaluate baselines and trained SLM-Lab agents, with fixed start views.

Baselines only (nbv env):
    python src/evaluate.py --cache data/cache/train --k 4 --max-files 200

With trained agents (SLM-Lab env):
    uv run --project SLM-Lab python src/evaluate.py --cache data/cache/test_seen --k 4 \
        --runs "full=SLM-Lab/data/nbv_full_*" --runs "hist1=SLM-Lab/data/nbv_hist1_*" --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import BASELINES, optimal_k3
from score import binary_coverage, curve, load_cache


def fixed_starts(files, n_views=100, seed=0):
    ''' One start view per object, the same for every method. '''
    rng = np.random.default_rng(seed)
    return {f: int(rng.integers(n_views)) for f in files}


def rollout(policy, env, file, start):
    ''' Run one episode of a trained agent from a given start view then return the chosen views. '''
    obs, _ = env.reset(options={"file": file, "start": start})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(policy.act(obs, env.action_masks()))
    return list(env.chosen)


def evaluate(files, k, policies=None, img=32, with_optimal=False, seed=0):
    ''' policies: {name: SLMPolicy}. Returns {method: {"final", "curves", "coverage"}},
        one entry per object in each list, objects in the order of files. '''
    starts = fixed_starts(files, seed=seed)
    rng = np.random.default_rng(seed)
    methods = dict(BASELINES)
    if with_optimal and k == 3:
        methods["optimal"] = optimal_k3
    policies = policies or {}
    res = {m: {"final": [], "curves": [], "coverage": []} for m in list(methods) + list(policies)}

    envs = {}
    if policies:
        from env import NBVEnv
        for name, pol in policies.items():
            e = pol.spec["env"]
            if e.get("k", k) != k:
                raise ValueError(f"{name} was trained with k={e.get('k')}, but --k is {k}")
            envs[name] = NBVEnv(files=files, k=k, img=img,
                                history=e.get("history", "all"), use_dirs=e.get("use_dirs", True))

    for f in files:
        o = load_cache(f)
        P, s = o["P"], starts[f]
        chosen = {m: fn(P, o["views"], s, k, rng) for m, fn in methods.items()}
        for name, pol in policies.items():
            chosen[name] = rollout(pol, envs[name], f, s)
        for m, views in chosen.items():
            c = curve(P, views, o["F_all"])
            res[m]["final"].append(c[-1])
            res[m]["curves"].append(c)
            res[m]["coverage"].append(binary_coverage(P, views))
    return res


def summarize(res):
    ''' One row per method. %greedy = share of greedy's gain over the start view. '''
    g = np.array(res["greedy"]["final"])
    print(f"{'method':14s} {'score':>7s} {'±obj':>6s} {'AUC':>6s} {'%greedy':>8s} {'binary':>7s}")
    for m, r in res.items():
        f = np.array(r["final"])
        start = np.array([c[0] for c in r["curves"]])
        pg = np.mean((f - start) / np.maximum(g - start, 1e-9))
        auc = np.mean([np.mean(c) for c in r["curves"]])
        print(f"{m:14s} {f.mean():7.3f} {f.std():6.3f} {auc:6.3f} {100*pg:7.1f}% {np.mean(r['coverage']):7.3f}")


def across_seeds(res, label):
    ''' Methods named "<label>_s<seed>": mean score per seed -> mean ± std across seeds. '''
    per_seed = {m: float(np.mean(r["final"])) for m, r in res.items() if m.startswith(label + "_s")}
    if not per_seed:
        print(f"{label:10s} no runs found")
        return per_seed
    v = np.array(list(per_seed.values()))
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    print(f"{label:10s} {v.mean():.3f} ± {sd:.3f} across {len(v)} seeds   {per_seed}")
    return per_seed


def plot_curves(res, out_path, title=""):
    plt.figure(figsize=(5, 3.5))
    for m, r in res.items():
        c = np.array(r["curves"])
        plt.plot(np.arange(1, c.shape[1] + 1), c.mean(axis=0), marker="o", label=m)
    plt.xlabel("views chosen"); plt.ylabel("normalized area score"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)                   # e.g. data/cache/test_seen
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--img", type=int, default=32)
    ap.add_argument("--runs", action="append", default=[])      # label=glob of SLM-Lab run folders
    ap.add_argument("--optimal", action="store_true")           # brute force, k = 3 only
    ap.add_argument("--max-files", type=int, default=None)      # use only the first N files
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.cache).glob("*.npz"))[:a.max_files]
    assert files, f"no .npz files in {a.cache}"

    policies, labels = {}, []
    if a.runs:
        from slm_policy import SLMPolicy, final_runs
    for kv in a.runs:
        label, pattern = kv.split("=", 1)
        labels.append(label)
        for d in final_runs(pattern):
            pol = SLMPolicy(d, device=a.device)
            policies[f"{label}_s{pol.spec['meta']['random_seed']}"] = pol

    res = evaluate(files, a.k, policies, img=a.img, with_optimal=a.optimal)
    print(f"{len(files)} objects from {a.cache}, k = {a.k}")
    summarize(res)
    for label in labels:
        across_seeds(res, label)

    Path(a.out).mkdir(exist_ok=True)
    tag = f"{Path(a.cache).name}_k{a.k}"
    with open(Path(a.out) / f"eval_{tag}.json", "w") as fh:
        json.dump({"files": files, **res}, fh)
    plot_curves(res, Path(a.out) / f"curves_{tag}.png", title=tag)


if __name__ == "__main__":
    main()
