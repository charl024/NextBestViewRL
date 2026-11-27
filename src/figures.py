"""Summary figures: final score per method on both test splits, learning curves, and view
figures for the objects where the agent gains the most over farthest-point and falls the
furthest behind greedy.

    uv run --project SLM-Lab python src/figures.py \
        --runs "full=SLM-Lab/data/nbv_full_*" --runs "hist1=SLM-Lab/data/nbv_hist1_*" \
        --runs "nodirs=SLM-Lab/data/nbv_nodirs_*" --random 3.25 --farthest 3.82 --greedy 4.91 \
        --view-run SLM-Lab/data/nbv_full_2026_11_24_093508
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from evaluate import fixed_starts
from metrics import load_run
from slm_policy import SLMPolicy, final_runs, load_spec
from view_figure import figure as view_figure

BASELINES = ["random", "farthest", "fixed_seq", "top_k_area", "greedy"]
COLORS = {"random": "#c7c7c7", "farthest": "#969696", "fixed_seq": "#7f7f7f", "top_k_area": "#636363", "greedy": "#252525",
          "full": "#1f77b4", "hist1": "#ff7f0e", "nodirs": "#2ca02c"}
NAMES = {"random": "random", "farthest": "farthest-point", "fixed_seq": "fixed sequence", "top_k_area": "top-k area",
         "greedy": "greedy (oracle)"}
SPLITS = {"test_seen": "seen classes", "test_unseen": "unseen classes"}
MARKERS = {"test_seen": "o", "test_unseen": "s"}
SMOOTH = 5


def load_eval(eval_dir, split, k):
    with open(Path(eval_dir) / f"eval_{split}_k{k}.json") as fh:
        return json.load(fh)


def seed_scores(res, label):
    return np.array([np.mean(r["final"]) for m, r in res.items() if m.startswith(label + "_s")])


def fig_main(evals, labels, out_path, detailed):
    ''' Final score per method; agents as mean +/- std across seeds. detailed adds each seed. '''
    fixed = ["fixed_seq"] if all("fixed_seq" in res for res in evals.values()) else []
    order = ["random", "farthest", *fixed, "top_k_area", *labels, "greedy"]
    size = 11 if detailed else 14
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for y, m in enumerate(order):
        xs = []
        for split, res in evals.items():
            agent = m in labels
            v = seed_scores(res, m) if agent else np.array([np.mean(res[m]["final"])])
            x, err = v.mean(), (v.std(ddof=1) if agent else 0.0)
            xs.append(x)
            face = COLORS[m] if split == "test_seen" else "white"
            ax.errorbar(x, y, xerr=err if agent else None, fmt=MARKERS[split], ms=9, mfc=face,
                        mec=COLORS[m], mew=2, ecolor=COLORS[m], elinewidth=2, capsize=4, zorder=3)
            if detailed and agent:
                ax.scatter(v, np.full(len(v), y + 0.22), s=12, color=COLORS[m], alpha=0.7, zorder=2)
        ax.plot(xs, [y, y], color=COLORS[m], lw=1.5, alpha=0.5, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES.get(m, f"RL agent: {m}") for m in order], fontsize=size)
    ax.set_xlabel("final normalized area score (k = 4)", fontsize=size)
    ax.tick_params(axis="x", labelsize=size - 2)
    ax.grid(axis="x", alpha=0.3)
    handles = [Line2D([], [], marker=MARKERS[s], ls="", ms=9, mfc="black" if s == "test_seen" else "white",
                      mec="black", mew=2, label=f"test, {name}") for s, name in SPLITS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=size - 2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_learning(curves, refs, out_path):
    ''' Mean +/- std across seeds of the smoothed eval return, with baseline returns as reference lines. '''
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, value in refs.items():
        ax.axhline(value, color=COLORS[name], ls="--", lw=1.2, zorder=1)
        ax.text(1.005, value, NAMES[name], transform=ax.get_yaxis_transform(), va="center",
                fontsize=10, color="0.3")
    for label, dfs in curves.items():
        n = min(len(df) for df in dfs)
        frames = dfs[0]["frame"].to_numpy()[:n]
        r = np.stack([df["total_reward"].rolling(SMOOTH, min_periods=1).mean().to_numpy()[:n] for df in dfs])
        mean, sd = r.mean(axis=0), r.std(axis=0, ddof=1)
        ax.fill_between(frames, mean - sd, mean + sd, color=COLORS[label], alpha=0.2, lw=0, zorder=2)
        ax.plot(frames, mean, color=COLORS[label], lw=2, label=f"{label} ({len(dfs)} seeds)", zorder=3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1e6:g}M"))
    ax.set_xlim(0, max(df["frame"].max() for dfs in curves.values() for df in dfs))
    ax.set_xlabel("environment frames", fontsize=11)
    ax.set_ylabel("eval return on training meshes\n(10 x score gain)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_views(evals, view_run, view_label, k, device, out_dir):
    ''' For each split, the object with the largest agent - farthest gain and the largest greedy - agent gap. '''
    policy = SLMPolicy(view_run, device=device)
    key = f"{view_label}_s{load_spec(view_run)['meta']['random_seed']}"
    for split, res in evals.items():
        files = res["files"]
        starts = fixed_starts(files)
        agent = np.array(res[key]["final"])
        picks = {"win": int(np.argmax(agent - np.array(res["farthest"]["final"]))),
                 "loss": int(np.argmax(np.array(res["greedy"]["final"]) - agent))}
        for kind, i in picks.items():
            out_path = Path(out_dir) / f"views_{split}_{kind}_{Path(files[i]).stem}.png"
            view_figure(files[i], policy, k, starts[files[i]], out_path)
            print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)   # label=glob of SLM-Lab run folders
    ap.add_argument("--random", type=float, required=True)      # baseline returns on the training meshes
    ap.add_argument("--farthest", type=float, required=True)
    ap.add_argument("--greedy", type=float, required=True)
    ap.add_argument("--eval-dir", default="results")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--view-run", default=None)                 # one SLM-Lab output folder
    ap.add_argument("--view-label", default="full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels, curves = [], {}
    for kv in a.runs:
        label, pattern = kv.split("=", 1)
        labels.append(label)
        curves[label] = [load_run(d)[1] for d in final_runs(pattern)]

    evals = {split: load_eval(a.eval_dir, split, a.k) for split in SPLITS}
    fig_main(evals, labels, out / "main_simple.png", detailed=False)
    fig_main(evals, labels, out / "main_detailed.png", detailed=True)
    fig_learning({l: c for l, c in curves.items() if c},
                 {"random": a.random, "farthest": a.farthest, "greedy": a.greedy},
                 out / "learning_curves.png")
    print(f"wrote {out}/main_simple.png, main_detailed.png, learning_curves.png")
    if a.view_run:
        fig_views(evals, a.view_run, a.view_label, a.k, a.device, out)


if __name__ == "__main__":
    main()
