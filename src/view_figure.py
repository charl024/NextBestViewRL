"""Show the views chosen by farthest-point, greedy, and a trained agent on the actual mesh.

Chosen cameras are numbered in the order picked; cameras are drawn closer to the object than
their true radius of 2.5.

    uv run --project SLM-Lab python src/view_figure.py --cache data/cache/test_unseen \
        --run SLM-Lab/data/nbv_full_2026_11_24_093508 --objects airplane_0639 chair_0922
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from baselines import farthest_point, greedy
from env import NBVEnv
from evaluate import fixed_starts, rollout
from score import curve, load_cache
from slm_policy import SLMPolicy

CAMERA_RADIUS = 1.45
MESH_COLOR = np.array([0.62, 0.70, 0.80])


def read_off(path):
    tokens = Path(path).read_text().split()
    n_v, n_f = int(tokens[1]), int(tokens[2])
    V = np.array(tokens[4:4 + 3 * n_v], dtype=np.float64).reshape(n_v, 3)
    faces, i = [], 4 + 3 * n_v
    for _ in range(n_f):
        n = int(tokens[i])
        idx = [int(t) for t in tokens[i + 1:i + 1 + n]]
        faces += [(idx[0], idx[j], idx[j + 1]) for j in range(1, n - 1)]
        i += n + 1
    return V, np.array(faces, dtype=np.int64)


def normalize(V):
    ''' Same as geometry.normalize: bounding-box center to the origin, farthest vertex at radius 1. '''
    V = V - (V.min(axis=0) + V.max(axis=0)) / 2
    return V / np.linalg.norm(V, axis=1).max()


def draw(ax, tris, dirs, views, title):
    ax.add_collection3d(Poly3DCollection(tris, facecolors=MESH_COLOR, edgecolors="none"))
    cams = dirs * CAMERA_RADIUS
    ax.scatter(*cams.T, s=4, color="0.6", depthshade=False)
    for i, v in enumerate(views, start=1):
        c = cams[v]
        ax.plot(*np.stack([c, dirs[v] * 1.05]).T, color="red", lw=1.2)
        ax.scatter(*c, s=40, color="red", depthshade=False)
        ax.text(*(c * 1.14), str(i), color="black", fontsize=10, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="red", lw=0.8))
    lim = CAMERA_RADIUS * 0.72
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def figure(npz, policy, k, start, out_path):
    o = load_cache(npz)
    V, F = read_off(str(np.load(npz)["mesh_path"]))
    V = normalize(V)
    tris = V[F]

    P, dirs = o["P"], o["views"] / np.linalg.norm(o["views"], axis=1, keepdims=True)
    env = NBVEnv(files=[npz], k=k, history=policy.spec["env"].get("history", "all"),
                 use_dirs=policy.spec["env"].get("use_dirs", True))
    picks = {"farthest-point": farthest_point(P, o["views"], start, k),
             "greedy (oracle)": greedy(P, o["views"], start, k),
             "RL agent": rollout(policy, env, npz, start)}

    fig = plt.figure(figsize=(16, 5.6))
    for i, (name, views) in enumerate(picks.items(), start=1):
        score = curve(P, views, o["F_all"])[-1]
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.view_init(elev=20, azim=-60)
        draw(ax, tris, dirs, views, f"{name}: score {score:.3f}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.86, wspace=0)
    fig.suptitle(f"{Path(npz).stem}, k = {k}, view 1 = shared start view", fontsize=12, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)                 # one SLM-Lab output folder
    ap.add_argument("--objects", nargs="*", default=None)   # cache file stems; default: 4 spread over the split
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/views")
    a = ap.parse_args()

    files = sorted(str(p) for p in Path(a.cache).glob("*.npz"))
    starts = fixed_starts(files)
    if a.objects:
        chosen = [str(Path(a.cache) / f"{name}.npz") for name in a.objects]
    else:
        chosen = [files[i] for i in np.linspace(0, len(files) - 1, 4).astype(int)]

    policy = SLMPolicy(a.run, device=a.device)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for npz in chosen:
        out_path = Path(a.out) / f"{Path(a.cache).name}_{Path(npz).stem}.png"
        figure(npz, policy, a.k, starts[npz], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
