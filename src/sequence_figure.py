"""Show what greedy and a trained agent see at each of their k views, in the order chosen.

Each panel is an orthographic render of the mesh from that view's camera, titled with the score
the view adds. The inset is the depth image the agent receives for that view.

    uv run --project SLM-Lab python src/sequence_figure.py --cache data/cache/test_unseen \
        --run SLM-Lab/data/nbv_full_2026_11_24_093508 --objects piano_0284
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import greedy
from env import NBVEnv
from evaluate import fixed_starts, rollout
from score import curve, load_cache
from slm_policy import SLMPolicy
from view_figure import MESH_CMAP, camera_light, normalize, read_off, vertex_shade

EXTENT = 1.0


def look_at(p):
    ''' Same camera frame as geometry.look_at: (forward, right, up). '''
    f = -p / np.linalg.norm(p)
    up0 = np.array([0., 1., 0.]) if abs(f[2]) > 0.9 else np.array([0., 0., 1.])
    r = np.cross(up0, f)
    r /= np.linalg.norm(r)
    return f, r, np.cross(f, r)


def render(ax, V, F, cam):
    ''' Smooth-shaded orthographic render from cam, image right = r and up = u, far faces drawn first. '''
    f, r, u = look_at(cam)
    shade = vertex_shade(V, F, f, camera_light(f, r, u))
    order = np.argsort(-(V[F].mean(axis=1) @ f))
    ax.tripcolor(V @ r, V @ u, F[order], shade, shading="gouraud", cmap=MESH_CMAP, vmin=0, vmax=1)
    ax.set_xlim(-EXTENT, EXTENT); ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def sequence(npz, policy, k, start, out_path):
    o = load_cache(npz)
    V, F = read_off(str(np.load(npz)["mesh_path"]))
    V = normalize(V)

    P, views = o["P"], o["views"]
    env = NBVEnv(files=[npz], k=k, history=policy.spec["env"].get("history", "all"),
                 use_dirs=policy.spec["env"].get("use_dirs", True))
    rows = {"greedy (oracle)": greedy(P, views, start, k), "RL agent": rollout(policy, env, npz, start)}

    fig, axes = plt.subplots(len(rows), k, figsize=(3.2 * k, 3.4 * len(rows)), squeeze=False)
    for row, (name, chosen) in zip(axes, rows.items()):
        c = curve(P, chosen, o["F_all"])
        for i, (ax, v) in enumerate(zip(row, chosen)):
            render(ax, V, F, views[v])
            title = f"view 1 (start): {c[0]:.3f}" if i == 0 else f"view {i + 1}: +{c[i] - c[i - 1]:.3f}"
            ax.set_title(title, fontsize=11)
            inset = ax.inset_axes([0.68, 0.68, 0.3, 0.3])
            inset.imshow(env._depth(v), cmap="gray", vmin=0, vmax=1)
            inset.set_xticks([]); inset.set_yticks([])
        row[0].set_ylabel(f"{name}\nfinal {c[-1]:.3f}", fontsize=12)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.03, top=0.88, wspace=0.05, hspace=0.18)
    fig.suptitle(f"{Path(npz).stem}, k = {k}, inset = depth image given to the agent", fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)                 # one SLM-Lab output folder
    ap.add_argument("--objects", nargs="*", default=None)   # cache file stems; default: 4 spread over the split
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/sequences")
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
        sequence(npz, policy, a.k, starts[npz], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
