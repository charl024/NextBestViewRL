"""Per-object patch-area tables, depth images, and areas, cached as .npz.

    python src/precompute.py --workers 4
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from geometry import fibonacci_sphere, make_scene, ortho_rays
from meshes import load_mesh, sample_patches

N_VIEWS, RADIUS, EXTENT = 100, 2.5, 1.0
GRID, IMG = 128, 64

def cast_view(scene, p, grid=GRID):
    ''' Cast the orthographic ray grid from viewpoint p.
        Returns (t_hit (g,g), hit mask (g,g), hit points (h,3), triangle ids (h,), cell area)
    '''
    rays, cell = ortho_rays(p, EXTENT, grid, grid)
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    hit = np.isfinite(t)
    r = rays.numpy()[hit]
    x = r[:, :3] + t[hit, None] * r[:, 3:]
    tri = ans["primitive_ids"].numpy()[hit].astype(np.int64)
    return t, hit, x, tri, cell

def assign_patches(tree, face_of_patch, x, tri, k=8):
    ''' Nearest patch center on the same triangle as the hit; nearest overall if none.
        Returns (ndarray (h,) int64 patch index per hit)
    '''
    _, nn = tree.query(x, k=k)
    same = face_of_patch[nn] == tri[:, None]
    first = same.argmax(axis=1)
    return np.where(same.any(axis=1), nn[np.arange(len(nn)), first], nn[:, 0])

def depth_image(t, hit, img=IMG):
    ''' Map hit distances [R-1, R+1] to (0.1, 1] with near = 1; misses = 0; pool to img x img.
        Returns (ndarray (img, img) float32)
    '''
    d = np.zeros_like(t, dtype=np.float32)
    d[hit] = 1.0 - 0.9 * np.clip((t[hit] - (RADIUS - 1.0)) / 2.0, 0.0, 1.0)
    f = d.shape[0] // img
    return d.reshape(img, f, img, f).mean(axis=(1, 3))

def compute_tables(V, F, mesh, seed=0):
    ''' Cast every view and credit each hit's cell area to its patch.
        Returns dict with P (N,M), depth (N,IMG,IMG), A (N,), F_all, views (N,3), points (M,3)
    '''
    pts, face = sample_patches(V, F, seed=seed)
    M = len(pts)
    tree = cKDTree(pts)
    scene = make_scene(mesh)
    views = fibonacci_sphere(N_VIEWS, RADIUS)

    P = np.zeros((N_VIEWS, M), np.float32)
    depth = np.zeros((N_VIEWS, IMG, IMG), np.float32)
    for v, p in enumerate(views):
        t, hit, x, tri, cell = cast_view(scene, p)
        patch = assign_patches(tree, face, x, tri)
        P[v] = np.bincount(patch, minlength=M) * cell
        depth[v] = depth_image(t, hit)

    return {"P": P, "depth": depth, "A": P.sum(axis=1), "F_all": P.max(axis=0).sum(),
            "views": views, "points": pts}

def process(args):
    ''' Compute and save one mesh's tables. Skips meshes whose .npz already exists.
        Returns (str output path)
    '''
    mesh_path, out_path = args
    if Path(out_path).exists():
        return out_path
    V, F, mesh = load_mesh(mesh_path)
    tab = compute_tables(V, F, mesh)
    save = {k: (v.astype(np.float16) if k in ("P", "depth") else v) for k, v in tab.items()}
    np.savez_compressed(out_path, mesh_path=str(mesh_path), **save)
    return out_path

def cache_name(mesh_path):
    return Path(mesh_path).stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = []
    for split_file in sorted(Path(a.splits).glob("*.txt")):
        out_dir = Path(a.cache) / split_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for line in split_file.read_text().split():
            jobs.append((line, str(out_dir / f"{cache_name(line)}.npz")))
    with Pool(a.workers) as pool:
        for _ in tqdm(pool.imap_unordered(process, jobs), total=len(jobs)):
            pass

if __name__ == "__main__":
    main()
