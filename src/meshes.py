from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh 

from geometry import prepare_mesh

PATCH_AREA = 0.006
M_MIN = 512
M_MAX = 8192

def fix_header_helper(path):
    ''' Helper function to rename .off files from ModelNet dataset
    '''
    path = Path(path)
    text = path.read_text()
    first, _, rest = text.partition("\n")
    first = first.strip()
    if (first.startswith("OFF") and first != "OFF"):
        path.write_text("OFF\n" + first[3:].strip() + "\n" + rest)
        return True 
    return False

def load_mesh(path):
    ''' Load mesh, normalize to a unit sphere.
        Return (V, (n, 3), float64, F (m, 3) int64, o3d TriangleMesh)
    '''
    mesh, _, _ = prepare_mesh(str(path), target_radius=1.0)
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.triangles, np.int64)
    return V, F, mesh

def sample_patches(V, F, seed=0):
    ''' Sample patch centers by area, M scales with surface area. Each patch is about
        PATCH_AREA-sized.
    '''
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    M = int(np.clip(round(tm.area / PATCH_AREA), M_MIN, M_MAX))
    pts, face = trimesh.sample.sample_surface(tm, M, seed=seed)
    return pts, face.astype(np.int64)

def make_splits(root, out_dir, seen_classes, unseen_classes, n_train=50, n_test=20, seed=0, suffix=""):
    ''' Write train / test_seen / test_unseen mesh lists to out_dir/<split><suffix>.txt.
        train and test_seen come from seen_classes (ModelNet train and test folders);
        test_unseen comes from the test folders of unseen_classes (skipped if None).
        Return (dict split name -> list of mesh paths)
    '''
    rng = np.random.default_rng(seed)
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pick(cls, split, n):
        files = sorted((root / cls / split).glob("*.off"))
        idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
        return [str(files[i]) for i in sorted(idx)]

    splits = {
        f"train{suffix}":     [f for c in seen_classes for f in pick(c, "train", n_train)],
        f"test_seen{suffix}": [f for c in seen_classes for f in pick(c, "test", n_test)],
    }
    if unseen_classes:
        splits[f"test_unseen{suffix}"] = [f for c in unseen_classes for f in pick(c, "test", n_test)]
    for name, files in splits.items():
        (out_dir / f"{name}.txt").write_text("\n".join(files) + "\n")
        print(f"{name}: {len(files)} meshes")
    return splits
