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
    