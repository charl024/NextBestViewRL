import numpy as np
import open3d as o3d

def fibonacci_sphere(n, radius=1.0):
    ''' Make n points with almost equal spacing on a sphere.
        Returns (ndarray (n, 3) float64)
    '''
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    theta = 2.0 * np.pi * i / ((1.0 + np.sqrt(5.0)) / 2.0)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    return pts * radius

def normalize(mesh, target_radius=1.0):
    ''' Move the bounding-box center to the origin. Scale the mesh so that the farthest
        vertex is at target_radius. Changes mesh in place.
        Returns (TriangleMesh, float64 original radius, ndarray (3,) float64 original center)
    '''
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    radius = np.linalg.norm(np.asarray(mesh.vertices), axis=1).max()
    mesh.scale(target_radius / radius, center=(0,0,0))
    return mesh, radius, center

def prepare_mesh(mesh_path, target_radius=1.0):
    ''' Load, normalize, and weld the mesh for the score calculation.
        Returns (TriangleMesh, float64 original radius, ndarray (3,) float64 original center)
    '''
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh, radius, center = normalize(mesh=mesh, target_radius=target_radius)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh, radius, center

def make_scene(mesh):
    ''' Make an Open3D raycasting scene that contains the mesh.
        Returns (RaycastingScene)
    '''
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene

def look_at(p):
    ''' Make a camera frame at point p. The camera looks at the origin.
        Returns (forward, right, up), each ndarray (3,) float64
    '''
    f = -p / np.linalg.norm(p)
    up0 = np.array([0., 1., 0.]) if abs(f[2]) > 0.9 else np.array([0., 0., 1.])
    r = np.cross(up0, f); r /= np.linalg.norm(r)
    return f, r, np.cross(f, r)

def ortho_rays(p, extent, W, H):
    ''' Make a W x H grid of parallel rays from the camera at p.
        The grid covers a square from -extent to +extent on the image plane.
        Returns (o3d.core.Tensor [H, W, 6] Float32 ray origins and directions, float cell area)
    '''
    f, r, u = look_at(p)
    s = -extent + 2*extent*(np.arange(W) + 0.5)/W
    t =  extent - 2*extent*(np.arange(H) + 0.5)/H
    S, T = np.meshgrid(s, t)
    o = p + S[..., None] * r + T[..., None] * u
    rays = np.concatenate([o, np.broadcast_to(f, o.shape)], axis=-1).astype(np.float32)
    return o3d.core.Tensor(rays), (2*extent/W) * (2*extent/H)

def view_score(scene, p, extent, W=512, H=512, weights=None, triangles=None):
    ''' Cast a W x H grid of parallel rays at the mesh from viewpoint p.
        Each hit adds one cell area. The sum is the projected area A.
        Returns (np.float64 A, np.float64 M or None)
    '''
    rays, cell = ortho_rays(p, extent, W, H)
    ans = scene.cast_rays(rays)
    hits = np.isfinite(ans['t_hit'].numpy())
    area = hits.sum() * cell
    if weights is None:
        return area, None
    # Use only the hits. A miss has the Open3D INVALID_ID as its primitive ID.
    ids = ans['primitive_ids'].numpy()[hits].astype(np.int64)
    uv = ans['primitive_uvs'].numpy()[hits].astype(np.float64)
    bary = np.stack([1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1]], axis=1)   # Open3D convention: P = (1-u-v) v0 + u v1 + v v2
    return area, (weights[triangles[ids]] * bary).sum() * cell
