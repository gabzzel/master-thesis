import time
from open3d.utility import DoubleVector
from open3d.geometry import TriangleMesh
from utils import format_number
import numpy as np


def BPA(point_cloud, radii, verbose=True):
    start_time = time.time()
    mesh = TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, DoubleVector(radii))
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number Triangles Formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: BPA ({ntf} triangles, radii {radii}) [{elapsed_time}s]")

    return mesh


def SPSR(point_cloud, octree_max_depth=8, density_quantile_threshold=0.1, verbose=True):
    start_time = time.time()

    # Densities is by how many vertices the other vertex is supported
    (mesh, densities) = TriangleMesh.create_from_point_cloud_poisson(point_cloud,
                                                                     depth=octree_max_depth,
                                                                     width=0,
                                                                     scale=1.1,
                                                                     linear_fit=False,
                                                                     n_threads=-1)

    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: SPSR ({ntf} tris, max octree depth {octree_max_depth}) [{elapsed_time}s]")

    density_quantile_threshold = min(1.0, max(density_quantile_threshold, 0.0))
    if density_quantile_threshold <= 0.0:
        return mesh

    vertices_to_remove = densities < np.quantile(densities, density_quantile_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if verbose:
        nrv = format_number(np.sum(vertices_to_remove))  # Number of Removed Vertices
        rem_tris = format_number(len(mesh.triangles))  # Remaining Triangles
        print(f"Removed {nrv} verts in {density_quantile_threshold} density quantile, tris remaining: {rem_tris}.")

    return mesh


def AlphaShapes(point_cloud, alpha=0.02, verbose=True):
    start_time = time.time()
    mesh = TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: Alpha Shapes ({ntf} tris, α={alpha}) [{elapsed_time}s]")

    return mesh
