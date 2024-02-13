import time

import open3d.utility
from open3d.utility import DoubleVector
from open3d.geometry import TriangleMesh, PointCloud

import utils
from utils import format_number
import numpy as np

import scipy.spatial


def BPA(point_cloud, radii, verbose=True):
    start_time = time.time()
    mesh = TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, DoubleVector(radii))
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number Triangles Formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: BPA ({ntf} triangles, radii {radii}) [{elapsed_time}s]")

    return mesh


def SPSR(point_cloud: PointCloud, octree_max_depth=8, density_quantile_threshold=0.1, verbose=True) -> TriangleMesh:
    """
    Create a triangulated mesh using Screened Poisson Surface Reconstruction.
    This function is a spiced up wrapper for the Open3D implementation.

    :param point_cloud: The Open3D PointCloud object out of which the mesh will be constructed.
    :param octree_max_depth: The maximum depth of the constructed octree which is used by the SPSR algorithm.
    A higher value indicates higher detail and a finer grained mesh, at the cost of computing time.
    :param density_quantile_threshold: Points with a density (i.e. support) below the quantile will be removed,
    cleaning up the mesh. Set to 0 to ignore.

    :return: Returns a TriangleMesh created using SPSR.
    """

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


def Delaunay(point_cloud: PointCloud, as_tris:bool = True):
    start_time = time.time()
    points = np.asarray(point_cloud.points)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    # For all options, see: http://www.qhull.org/html/qh-optq.htm#QJn
    # Use Qhull in the background, with default:
    # Qbb = scale the last coordinate to [0,m] for Delaunay
    # Qc = keep coplanar points with nearest facet
    # Qz =  add a point-at-infinity for Delaunay triangulations
    # Q12 = allow wide facets and wide dupridge
    surface = scipy.spatial.Delaunay(points=points, furthest_site=False, incremental=False, qhull_options="Qc Qz")
    vertices = open3d.utility.Vector3dVector(point_cloud.points)
    simplices = np.array(surface.simplices)

    if as_tris:
        triangles: np.ndarray = tetrahedra_to_triangles_numpy(simplices)
        triangles: open3d.utility.Vector3iVector = open3d.utility.Vector3iVector(triangles)
        mesh = TriangleMesh(vertices, triangles)
        count = utils.format_number(len(triangles))
    else:
        tetras = open3d.utility.Vector4iVector(simplices)
        mesh = open3d.geometry.TetraMesh(vertices, tetras)
        count = utils.format_number(len(tetras))

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 3)
    form = 'tris' if as_tris else 'tetras'
    print(f"Created mesh: Delaunay ({count} {form}) [{elapsed_time}s]")
    return mesh


def tetrahedra_to_triangles_numpy(tetrahedra: np.ndarray) -> np.ndarray:
    triangle_indices = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])

    # Create triangles by indexing vertices
    triangles: np.ndarray = tetrahedra[:, triangle_indices]
    triangles = np.reshape(triangles, newshape=(triangles.shape[0] * triangles.shape[1], triangles.shape[2]))
    return triangles
