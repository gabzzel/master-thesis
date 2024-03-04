import time
from typing import Union

import open3d

import utils
from utils import format_number
import numpy as np
import scipy.spatial


def BPA(point_cloud: open3d.geometry.PointCloud, radii: Union[np.ndarray, list], verbose=True):
    start_time = time.time()
    radii_open3d = open3d.utility.DoubleVector(radii)
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=point_cloud, radii=radii_open3d)
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number Triangles Formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: BPA ({ntf} triangles, radii {radii}) [{elapsed_time}s]")

    return mesh


def SPSR(point_cloud: open3d.geometry.PointCloud, octree_max_depth=8, density_quantile_threshold=0.1,
         verbose=True) -> open3d.geometry.TriangleMesh:
    """
    Create a triangulated mesh using Screened Poisson Surface Reconstruction.
    This function is a spiced up wrapper for the Open3D implementation.

    :param verbose: Whether to print progress, elapsed time and other information.
    :param point_cloud: The Open3D PointCloud object out of which the mesh will be constructed.
    :param octree_max_depth: The maximum depth of the constructed octree which is used by the SPSR algorithm.
    A higher value indicates higher detail and a finer grained mesh, at the cost of computing time.
    :param density_quantile_threshold: Points with a density (i.e. support) below the quantile will be removed,
    cleaning up the mesh. Set to 0 to ignore.

    :return: Returns a TriangleMesh created using Screened Poisson Surface Reconstruction.
    """

    start_time = time.time()

    # Densities are by how many vertices the other vertex is supported

    mesh = open3d.geometry.TriangleMesh()
    (mesh, densities) = mesh.create_from_point_cloud_poisson(pcd=point_cloud,
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
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: Alpha Shapes ({ntf} tris, α={alpha}) [{elapsed_time}s]")

    return mesh


def Delaunay(point_cloud: open3d.geometry.PointCloud, as_tris: bool = True) \
        -> Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh]:
    """
    Computes the Delaunay triangulation for surface reconstruction.

    :param point_cloud: The Open3D point cloud to triangulate.
    :param as_tris: Whether to return a triangulated mesh or a TetraMesh made of tetrahedrons.
    :returns: Either a TriangleMesh or TetraMesh constructed from the point cloud points.
    """

    start_time = time.time()
    # Make a copy just to be sure

    # QJ = "joggled input to avoid precision errors" http://www.qhull.org/html/qh-optq.htm#QJn
    # Qbb = "scale the last coordinate to [0,m] for Delaunay" http://www.qhull.org/html/qh-optq.htm#Qbb
    # Qc = "keep coplanar points with nearest facet" http://www.qhull.org/html/qh-optq.htm#Qc
    # Qz = "add a point-at-infinity for Delaunay triangulations" http://www.qhull.org/html/qh-optq.htm#Qz
    # Q12 = "allow wide facets and wide dupridge" http://www.qhull.org/html/qh-optq.htm#Q12
    qhull_options = "Qt Qbb Qc Qz Q12"

    # From the Scipy docs: (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
    # Option “Qt” is always enabled. Default: "Qbb Qc Qz Qx Q12" for ndim > 4 and “Qbb Qc Qz Q12” otherwise.
    # Incremental mode omits “Qz”.
    delaunay = scipy.spatial.Delaunay(np.asarray(point_cloud.points).copy(), qhull_options=qhull_options)

    if as_tris:
        tetrahedra = delaunay.simplices
        tris = np.vstack((tetrahedra[:, [0, 1, 2]],
                          tetrahedra[:, [0, 1, 3]],
                          tetrahedra[:, [0, 2, 3]],
                          tetrahedra[:, [1, 2, 3]]))

        # Sort and remove duplicates
        tris.sort(axis=1)
        tris = np.unique(tris, axis=0)
        isin = np.isin(tris, delaunay.convex_hull.flatten())
        is_convex_hull_facet = np.any(isin, axis=1)
        tris = tris[~is_convex_hull_facet]
        mesh = open3d.geometry.TriangleMesh(vertices=open3d.utility.Vector3dVector(delaunay.points),
                                            triangles=open3d.utility.Vector3iVector(tris))
        count = utils.format_number(len(tris))
    else:
        count = utils.format_number(len(delaunay.simplices))
        mesh = open3d.geometry.TetraMesh(vertices=open3d.utility.Vector3dVector(delaunay.points),
                                         tetras=open3d.utility.Vector4iVector(delaunay.simplices))

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 3)
    form = 'tris' if as_tris else 'tetras'
    print(f"Created mesh: Delaunay ({count} {form}) [{elapsed_time}s]")
    return mesh


def tetrahedra_to_triangles_numpy(tetrahedra: np.ndarray) -> np.ndarray:
    # triangle_indices = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])
    # Create triangles by indexing vertices
    tri1 = tetrahedra[:, (0, 1, 2)]
    tri2 = tetrahedra[:, (0, 2, 3)]
    tri3 = tetrahedra[:, (0, 1, 3)]
    tri4 = tetrahedra[:, (1, 2, 3)]

    triangles = np.vstack((tri1, tri2, tri3, tri4))
    return triangles
