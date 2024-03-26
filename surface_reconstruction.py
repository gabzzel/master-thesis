import time
from typing import Union, Optional

import numpy as np
import open3d
import scipy.spatial

from utilities import utils
from utilities.enumerations import SurfaceReconstructionMethod as SRM, SurfaceReconstructionParameters as SRP
from utilities.run_configuration import RunConfiguration
from utilities.evaluation_results import EvaluationResults


def get_surface_reconstruction_method(to_evaluate: Union[str, SRM]) -> SRM:
    if isinstance(to_evaluate, SRM):
        return to_evaluate

    t = to_evaluate.lower().strip()
    t = t.replace(" ", "_")

    if t == "alpha_shapes" or t == "alpha" or t == "a":
        return SRM.ALPHA_SHAPES

    elif t == "ball_pivoting_algorithm" or t == "ball_pivoting" or t == "bpa" or t == "b":
        return SRM.BALL_PIVOTING_ALGORITHM

    elif t == "screened_poisson_surface_reconstruction" or t == "poisson_surface_reconstruction" or t == "poisson" or \
            t == "screened_poisson" or t == "spsr" or t == "psr" or t == "p":
        return SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION

    elif t == "delaunay" or t == "d" or t == "delaunay_triangulation":
        return SRM.DELAUNAY_TRIANGULATION

    return SRM.DELAUNAY_TRIANGULATION


def run(pcd: open3d.geometry.PointCloud,
        config: RunConfiguration,
        results: EvaluationResults,
        verbose: bool = True) -> Optional[open3d.geometry.TriangleMesh]:

    start_time = time.time()
    if config.surface_reconstruction_method not in SRM:
        raise ValueError(f"Unknown algorithm {config.surface_reconstruction_method}. "
                         f"Must be one of {list(SRM)}")

    if verbose:
        print(f"Starting surface reconstruction using {config.surface_reconstruction_method}")

    mesh = None

    if config.surface_reconstruction_method == SRM.BALL_PIVOTING_ALGORITHM:
        radii = config.surface_reconstruction_params[SRP.BPA_RADII]
        mesh = ball_pivoting_algorithm(point_cloud=pcd, radii=radii, verbose=verbose)

    elif config.surface_reconstruction_method == SRM.ALPHA_SHAPES:
        alpha = config.surface_reconstruction_params[SRP.ALPHA]
        if isinstance(alpha, list) or isinstance(alpha, tuple):
            alpha = alpha[0]
        mesh = alpha_shapes(point_cloud=pcd, alpha=alpha, verbose=verbose)

    elif config.surface_reconstruction_method == SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION:
        octree_max_depth = config.surface_reconstruction_params[SRP.POISSON_OCTREE_MAX_DEPTH]
        poisson_density_quantile = config.surface_reconstruction_params[SRP.POISSON_DENSITY_QUANTILE_THRESHOLD]
        mesh = screened_poisson_surface_reconstruction(point_cloud=pcd,
                                                       octree_max_depth=octree_max_depth,
                                                       density_quantile_threshold=poisson_density_quantile,
                                                       processes=1,
                                                       verbose=verbose)

    elif config.surface_reconstruction_method == SRM.DELAUNAY_TRIANGULATION:
        mesh = delaunay_triangulation(point_cloud=pcd, as_tris=True)

    elif verbose:
        print(f"Unknown algorithm {config.surface_reconstruction_method}"
              f" or invalid parameters {config.surface_reconstruction_params}.")

    results.surface_reconstruction_time = time.time() - start_time
    return mesh


def alpha_shapes(point_cloud, alpha: float = 0.02, verbose=True):
    start_time = time.time()
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    end_time = time.time()
    if verbose:
        ntf = utils.format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: Alpha Shapes ({ntf} tris, α={alpha}) [{elapsed_time}s]")

    return mesh


def ball_pivoting_algorithm(point_cloud: open3d.geometry.PointCloud,
                            radii: Union[np.ndarray, list],
                            verbose=True) -> open3d.geometry.TriangleMesh:
    start_time = time.time()
    radii = sorted(radii)
    radii_open3d = open3d.utility.DoubleVector(radii)
    mesh = open3d.geometry.TriangleMesh()
    mesh = mesh.create_from_point_cloud_ball_pivoting(pcd=point_cloud, radii=radii_open3d)
    end_time = time.time()
    if verbose:
        ntf = utils.format_number(len(mesh.triangles))  # Number Triangles Formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: BPA ({ntf} triangles, radii {radii}) [{elapsed_time}s]")

    return mesh


def delaunay_triangulation(point_cloud: open3d.geometry.PointCloud, as_tris: bool = True) \
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


def screened_poisson_surface_reconstruction(point_cloud: open3d.geometry.PointCloud,
                                            octree_max_depth=8,
                                            density_quantile_threshold=0.1,
                                            processes=1,
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
                                                             n_threads=processes)

    end_time = time.time()
    if verbose:
        ntf = utils.format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: SPSR ({ntf} tris, max octree depth {octree_max_depth}) [{elapsed_time}s]")

    density_quantile_threshold = min(1.0, max(density_quantile_threshold, 0.0))
    if density_quantile_threshold <= 0.0:
        return mesh

    vertices_to_remove = densities < np.quantile(densities, density_quantile_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if verbose:
        nrv = utils.format_number(np.sum(vertices_to_remove))  # Number of Removed Vertices
        rem_tris = utils.format_number(len(mesh.triangles))  # Remaining Triangles
        print(f"Removed {nrv} verts in {density_quantile_threshold} density quantile, tris remaining: {rem_tris}.")

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
