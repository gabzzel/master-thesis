import time
from pathlib import Path
from typing import Union, Optional, Tuple, List

import numpy as np
import open3d
import scipy.spatial
import tqdm
from open3d.cpu.pybind.utility import Vector3dVector
import bisect

from utilities import utils
from utilities.enumerations import SurfaceReconstructionMethod as SRM, SurfaceReconstructionParameters as SRP
from utilities.run_configuration import RunConfiguration
from utilities.evaluation_results import EvaluationResults

CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

def get_surface_reconstruction_method(to_evaluate: Union[str, SRM]) -> SRM:
    if isinstance(to_evaluate, SRM):
        return to_evaluate

    t = to_evaluate.lower().strip()
    t = t.replace(" ", "_")

    if t == "alpha_shapes" or "alpha" in t or t == "a":
        return SRM.ALPHA_SHAPES

    elif "ball" in t or t == "bpa" or t == "b":
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
        verbose: bool = True) \
        -> Tuple[List[open3d.geometry.TriangleMesh], List[np.ndarray], List[str]]:
    if config.surface_reconstruction_method not in SRM:
        raise ValueError(f"Unknown algorithm {config.surface_reconstruction_method}. "
                         f"Must be one of {list(SRM)}")

    if config.surface_reconstruction_method == SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION and \
            config.orient_normals <= 0:
        print("Specified SPSR without orienting normals... This is probably undesired behaviour.")

    if verbose:
        print(f"Starting surface reconstruction using {config.surface_reconstruction_method}")

    point_clouds: List[Tuple[open3d.geometry.PointCloud, np.ndarray, Optional[int]]] = [(pcd, np.arange(len(pcd.points)), None)]

    # Apply the segmentation
    if config.segments_path is not None and config.segments_path.exists():
        print(f"Found segments file at {config.segments_path}")
        point_clouds.clear()
        segments_per_point = np.load(config.segments_path)
        assert len(segments_per_point) == len(pcd.points)

        classifications = None
        if config.classifications_path is not None and config.classifications_path.exists():
            print(f"Found classifications file at {config.classifications_path}")
            classifications = np.load(config.classifications_path)

        point_clouds = apply_segmentation(pcd=pcd,
                                          segment_index_per_point=segments_per_point,
                                          classifications_per_point=classifications)
    else:
        print(f"Skipping meshing segments, no segments file found.")

    start_time = time.time()

    all_densities: List[np.ndarray] = []
    class_per_mesh: List[str] = []

    resulting_meshes: List[open3d.geometry.TriangleMesh] = []
    skipped: int = 0
    total_points_meshed: int = 0

    pbar = tqdm.tqdm(point_clouds, desc="Meshing point clouds...", unit="pointcloud", miniters=1)
    for point_cloud, point_indices, classification in pbar:

        if skipped > 0:
            pbar.set_description(f"Meshing point clouds... (Skipped {skipped} because of size or errors)")

        if len(point_cloud.points) < 10:
            skipped += 1
            continue

        mesh: Optional[open3d.geometry.TriangleMesh] = None
        if config.surface_reconstruction_method == SRM.BALL_PIVOTING_ALGORITHM:
            radii = config.surface_reconstruction_params[SRP.BPA_RADII]
            mesh = ball_pivoting_algorithm(point_cloud=point_cloud, radii=radii, verbose=len(point_clouds) == 1)

        elif config.surface_reconstruction_method == SRM.ALPHA_SHAPES:
            alpha = config.surface_reconstruction_params[SRP.ALPHA]
            if isinstance(alpha, list) or isinstance(alpha, tuple):
                alpha = alpha[0]
            try:
                mesh = alpha_shapes(point_cloud=point_cloud, alpha=alpha, verbose=len(point_clouds) == 1)
            except Exception as e:
                print(f"Skipping alpha shapes reconstruction due to {e}.")
                skipped += 1
                continue

        elif config.surface_reconstruction_method == SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION:
            octree_max_depth: int = config.surface_reconstruction_params[SRP.POISSON_OCTREE_MAX_DEPTH]

            # Adjust the octree depth based on the size of the object that we need to mesh
            if config.poisson_adaptive:
                pts = np.asarray(point_cloud.points)
                pts_max = pts.max(axis=0)
                pts_min = pts.min(axis=0)
                bb_max_size = np.max(
                    np.abs(pts_max - pts_min)) * 1.1  # The size of the bounding cube, the initial octree cell
                octree_max_depth = min(octree_max_depth, max(2, int(np.ceil(np.log2(bb_max_size / 0.01)))))

            mesh, densities = screened_poisson_surface_reconstruction(point_cloud=point_cloud,
                                                                      octree_max_depth=octree_max_depth,
                                                                      processes=config.processes,
                                                                      verbose=len(point_clouds) == 1)

            all_densities.append(densities)

        # elif config.surface_reconstruction_method == SRM.DELAUNAY_TRIANGULATION:
        #     mesh = delaunay_triangulation(point_cloud=pcd, as_tris=True)

        elif verbose:
            print(f"Unknown algorithm {config.surface_reconstruction_method}"
                  f" or invalid parameters {config.surface_reconstruction_params}.")

        if mesh is not None:
            total_points_meshed += len(point_cloud.points)
            resulting_meshes.append(mesh)
            if classification is not None:
                class_per_mesh.append(CLASSES[classification])

    results.surface_reconstruction_time = time.time() - start_time
    print(
        f"Meshed {total_points_meshed}/{len(pcd.points)} ({round((total_points_meshed / float(len(pcd.points)) * 100), 4)}%) points")

    return resulting_meshes, all_densities, class_per_mesh


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
        # tetrahedra = delaunay.simplices
        tris = vertex_neighbours_to_triangles(delaunay.vertex_neighbor_vertices)
        #tris = np.vstack((tetrahedra[:, [0, 1, 2]],
        #                  tetrahedra[:, [0, 1, 3]],
        #                  tetrahedra[:, [0, 2, 3]],
        #                  tetrahedra[:, [1, 2, 3]]))

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


def vertex_neighbours_to_triangles(vertex_neighbours: Tuple[np.ndarray, np.ndarray]):
    indptr = vertex_neighbours[0]
    indices = vertex_neighbours[1]
    triangles = []

    for vertex_index in range(len(indptr) - 1):
        neighbours = indices[indptr[vertex_index]:indptr[vertex_index + 1]]
        for slice_index in range(len(neighbours) - 1):
            triangles.append([vertex_index, neighbours[slice_index], neighbours[slice_index + 1]])

    return np.array(triangles)


def screened_poisson_surface_reconstruction(point_cloud: open3d.geometry.PointCloud,
                                            octree_max_depth=8,
                                            processes=1,
                                            verbose=True) -> Tuple[open3d.geometry.TriangleMesh, np.ndarray]:
    """
    Create a triangulated mesh using Screened Poisson Surface Reconstruction.
    This function is a spiced up wrapper for the Open3D implementation.

    :param processes: The amount of workers to use. Set to -1 for determining automatically.
    :param verbose: Whether to print progress, elapsed time and other information.
    :param point_cloud: The Open3D PointCloud object out of which the mesh will be constructed.
    :param octree_max_depth: The maximum depth of the constructed octree which is used by the SPSR algorithm.
    A higher value indicates higher detail and a finer grained mesh, at the cost of computing time.

    :return: Returns a TriangleMesh created using Screened Poisson Surface Reconstruction.
    """

    start_time = time.time()

    # Densities are by how many vertices the other vertex is supported

    mesh = open3d.geometry.TriangleMesh()
    mesh, densities = mesh.create_from_point_cloud_poisson(pcd=point_cloud,
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

    return mesh, np.asarray(densities)


def tetrahedra_to_triangles_numpy(tetrahedra: np.ndarray) -> np.ndarray:
    # triangle_indices = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])
    # Create triangles by indexing vertices
    tri1 = tetrahedra[:, (0, 1, 2)]
    tri2 = tetrahedra[:, (0, 2, 3)]
    tri3 = tetrahedra[:, (0, 1, 3)]
    tri4 = tetrahedra[:, (1, 2, 3)]

    triangles = np.vstack((tri1, tri2, tri3, tri4))
    return triangles


def apply_segmentation(pcd: open3d.geometry.PointCloud,
                       segment_index_per_point: np.ndarray,
                       classifications_per_point: Optional[np.ndarray] = None,
                       visualize: bool = False) \
        -> List[Tuple[open3d.geometry.PointCloud, np.ndarray, Optional[int]]]:

    results: List[Tuple[open3d.geometry.PointCloud, np.ndarray, Optional[int]]] = []
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    sorted_indices = np.argsort(segment_index_per_point)
    sorted_segment_ids = segment_index_per_point[sorted_indices]
    max_segment_index = sorted_segment_ids[-1]

    visualize_pcds = []
    rng = np.random.default_rng()
    colors_per_segment = rng.random(size=(max_segment_index + 1, 3))

    start_index = 0
    for segment_id in tqdm.trange(max_segment_index + 1, desc="Applying segmentation...", miniters=1, unit="segment"):
        end_index = bisect.bisect_right(sorted_segment_ids, segment_id, lo=start_index)
        segmented_pcd_indices = sorted_indices[start_index:end_index]
        segmented_pcd_points = open3d.utility.Vector3dVector(points[segmented_pcd_indices])
        segmented_pcd = open3d.geometry.PointCloud(segmented_pcd_points)
        if colors is not None:
            segmented_pcd.colors = open3d.utility.Vector3dVector(colors[segmented_pcd_indices])
        if normals is not None:
            segmented_pcd.normals = open3d.utility.Vector3dVector(normals[segmented_pcd_indices])

        class_index = classifications_per_point[segmented_pcd_indices[0]] if classifications_per_point is not None else None
        results.append((segmented_pcd, segmented_pcd_indices, class_index))
        start_index = end_index

        if visualize:
            visualize_pcd = open3d.geometry.PointCloud(segmented_pcd_points)
            visualize_pcd.colors = open3d.utility.Vector3dVector(np.full(shape=(len(segmented_pcd_points),3),
                                                                         fill_value=colors_per_segment[segment_id]))
            visualize_pcds.append(visualize_pcd)

    if visualize:
        open3d.visualization.draw_geometries(visualize_pcds)

    return results