import time
from typing import Union, Tuple, Optional

import numpy as np
import open3d
from open3d.cpu.pybind.core import Tensor
from open3d.cpu.pybind.t.geometry import RaycastingScene
from open3d.geometry import TriangleMesh
from point_cloud_utils import k_nearest_neighbors

import mesh_quality
import utils


def format_number(number, digits=1):
    if number >= 1000000000:
        return str(round(number / 1000000000, digits)) + "B"

    if number >= 1000000:
        return str(round(number / 1000000, digits)) + "M"

    if number >= 1000:
        return str(round(number / 1000, digits)) + "K"

    return str(number)


def get_stats(a: np.array, name: str, print_results=True, round_digits=3, return_results=False):
    _max = round(np.max(a), round_digits)
    _min = round(np.min(a), round_digits)
    avg = round(np.average(a), round_digits)
    med = round(float(np.median(a=a)), round_digits)
    std = round(float(np.std(a=a)), round_digits)
    count = len(a)

    if print_results:
        print(f"{name} stats: Count={count}, Max={_max}, Min={_min}, Avg/Mean={avg}, Med={med}, Std={std}")
    if return_results:
        return _max, _min, avg, med, std


def get_mesh_edge_lengths(vertices, triangles):
    # Compute edge lengths and remove duplicates!
    edges = np.sort(triangles[:, [0, 1]])  # Sort the vertices by index, take only the first and second vertex
    edges = np.unique(edges, axis=0)  # Only keep the unique ones
    edges = np.vstack((edges, np.sort(triangles[:, [1, 2]])))
    edges = np.unique(edges, axis=0)
    edges = np.vstack((edges, np.sort(triangles[:, [2, 0]])))
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return edge_lengths


def clean_mesh_simple(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh], verbose: bool = True):
    """
    Clean a mesh using simple mesh cleaning actions:
    - Remove duplicated vertices
    - Remove unreferenced vertices (i.e. vertices not in any triangle/tetra)
    - Remove duplicated triangles/tetras
    - Remove degenerate triangles/tetras (i.e. triangles/tetras that reference the same vertex multiple times)

    These actions are executed in the order displayed above.

    :param mesh: The mesh to clean.
    :param verbose: Whether to print the results.
    :return: Nothing.
    """
    is_triangle_mesh = check_mesh_type(mesh=mesh)

    if verbose:
        print(f"Cleaning mesh (Simple)...")

    start_time = time.time()
    nvo, nto = get_mesh_verts_and_tris(mesh=mesh)

    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if is_triangle_mesh:
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
    else:
        mesh.remove_duplicated_tetras()
        mesh.remove_degenerate_tetras()

    nvc, ntc = get_mesh_verts_and_tris(mesh=mesh)
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        form = "tris" if is_triangle_mesh else "tetras"
        print(f"Cleaned mesh (simple) ({nvo} -> {nvc} verts, {nto} -> {ntc} {form}) [{elapsed}s]")


def clean_mesh_metric(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh],
                      metric: str = "aspect_ratio",
                      quantile: float = 0.95,
                      absolute: float = 1000,
                      verbose: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clean a mesh using threshold on a certain metric.

    :param mesh: The mesh to clean.
    :param metric: Which metric to clean the mesh up with. Can be either "aspect_ratio" / "ar" or "edge_length" / "el"
    :param quantile: The quantile to calculate the threshold on. The lower value of this and the absolute value is \
     used. Set to 0 to disable.
    :param absolute: The absolute threshold. The lower value of this and the quantile value is used. Set to 0 to disable.
    :param verbose: Whether to print the progress and result.
    :return: A tuple of numpy arrays containing the metric calculated for all vertices, edges or triangles at index 0 \
     and the cleaned subset/subarray at index 1. Returns None if any error occurs.
    """

    is_triangle_mesh = check_mesh_type(mesh=mesh)
    if not is_triangle_mesh:
        print(f"Cannot clean a mesh that is not an instance of {type(open3d.geometry.TriangleMesh)}. Returning")
        return None

    if quantile <= 0.0 and absolute <= 0.0:
        print("Both quantile and absolute value are 0. Returning None.")
        return None

    available_metrics = ["aspect_ratio", "ar", "edge_length", "el"]

    if not isinstance(metric, str) or not (metric.lower() in available_metrics):
        print(f"Invalid specified metric {metric}. Must be one of {available_metrics}. Defaulting to aspect ratios.")
        metric = "aspect_ratio"

    if verbose:
        print(f"Cleaning mesh ({metric})...  Thresholds: Quantile={quantile}, Absolute={absolute})")

    start_time = time.time()
    nvo, nto = get_mesh_verts_and_tris(mesh=mesh)
    quantile = min(1.0, max(quantile, 0.0))
    metric_all = None
    metric_cleaned = None

    if quantile > 0 or absolute > 0:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        edge_lengths = utils.get_mesh_edge_lengths(vertices=vertices, triangles=triangles)
        if metric == "aspect_ratio" or metric == "ar":
            metric_all = utils.aspect_ratios(edge_lengths=edge_lengths)
        elif metric == "edge_length" or metric == "el":
            metric_all = edge_lengths

        if quantile > 0 and absolute > 0:
            threshold = min(np.quantile(metric_all, quantile), absolute)
        elif quantile > 0:
            threshold = np.quantile(metric_all, quantile)
        else:
            threshold = absolute
        print(f"Actual threshold: {threshold}")
        mesh.remove_triangles_by_mask(metric_all > threshold)
        metric_cleaned = metric_all[metric_all <= threshold]
        mesh.remove_unreferenced_vertices()

    if verbose:
        print_cleaning_result(mesh=mesh, start_time=start_time, vertices_before=nvo, simplices_before=nto)

    return metric_all, metric_cleaned


def get_mesh_triangle_normal_deviations(triangles: np.ndarray, triangle_normals: np.ndarray):
    triangles_sorted = np.sort(triangles, axis=1)
    triangle_count = len(triangles_sorted)
    triangle_indices = np.arange(triangle_count)

    # Add the indices to the sorted triangles
    triangles_sorted = np.hstack((triangle_indices[:, np.newaxis], triangles_sorted))
    print(f"Calculating normal deviations")
    dots = []

    for i in range(triangle_count):
        current_triangle = triangles_sorted[i]
        to_compare_against = triangles_sorted[i + 1:triangle_count]
        compare = triangles_sorted[i] == to_compare_against
        sums = np.sum(compare, axis=1)
        adjacent = to_compare_against[sums == 2]
        if len(adjacent) == 0:
            continue

        current_triangle_normal = triangle_normals[current_triangle[0]]
        adjacent_normals = triangle_normals[adjacent[:, 0]]
        dots += np.clip(np.dot(adjacent_normals, current_triangle_normal), -1.0, 1.0).tolist()

    return np.degrees(np.arccos(dots))


def get_points_to_mesh_distances(points: np.ndarray, mesh: TriangleMesh):
    """
    Compute the distances from a set of points to the closest points on the surface of a mesh.

    :param points: A (n, 3) numpy array containing the query point coordinates.
    :param mesh: An Open3D triangulated mesh which represents the target for the distance calculations.
    :returns: A 1D numpy array containing the distances from the query points to the closest point on the target mesh.
    """

    rcs = RaycastingScene()
    # noinspection PyArgumentList
    tensor_mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rcs.add_triangles(tensor_mesh)
    pts = Tensor(np.array(points).astype(dtype=np.float32))
    closest_points = rcs.compute_closest_points(pts)['points'].numpy()
    distances_pts_to_mesh = np.linalg.norm(points - closest_points, axis=1)
    return distances_pts_to_mesh


def get_distances_closest_point(x, y):
    dists_y_to_x, corrs_y_to_x = k_nearest_neighbors(y, x, k=1, squared_distances=False)
    dists_x_to_y = np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=2)
    return dists_x_to_y


def aspect_ratios(edge_lengths: np.ndarray) -> np.ndarray:
    # Calculate aspect ratio for each triangle
    min_edge_lengths = np.min(edge_lengths, axis=1)
    max_edge_lengths = np.max(edge_lengths, axis=1)
    min_edge_lengths[min_edge_lengths == 0] = np.finfo(float).eps  # Handle cases where min edge length is zero

    # Return 0 where the min edge length is 0. Return max / min where min != 0
    result = np.where(min_edge_lengths <= np.finfo(float).eps, 0.0, max_edge_lengths / min_edge_lengths)
    return result


def aspect_ratios(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    # Compute edge lengths for each triangle
    edge_lengths = np.zeros((len(triangles), 3))
    for i in range(3):
        v0 = vertices[triangles[:, i]]
        v1 = vertices[triangles[:, (i + 1) % 3]]
        edge_lengths[:, i] = np.linalg.norm(v0 - v1, axis=1)

    return aspect_ratios(edge_lengths=edge_lengths)


def check_mesh_type(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh]) -> bool:
    is_triangle_mesh = isinstance(mesh, open3d.geometry.TriangleMesh)
    is_tetra_mesh = isinstance(mesh, open3d.geometry.TetraMesh)

    if not is_triangle_mesh and not is_tetra_mesh:
        raise TypeError(
            f"mesh must be of type {type(open3d.geometry.TriangleMesh)} or {type(open3d.geometry.TetraMesh)}")

    return is_triangle_mesh


def get_mesh_verts_and_tris(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh], digits: int = 2) -> \
        Tuple[str, str]:
    """
    Get a formatted containing the amount of triangles/tetras and vertices of a mesh.

    :param mesh: The mesh from which to calculate these stats
    :param digits: The amount of digits to which to round the amounts
    :return: A tuple of strings, with index 0 being the number of vertices and index 1 the number of triangles/tetras
    """
    is_triangle_mesh = check_mesh_type(mesh=mesh)
    nvo = format_number(len(mesh.vertices), digits=digits)  # Number of Vertices in Original
    nto = format_number(len(mesh.triangles), digits=digits) if is_triangle_mesh else format_number(len(mesh.tetras))
    return nvo, nto


def print_cleaning_result(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh],
                          start_time: float,
                          vertices_before: str,
                          simplices_before: str):
    is_triangle_mesh = check_mesh_type(mesh=mesh)

    nvc, ntc = get_mesh_verts_and_tris(mesh=mesh)
    end_time = time.time()
    elapsed = round(end_time - start_time, 3)
    form = "tris" if is_triangle_mesh else "tetras"
    print(f"Cleaned mesh ({vertices_before} -> {nvc} verts, {simplices_before} -> {ntc} {form}) [{elapsed}s]")
