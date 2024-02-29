import math
from typing import Union

import numpy as np
import open3d
from open3d.cpu.pybind.core import Tensor
from open3d.cpu.pybind.t.geometry import RaycastingScene
from open3d.geometry import TriangleMesh
import time
from point_cloud_utils import k_nearest_neighbors

import mesh_quality
from mesh_quality import aspect_ratios


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


def clean_mesh(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh],
               aspect_ratio_quantile_threshold: float = 0.95,
               aspect_ratio_abs_threshold: float = 1000,
               verbose: bool = True):
    """
    Clean up a mesh: \n
    - Remove unreferenced vertices \n
    - Remove duplicated triangles \n
    - Remove degenerate triangles (i.e. triangles that reference the same vertex multiple times) \n
    - Optionally remove all triangles with a large aspect ratio.

    :param verbose: Whether to print the progress.
    :param mesh: The mesh to clean up.
    :param aspect_ratio_quantile_threshold: Every triangle with an aspect ratio in the 'quantile' above this threshold will be removed. Set to 0 to ignore.
    :param aspect_ratio_abs_threshold: Every triangle with an aspect ratio above this absolute value will be removed. Set to 0 to ignore.
    """

    is_triangle_mesh = isinstance(mesh, open3d.geometry.TriangleMesh)
    is_tetra_mesh = isinstance(mesh, open3d.geometry.TetraMesh)

    if not is_triangle_mesh and not is_tetra_mesh:
        raise TypeError(
            f"mesh must be of type {type(open3d.geometry.TriangleMesh)} or {type(open3d.geometry.TetraMesh)}")

    if verbose:
        print(f"Cleaning mesh... (Aspect Ratio Thresholds: Quantile={aspect_ratio_quantile_threshold}," +
              f"Absolute={aspect_ratio_abs_threshold})")

    start_time = time.time()

    nvo = format_number(len(mesh.vertices), 2)  # Number of Vertices in Original
    nto = format_number(len(mesh.triangles), 2) if is_triangle_mesh else format_number(len(mesh.tetras))

    # Do some obvious cleanups
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if is_triangle_mesh:
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
    else:
        mesh.remove_duplicated_tetras()
        mesh.remove_degenerate_tetras()

    # Remove all aspect ratios that exceed the threshold(s)
    aspect_ratio_quantile_threshold = min(1.0, max(aspect_ratio_quantile_threshold, 0.0))
    all_aspect_ratios = None  # Aspect ratios.
    aspect_ratios_remaining = None

    # Removing triangles by aspect ratio is only applicable on triangle meshes
    if is_triangle_mesh and (aspect_ratio_quantile_threshold > 0 or aspect_ratio_abs_threshold > 0):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        all_aspect_ratios = mesh_quality.aspect_ratios(vertices, triangles)

        if aspect_ratio_quantile_threshold > 0 and aspect_ratio_abs_threshold > 0:
            threshold = min(np.quantile(all_aspect_ratios, aspect_ratio_quantile_threshold), aspect_ratio_abs_threshold)
        elif aspect_ratio_quantile_threshold > 0:
            threshold = np.quantile(all_aspect_ratios, aspect_ratio_quantile_threshold)
        else:
            threshold = aspect_ratio_abs_threshold
        print(f"Actual aspect ratio threshold: {threshold}")
        mesh.remove_triangles_by_mask(all_aspect_ratios >= threshold)
        aspect_ratios_remaining = all_aspect_ratios[all_aspect_ratios < threshold]
        mesh.remove_unreferenced_vertices()

    nvc = format_number(len(mesh.vertices), 2)
    ntc = format_number(len(mesh.triangles), 2) if is_triangle_mesh else format_number(len(mesh.tetras))
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        form = "tris" if is_triangle_mesh else "tetras"
        print(f"Cleaned mesh ({nvo} -> {nvc} verts, {nto} -> {ntc} {form}) [{elapsed}s]")

    return all_aspect_ratios, aspect_ratios_remaining


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
