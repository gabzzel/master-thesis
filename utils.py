import numpy as np
from open3d.geometry import TriangleMesh
import time


def format_number(number, digits=1):
    if number >= 1000000000:
        return str(round(number / 1000000000, digits)) + "B"

    if number >= 1000000:
        return str(round(number / 1000000, digits)) + "M"

    if number >= 1000:
        return str(round(number / 1000, digits)) + "K"

    return str(number)


def get_stats(a: np.array, name: str, print_only=True, round_digits=3):
    _max = round(np.max(a), round_digits)
    _min = round(np.min(a), round_digits)
    avg = round(np.average(a), round_digits)
    med = round(np.median(a), round_digits)
    std = round(np.std(a), round_digits)

    if print_only:
        print(f"{name} stats: Max={_max}, Min={_min}, Avg/Mean={avg}, Med={med}, Std={std}")
    else:
        return _max, _min, avg, med, std


def get_mesh_aspect_ratios(mesh: TriangleMesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    # Compute edge lengths for each triangle
    edge_lengths = np.zeros((len(triangles), 3))
    for i in range(3):
        v0 = vertices[triangles[:, i]]
        v1 = vertices[triangles[:, (i + 1) % 3]]
        edge_lengths[:, i] = np.linalg.norm(v0 - v1, axis=1)

    # Calculate aspect ratio for each triangle
    min_edge_lengths = np.min(edge_lengths, axis=1)
    max_edge_lengths = np.max(edge_lengths, axis=1)
    min_edge_lengths[min_edge_lengths == 0] = np.finfo(float).eps  # Handle cases where min edge length is zero
    aspect_ratios = max_edge_lengths / min_edge_lengths
    return aspect_ratios


def clean_mesh(mesh: TriangleMesh, ar_quantile_threshold=0.95, ar_abs_threshold=1000, verbose=True):
    """
    Clean up a mesh:
    - Remove unreferenced vertices
    - Remove duplicated triangles
    - Remove degenerate triangles (i.e. triangles that reference the same vertex multiple times)
    - Optionally remove all triangles with a large aspect ratio.

    :param ar_quantile_threshold: Every triangle with an aspect ratio in the 'quantile' above
    this threshold will be removed. Set to 0 to ignore.
    :param ar_abs_threshold: Every triangle with an aspect ratio above this absolute value will
    be removed. Set to 0 to ignore.
    """

    if verbose:
        print(f"Cleaning mesh... (Aspect Ratio Thresholds: Quantile={ar_quantile_threshold}," +
              f"Absolute={ar_abs_threshold})")

    start_time = time.time()
    nvo = len(mesh.vertices)  # Number of Vertices in Original
    nto = len(mesh.triangles)

    # Do some obvious cleanups
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    ar_quantile_threshold = min(1.0, max(ar_quantile_threshold, 0.0))
    aspect_ratios = None
    if ar_quantile_threshold > 0 or ar_abs_threshold > 0:
        aspect_ratios = get_mesh_aspect_ratios(mesh)

        threshold = 0
        if ar_quantile_threshold > 0 and ar_abs_threshold > 0:
            threshold = min(np.quantile(aspect_ratios, ar_quantile_threshold), ar_abs_threshold)
        elif ar_quantile_threshold > 0:
            threshold = np.quantile(aspect_ratios, ar_quantile_threshold)
        else:
            threshold = ar_abs_threshold

        triangles_to_remove = aspect_ratios >= threshold
        mesh.remove_triangles_by_mask(triangles_to_remove)

    nvc = len(mesh.vertices)
    ntc = len(mesh.triangles)
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        print(f"Cleaned mesh ({format_number(nvo)} -> {format_number(nvc)} verts, {format_number(nto)} -> {format_number(ntc)} tris) [{elapsed}s]")

    return aspect_ratios
