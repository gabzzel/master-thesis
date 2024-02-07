import numpy as np
from open3d.geometry import TriangleMesh
import time
import trimesh


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


def get_mesh_aspect_ratios(vertices: np.ndarray, triangles: np.ndarray):
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

    # Remove all aspect ratios that exceed the threshold(s)
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
        print(f"Actual aspect ratio threshold: {threshold}")
        triangles_to_remove = aspect_ratios >= threshold
        mesh.remove_triangles_by_mask(triangles_to_remove)
        aspect_ratios = aspect_ratios[aspect_ratios < threshold]
        mesh.remove_unreferenced_vertices()

    nvc = len(mesh.vertices)
    ntc = len(mesh.triangles)
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        print(f"Cleaned mesh ({format_number(nvo)} -> {format_number(nvc)} verts, {format_number(nto)} -> {format_number(ntc)} tris) [{elapsed}s]")

    # TODO, remove aspect ratios outside threshold before retunring!
    return aspect_ratios


def get_mesh_triangle_normal_deviations(mesh: TriangleMesh):
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

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


def get_mesh_curvature(vertices, vertex_normals, triangles, triangle_normals, sample_ratio=0.01, radius=0.1):
    """
    Get the discrete mean curvature of the mesh.

    :param vertices: The (n, 3) numpy array containing the vertices (coordinates) of the mesh
    :param vertex_normals: The (n, 3) numpy array containing the vertex normals
    :param triangles: The (n, 3) numpy array containing the triangles (i.e. vertex indices)
    :param triangle_normals: The (n, 3) numpy array containing the triangle normals
    :param sample_ratio: The ratio of points used to estimate curvature, e.g. 0.1 = 10% points are used
    :param radius: The neighbour search radius used during curvature estimation. Higher values give a more global indication of the curvature, while lower values give a more fine-grained indication.
    :returns: A 1D numpy array containing the curvatures for each sampled point.
    """
    sample_ratio = max(0.0, min(1.0, sample_ratio))
    t_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, face_normals=triangle_normals,
                             vertex_normals=vertex_normals)
    rng = np.random.default_rng()
    chosen_points = rng.choice(a=vertices, size=int(sample_ratio * len(vertices)))
    curvature = trimesh.curvature.discrete_mean_curvature_measure(t_mesh, points=chosen_points, radius=radius)
    return curvature
