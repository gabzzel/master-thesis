import math

import numpy as np
import open3d
from open3d.cpu.pybind.core import Tensor
from open3d.cpu.pybind.t.geometry import RaycastingScene
from open3d.geometry import TriangleMesh
import time
import trimesh
from point_cloud_utils import k_nearest_neighbors


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
    med = round(np.median(a=a), round_digits)
    std = round(np.std(a=a), round_digits)

    if print_results:
        print(f"{name} stats: Max={_max}, Min={_min}, Avg/Mean={avg}, Med={med}, Std={std}")
    if return_results:
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


def clean_mesh(mesh: TriangleMesh,
               aspect_ratio_quantile_threshold: float = 0.95,
               aspect_ratio_abs_threshold: float = 1000,
               verbose: bool = True):
    """
    Clean up a mesh: \n
    - Remove unreferenced vertices \n
    - Remove duplicated triangles \n
    - Remove degenerate triangles (i.e. triangles that reference the same vertex multiple times) \n
    - Optionally remove all triangles with a large aspect ratio. \n

    :param verbose: Whether to print the progress.
    :param mesh: The mesh to clean up.
    :param aspect_ratio_quantile_threshold: Every triangle with an aspect ratio in the 'quantile' above
    this threshold will be removed. Set to 0 to ignore.
    :param aspect_ratio_abs_threshold: Every triangle with an aspect ratio above this absolute value will
    be removed. Set to 0 to ignore.
    """

    if verbose:
        print(f"Cleaning mesh... (Aspect Ratio Thresholds: Quantile={aspect_ratio_quantile_threshold}," +
              f"Absolute={aspect_ratio_abs_threshold})")

    start_time = time.time()
    nvo = format_number(len(mesh.vertices), 2)  # Number of Vertices in Original
    nto = format_number(len(mesh.triangles), 2)

    # Do some obvious cleanups
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # Remove all aspect ratios that exceed the threshold(s)
    aspect_ratio_quantile_threshold = min(1.0, max(aspect_ratio_quantile_threshold, 0.0))
    aspect_ratios = None
    aspect_ratios_remaining = None
    if aspect_ratio_quantile_threshold > 0 or aspect_ratio_abs_threshold > 0:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        aspect_ratios = get_mesh_aspect_ratios(vertices, triangles)

        threshold = 0
        if aspect_ratio_quantile_threshold > 0 and aspect_ratio_abs_threshold > 0:
            threshold = min(np.quantile(aspect_ratios, aspect_ratio_quantile_threshold), aspect_ratio_abs_threshold)
        elif aspect_ratio_quantile_threshold > 0:
            threshold = np.quantile(aspect_ratios, aspect_ratio_quantile_threshold)
        else:
            threshold = aspect_ratio_abs_threshold
        print(f"Actual aspect ratio threshold: {threshold}")
        triangles_to_remove = aspect_ratios >= threshold
        mesh.remove_triangles_by_mask(triangles_to_remove)
        aspect_ratios_remaining = aspect_ratios[aspect_ratios < threshold]
        mesh.remove_unreferenced_vertices()

    nvc = format_number(len(mesh.vertices), 2)
    ntc = format_number(len(mesh.triangles), 2)
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        print(f"Cleaned mesh ({nvo} -> {nvc} verts, {nto} -> {ntc} tris) [{elapsed}s]")

    return aspect_ratios, aspect_ratios_remaining


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


def get_mesh_triangle_normal_deviations_v2(triangles: np.ndarray, triangle_normals: np.ndarray, chunk_size=100):

    # Prepare the triangles array
    triangles_sorted = np.sort(triangles, axis=1)  # Sort the vertex indices within the triangles.
    triangle_count = len(triangles_sorted)
    index_pairs = []
    sort_during = True

    # Expected maximum memory size is triangle count * chunksize * dimensions (3) * integer bits (32)
    # For example, for 300K triangles and chunk size default 100, expected memory size = 300K*100*3*32/8=720MB
    num_chunks = int(math.ceil(triangle_count / chunk_size))
    for i in range(0, triangle_count, chunk_size):
        print(f"Processing chunk {int(i/chunk_size)+1} of {num_chunks}...")
        chunk = triangles_sorted[i:i + chunk_size]

        # Find out where the current chunk and all triangles are the same and take the sum
        sums = np.sum(chunk[:, np.newaxis] == triangles_sorted, axis=2)

        # Find the indices for both axes where the result (i.e. sums) are 2 (which means the triangles share 2 vertices)
        indices = np.where(sums == 2)

        # Stack and transpose these indices into the right shape
        stacked = np.transpose(np.vstack((indices[0]+i, indices[1])))

        # Optional! Sort the pairs within, and then get the unique pairs.
        if sort_during:
            stacked = np.unique(np.sort(stacked, axis=1), axis=0)

        index_pairs.extend(stacked.tolist())

    index_pairs = np.unique(np.sort(np.array(index_pairs), axis=1), axis=0)

    # Calculate the angles in degrees between the normals
    normals_left = triangle_normals[index_pairs[:, 0]]
    normals_right = triangle_normals[index_pairs[:, 1]]
    dots = np.sum(normals_left * normals_right, axis=1)
    clipped_dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(clipped_dots))
    return angles


def get_mesh_discrete_curvature(vertices, vertex_normals, triangles, triangle_normals, sample_ratio=0.01, radius=0.1):
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


def get_points_to_mesh_distances(points: np.ndarray, mesh: TriangleMesh):
    """
    Compute the distances from a set of points to the closest points on the surface of a mesh.

    :param points: A (n, 3) numpy array containing the query point coordinates.
    :param mesh: An Open3D triangulated mesh which represents the target for the distance calculations.
    :returns: A 1D numpy array containing the distances from the query points to the closest point on the target mesh.
    """

    rcs = RaycastingScene()
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