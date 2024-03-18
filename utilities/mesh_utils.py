from typing import Union, Tuple

import numpy as np
import open3d
import point_cloud_utils
from open3d.cpu.pybind.core import Tensor
from open3d.cpu.pybind.t.geometry import RaycastingScene

from utilities import utils


def get_edge_lengths_flat(vertices: np.ndarray, triangles: np.ndarray):
    """
    Compute the edge lengths given a set of vertices / coordinates and a set of triangles with vertex indices.
    Returns a flat representation of all edge lengths, with the edge lengths being sorted and guaranteed unique.

    :param vertices: The vertices of the triangle mesh for which to calculate the edge lengths.
    :param triangles: The triangles of the triangle mesh containing the indices of the vertices.
    :return: A 1D numpy array containing all the edge lengths of the triangle mesh.
    """

    edges = np.sort(triangles[:, [0, 1]])  # Sort the vertices by index, take only the first and second vertex
    edges = np.unique(edges, axis=0)  # Only keep the unique ones
    edges = np.vstack((edges, np.sort(triangles[:, [1, 2]])))
    edges = np.unique(edges, axis=0)
    edges = np.vstack((edges, np.sort(triangles[:, [2, 0]])))
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return edge_lengths


def get_edge_lengths_per_triangle(vertices: np.ndarray, triangles: np.ndarray):
    """
    Compute the edge lengths given a set of vertices / coordinates and a set of triangles with vertex indices,
    Returns a 2D numpy array containing the three edge lengths per triangle.
    :param vertices: A 1D numpt array containing the vertices of the triangle mesh for which to compute the edge \
    lengths.
    :param triangles: A 2D numpy array containing the triangles of the triangle mesh for which to compute the edge \
    lengths.
    :return: A 2D numpy array containing the edge lengths for each triangle.
    """
    edge_lengths = np.zeros((len(triangles), 3))
    for i in range(3):
        v0 = vertices[triangles[:, i]]
        v1 = vertices[triangles[:, (i + 1) % 3]]
        edge_lengths[:, i] = np.linalg.norm(v0 - v1, axis=1)
    return edge_lengths


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


def get_points_to_mesh_distances(points: np.ndarray, mesh: open3d.geometry.TriangleMesh) -> np.ndarray:
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
    dists_y_to_x, corrs_y_to_x = point_cloud_utils.k_nearest_neighbors(y, x, k=1, squared_distances=False)
    dists_x_to_y = np.linalg.norm(x[corrs_y_to_x] - y, axis=-1, ord=2)
    return dists_x_to_y


def aspect_ratios_edge_lengths(edge_lengths: np.ndarray) -> np.ndarray:
    # Calculate aspect ratio for each triangle
    min_edge_lengths = np.min(edge_lengths, axis=1)
    max_edge_lengths = np.max(edge_lengths, axis=1)
    min_edge_lengths[min_edge_lengths == 0] = np.finfo(float).eps  # Handle cases where min edge length is zero

    # Return 0 where the min edge length is 0. Return max / min where min != 0
    result = np.where(min_edge_lengths <= np.finfo(float).eps, 0.0, max_edge_lengths / min_edge_lengths)
    return result


def aspect_ratios(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edge_lengths = get_edge_lengths_per_triangle(vertices=vertices, triangles=triangles)
    return aspect_ratios_edge_lengths(edge_lengths=edge_lengths)


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
    nvo = utils.format_number(len(mesh.vertices), digits=digits)  # Number of Vertices in Original
    nto = utils.format_number(len(mesh.triangles), digits=digits) \
        if is_triangle_mesh \
        else utils.format_number(len(mesh.tetras))
    return nvo, nto


