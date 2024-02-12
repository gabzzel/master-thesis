import time

import trimesh
import numpy as np


def discrete_curvature(vertices, vertex_normals, triangles, triangle_normals, sample_ratio=0.01, radius=0.1):
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
    if sample_ratio < 1.0:
        chosen_points = rng.choice(a=vertices, size=int(sample_ratio * len(vertices)))
    else:
        chosen_points = vertices
    curvature = trimesh.curvature.discrete_mean_curvature_measure(t_mesh, points=chosen_points, radius=radius)
    return curvature


def aspect_ratios(vertices: np.ndarray, triangles: np.ndarray):
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

    # Return 0 where the min edge length is 0. Return max / min where min != 0
    result = np.where(min_edge_lengths <= np.finfo(float).eps, 0.0, max_edge_lengths / min_edge_lengths)
    return result


def triangle_normal_deviations(triangles, triangle_normals):
    indices = np.arange(len(triangles))
    triangles_sorted_within = np.sort(triangles, axis=1)
    triangles_incl_indices = np.hstack((triangles_sorted_within, indices[:, np.newaxis]))

    # Every triangle can only have a maximum of 3 neighbours that share an edge with this triangle.
    # Given this observation, this array will contain the neighbouring indices for each triangle.
    # neighbours = np.full(shape=(len(triangles), 3), fill_value=-1, dtype=int)

    sort_indices_0 = np.argsort(triangles_incl_indices[:, 0], axis=0)
    triangles_sorted_0 = triangles_incl_indices[sort_indices_0]
    sort_indices_1 = np.argsort(triangles_incl_indices[:, 1], axis=0)
    triangles_sorted_1 = triangles_incl_indices[sort_indices_1]
    sort_indices_2 = np.argsort(triangles_incl_indices[:, 2], axis=0)
    triangles_sorted_2 = triangles_incl_indices[sort_indices_2]

    triangles_sorted_list = [triangles_sorted_0, triangles_sorted_1, triangles_sorted_2]
    start_time = time.time()

    deviations = []
    occurrences = np.zeros(shape=(len(triangles)))

    for tri_index in range(len(triangles)):
        if occurrences[tri_index] >= 3:
            continue

        start_time = time.time()
        candidates = {}

        for i in range(3):
            # The 2D array containing the first vertex index for each of the triangles.
            search_range = triangles_sorted_list[i][:, i]

            # The vertex we want to search for in other triangles whether they have it as well.
            target_vertex = triangles_sorted_within[tri_index][i]

            # The first and last index in the *sorted* triangles array that also have the target vertex
            candidate_index_min = np.searchsorted(a=search_range, v=target_vertex, side='left')
            candidate_index_max = np.searchsorted(a=search_range, v=target_vertex, side='right')

            if candidate_index_max - candidate_index_min == 0:
                break

            # The indices of all triangles in the *sorted* triangles array that also have this vertex
            candidates_indices_in_sorted = np.arange(start=candidate_index_min,  stop=candidate_index_max)
            # Convert the indices from the sorted array back to the original indices
            candidate_indices = triangles_sorted_list[i][candidates_indices_in_sorted][:, 3]

            for candidate_index in candidate_indices:
                if candidate_index == tri_index:
                    continue

                if candidate_index in candidates.keys():
                    candidates[candidate_index] += 1
                else:
                    candidates[candidate_index] = 1

        for (other_index, count) in candidates.items():
            if count < 2:
                continue

            occurrences[other_index] += 1

            normal_1 = triangle_normals[tri_index]
            normal_2 = triangle_normals[other_index]
            clipped = np.clip(np.dot(normal_1, normal_2), -1.0, 1.0)
            angle = np.degrees(np.arccos(clipped))
            deviations.append(angle)

        end_time = time.time()
        print(f"Triangle {tri_index} times took {str(round(end_time - start_time, 5))}s")

    return deviations
