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


def triangle_normal_deviations_naive(triangles, triangle_normals):
    indices = np.arange(len(triangles))

    # Every triangle can only have a maximum of 3 neighbours that share an edge with this triangle.
    # Given this observation, this array will contain the neighbouring indices for each triangle.
    # neighbours = np.full(shape=(len(triangles), 3), fill_value=-1, dtype=int)

    deviations = []
    occurrences = np.zeros(shape=(len(triangles)))

    for tri_index in range(len(triangles)):
        # Since triangles can only have 3 neighbours max, we can skip any triangle we have seen 3 or more times.
        if occurrences[tri_index] >= 3:
            continue

        start_time = time.time()
        triangle_1 = triangles[tri_index]
        normal_1 = triangle_normals[tri_index]

        found_neighbours = 0

        for tri_index_2 in range(tri_index+1, len(triangles)):
            # If we have already encountered this triangle 3 times before, we don't need to check it anymore.
            if occurrences[tri_index] >= 3:
                break

            triangle_2 = triangles[tri_index_2]

            # If the number of intersecting vertex indices is not 2, we can just ignore this.
            if len(np.intersect1d(triangle_1, triangle_2)) != 2:
                continue

            # Calculate the deviation
            normal_2 = triangle_normals[tri_index]
            clipped_dot = np.clip(np.dot(normal_1, normal_2), -1.0, 1.0)
            deviations.append(np.degrees(np.arccos(clipped_dot)))

            # Make sure we count this occurrence!
            occurrences[triangle_2] += 1
            found_neighbours += 1

            # If we have found 3 neighbours for this triangle...
            # We can break out of the loop and skip this triangle forever
            if found_neighbours == 3:
                occurrences[tri_index] = 3
                break


        end_time = time.time()
        print(f"Triangle {tri_index} took {str(round(end_time - start_time, 5))}s")

    return deviations


def triangle_normal_deviations(triangles, triangle_normals):
    indices = np.arange(len(triangles))
    triangles_incl_indices = np.hstack((triangles, indices[:, np.newaxis]))

    # Every triangle can only have a maximum of 3 neighbours that share an edge with this triangle.
    # Given this observation, this array will contain the neighbouring indices for each triangle.
    # neighbours = np.full(shape=(len(triangles), 3), fill_value=-1, dtype=int)


    # This list contains 3 numpy arrays. Each of these inner arrays are 1D, which contain the indices (in order)
    # of the original triangles array. However, the order of the original triangle (array) indices are as if
    # they would be if the original triangles would be ordered by x-th vertex. For example, at index 2, this array
    # contains the 1D (sub) array which contains triangle indices in order of how they should be sorted when sorting
    # on third vertex.
    tris_sort_indices = []

    # This list contains 3 numpy arrays. Each of these numpy arrays contains all triangles, but sorted according
    # to a specific vertex. For example: at index 1, this array contains all triangles, sorted ascending by their
    # second (at index 1) vertex.
    tris_sorted_per_vertex = []

    for target_vertex_index in range(3):
        tris_sort_indices.append(np.argsort(triangles_incl_indices[:, target_vertex_index]))
        tris_sorted_per_vertex.append(triangles_incl_indices[tris_sort_indices[target_vertex_index]])

    deviations = []
    occurrences = np.zeros(shape=(len(triangles)))
    start_time = time.time()
    for tri_index in range(len(triangles)):
        # Since triangles can only have 3 neighbours max, we can skip any triangle we have seen 3 or more times.
        if occurrences[tri_index] >= 3:
            continue

        triangle_1 = triangles[tri_index]
        normal_1 = triangle_normals[tri_index]

        found_neighbours = 0
        candidates = {}

        for target_vertex_index in range(3):
            # The vertex we want to search for in other triangles whether they have it as well.
            target_vertex = triangles[tri_index][target_vertex_index]

            for search_vertex_index in range(3):
                search_range = tris_sorted_per_vertex[search_vertex_index][:, search_vertex_index]

                # The first and last index in the *sorted* triangles array that also have the target vertex
                candidate_index_min = np.searchsorted(a=search_range, v=target_vertex, side='left')
                candidate_index_max = np.searchsorted(a=search_range, v=target_vertex, side='right')

                # If we have found zero
                if candidate_index_max - candidate_index_min == 0:
                    break

                # The indices of all triangles in the *sorted* triangles array that also have this vertex
                candidates_indices_in_sorted = np.arange(start=candidate_index_min, stop=candidate_index_max)
                # Convert the indices from the sorted array back to the original indices
                candidate_indices = tris_sorted_per_vertex[search_vertex_index][candidates_indices_in_sorted][:, 3]

                for candidate_index in candidate_indices:
                    if candidate_index == tri_index:
                        continue
                    if candidate_index in candidates.keys():
                        candidates[candidate_index] += 1
                    else:
                        candidates[candidate_index] = 1

        for (candidate_index, count) in candidates.items():
            if count != 2:
                continue

            normal_2 = triangles[candidate_index]
            clipped_dot = np.clip(np.dot(normal_1, normal_2), -1.0, 1.0)
            deviations.append(np.degrees(np.arccos(clipped_dot)))

            found_neighbours += 1
            if found_neighbours == 3:
                break

    end_time = time.time()
    print(f"Normal deviations took {str(round(end_time - start_time, 5))}s")

    return deviations