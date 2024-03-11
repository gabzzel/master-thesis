import bisect
import time
from typing import Optional, Union

import numpy as np
import open3d
import trimesh
from matplotlib import pyplot as plt
import point_cloud_utils as pcu

from utilities import mesh_utils, utils
from utilities.enumerations import MeshEvaluationMetric


def get_mesh_quality_metric(text: str) -> Optional[MeshEvaluationMetric]:
    t = text.lower().strip()
    t = t.replace(" ", "_")

    if t == "edge_lengths" or t == "edge_length" or t == "el":
        return MeshEvaluationMetric.EDGE_LENGTHS
    elif t == "triangle_aspect_ratios" or t == "aspect_ratio" or t == "aspect_ratios" or t == "ar":
        return MeshEvaluationMetric.TRIANGLE_ASPECT_RATIOS
    elif t == "triangle_normal_deviations" or t == "normal_deviations" or t == "normal_deviation" or t == "nd":
        return MeshEvaluationMetric.TRIANGLE_NORMAL_DEVIATIONS
    elif t == "discrete_curvature" or t == "curvature" or t == "dc":
        return MeshEvaluationMetric.DISCRETE_CURVATURE
    elif t == "connectivity" or t == "conn" or t == "co" or t == "c":
        return MeshEvaluationMetric.CONNECTIVITY

    return None


def edge_lengths(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return mesh_utils.get_edge_lengths_flat(vertices=vertices, triangles=triangles)


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


def triangle_normal_deviations_naive(triangles, triangle_normals):
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

        for tri_index_2 in range(tri_index + 1, len(triangles)):
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


def triangle_normal_deviations_sorted(triangles, triangle_normals):
    indices = np.arange(len(triangles))
    triangles_incl_indices = np.hstack((triangles, indices[:, np.newaxis]))

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
            # Scalar: The vertex we want to search for in other triangles whether they have it as well.
            target_vertex = triangles[tri_index][target_vertex_index]

            for search_vertex_index in range(3):
                # 1D array containing the vertices of all triangles at a specific index within the triangle.
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


def triangle_normal_deviations_adjacency(adjacency_list, triangles: np.ndarray, triangle_normals):
    start_time = time.time()
    indices = np.arange(len(triangles))
    triangles_incl_indices = np.hstack((triangles, indices[:, np.newaxis]))

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
    tris_sorted_per_vertex_single = []

    # Use sorting to improve the speed dramatically of lookups
    for i in range(3):
        # The indices of the triangles, if they were sorted by vertices on vertex index 0, 1 or 2
        argsort = np.argsort(triangles_incl_indices[:, i])
        tris_sort_indices.append(argsort)

        # The triangles when they are sorted by that specific vertex index
        triangles_sorted_by_vertex = triangles_incl_indices[tris_sort_indices[i]]
        tris_sorted_per_vertex.append(triangles_sorted_by_vertex)
        # The same, but only keeping the vertex itself saved
        single = triangles_sorted_by_vertex[:, i]
        tris_sorted_per_vertex_single.append(single.tolist())

    deviations = []
    triangle_count = len(triangles)
    found_indices_min = [0, 0, 0]
    for v1 in range(len(adjacency_list)):
        vertex_neighbours: set = adjacency_list[v1]
        if len(vertex_neighbours) == 0:
            continue

        T = None

        # Find all triangles with v1 at index i
        for i in range(3):
            search_range = tris_sorted_per_vertex_single[i]
            found_index_min = bisect.bisect_left(search_range, v1, lo=found_indices_min[i], hi=triangle_count)
            found_indices_min[i] = found_index_min
            found_index_max = bisect.bisect_right(search_range, v1, lo=found_index_min, hi=triangle_count)

            if found_index_min == found_index_max:  # No triangles found.
                continue

            indices_in_sorted: np.ndarray = indices[found_index_min:found_index_max]

            if T is None:
                T = set(tris_sorted_per_vertex[i][indices_in_sorted][:, 3])
            else:
                T = T.union(tris_sorted_per_vertex[i][indices_in_sorted][:, 3])

        # Find all triangles that have both v1 and its neighbour v2
        for v2 in vertex_neighbours:

            # We only need to search every pair of vertices once. We can remove this.
            adjacency_list[v2].remove(v1)

            t1 = None
            t2 = None

            for triangle_with_v1_index in T:
                t = triangles[triangle_with_v1_index]
                if t[0] != v2 and t[1] != v2 and t[2] != v2:
                    continue

                # We have found a triangle with both v1 and v2!
                if t1 is None:
                    t1 = triangle_normals[triangle_with_v1_index]
                elif t2 is None:
                    t2 = triangle_normals[triangle_with_v1_index]

                # Only 2 triangles can share an edge.
                else:
                    break

            # If we did not find two triangles that share the edge between v1 and v2, we can just continue.
            if t1 is None or t2 is None:
                continue

            # Make sure dot value is valid.
            dot = np.dot(t1, t2)
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            deviations.append(np.degrees(np.arccos(dot)))

    end_time = time.time()
    print(f"Normal deviations calculation took {round(end_time - start_time, 3)}s")
    return deviations


def evaluate_connectivity(triangles, vertices, verbose: bool = True):
    # cv is the index of the connected component of each vertex
    # nv is the number of vertices per component
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    cv, nv, cf, nf = pcu.connected_components(vertices, triangles)
    # If nf is a 0-dimensional array or has length 1, there is only a single component
    if nf.ndim == 0 or len(nf) == 1:
        num_conn_comp = 1
        largest_component_ratio = 1.0
    else:
        num_conn_comp = len(nf)  # Number of connected components (according to triangles)
        largest_component_ratio = round(np.max(nf) / np.sum(nf), 3)

    if verbose:
        print(f"Connectivity: Connected Components={num_conn_comp}, largest component ratio={largest_component_ratio}")


def evaluate_point_cloud_mesh(point_cloud: open3d.geometry.PointCloud,
                              mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh]):
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)

    # Compute Hausdorff and Chamfer distances
    hausdorff = round(pcu.hausdorff_distance(pcd_points, mesh_points), 4)
    chamfer = round(pcu.chamfer_distance(pcd_points, mesh_points), 4)
    print(f"Hausdorff={hausdorff}, Chamfer={chamfer}")

    distances_pts_to_mesh = mesh_utils.get_points_to_mesh_distances(pcd_points, mesh)
    distance_results = utils.get_stats(distances_pts_to_mesh, "Point-to-mesh distances", return_results=True)

    distances_normalized = distances_pts_to_mesh / distance_results[0]  # Index 0 is the max distance.
    colors = np.full(shape=(len(distances_normalized), 3), fill_value=0.0)
    colors[:, 0] = distances_normalized
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    plt.hist(distances_pts_to_mesh, histtype='step', log=True, bins=100)
