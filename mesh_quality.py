import bisect
import time
from typing import Optional, Union, Tuple
import multiprocessing

import numpy as np
import open3d
from open3d.geometry import PointCloud, TriangleMesh, TetraMesh
import trimesh
from matplotlib import pyplot as plt
import point_cloud_utils as pcu

from utilities import mesh_utils, utils
from utilities.enumerations import MeshEvaluationMetric
from utilities.evaluation_results import EvaluationResults
from utilities.enumerations import TriangleNormalDeviationMethod


def get_normal_deviation_method(text: str) -> Optional[TriangleNormalDeviationMethod]:
    t = text.lower().strip()

    if t == "n" or t == "naive":
        return TriangleNormalDeviationMethod.NAIVE
    elif t == "s" or t == "sorted":
        return TriangleNormalDeviationMethod.SORTED
    elif t == "a" or t == "adj" or t == "adjacency":
        return TriangleNormalDeviationMethod.ADJANCENCY

    return None


def get_mesh_quality_metric(text: str) -> Optional[MeshEvaluationMetric]:
    t = text.lower().strip()
    t = t.replace(" ", "_")

    if "edge" in t or t == "el":
        return MeshEvaluationMetric.EDGE_LENGTHS
    elif "aspect" in t or t == "ar":
        return MeshEvaluationMetric.TRIANGLE_ASPECT_RATIOS
    elif "normal" in t or t == "normal_deviations" or t == "normal_deviation" or t == "nd":
        return MeshEvaluationMetric.TRIANGLE_NORMAL_DEVIATIONS
    elif t == "discrete_curvature" or t == "curvature" or t == "dc":
        return MeshEvaluationMetric.DISCRETE_CURVATURE
    elif t == "connectivity" or t == "conn" or t == "co" or t == "c":
        return MeshEvaluationMetric.CONNECTIVITY

    elif t == "hausdorff_distance" or t == "hausdorff" or t == "h":
        return MeshEvaluationMetric.HAUSDORFF_DISTANCE
    elif t == "chamfer_distance" or t == "chamfer" or t == "ch":
        return MeshEvaluationMetric.CHAMFER_DISTANCE
    elif t == "mesh_to_cloud_distance" or t == "m2cd" or t == "mesh_to_cloud":
        return MeshEvaluationMetric.MESH_TO_CLOUD_DISTANCE
    elif t == "all":
        return MeshEvaluationMetric.ALL

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


def triangle_normal_deviations_adjacency(adjacency_list, triangles, triangle_normals, chunk_size, num_workers):

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

    if num_workers is None or num_workers <= 0:
        num_workers = multiprocessing.cpu_count()

    if num_workers == 1:
        return triangle_normal_deviations_adjacency_batch(0,
                                                          len(adjacency_list),
                                                          triangles,
                                                          adjacency_list,
                                                          indices,
                                                          tris_sorted_per_vertex,
                                                          tris_sorted_per_vertex_single,
                                                          triangle_normals)

    args = (triangles, adjacency_list, indices, tris_sorted_per_vertex,
            tris_sorted_per_vertex_single, triangle_normals)

    num_chunks = len(adjacency_list) // num_workers

    if len(adjacency_list) % chunk_size:
        num_chunks += 1

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(adjacency_list))) for i in range(num_chunks)]

    pool = multiprocessing.Pool(processes=num_workers)
    async_results = []

    # Apply async function to each batch
    for start, stop in chunks:
        args = (start, stop, triangles, adjacency_list, indices, tris_sorted_per_vertex, tris_sorted_per_vertex_single, triangle_normals)
        async_result = pool.apply_async(triangle_normal_deviations_adjacency_batch, args=args)
        async_results.append(async_result)

    # Get the results from async results
    results = [async_result.get() for async_result in async_results]

    end_time = time.time()
    print(f"Normal deviations calculation took {round(end_time - start_time, 3)}s")
    return np.concatenate(results)


def triangle_normal_deviations_adjacency_batch(start_index: int,
                                               stop_index: int,
                                               triangles,
                                               adjacency_list,
                                               indices,
                                               tris_sorted_per_vertex,
                                               tris_sorted_per_vertex_single,
                                               triangle_normals):

    deviations = []
    triangle_count = len(triangles)
    found_indices_min = [0, 0, 0]
    for v1 in range(start_index, stop_index):
        vertex_neighbours: set = adjacency_list[v1]
        if len(vertex_neighbours) == 0:
            continue

        T = None

        # Find all triangles with v1 at index i
        for i in range(3):
            found_index_min = bisect.bisect_left(tris_sorted_per_vertex_single[i], v1, lo=found_indices_min[i], hi=triangle_count)
            found_indices_min[i] = found_index_min
            found_index_max = bisect.bisect_right(tris_sorted_per_vertex_single[i], v1, lo=found_index_min, hi=triangle_count)

            if found_index_min == found_index_max:  # No triangles found.
                continue

            if T is None:
                T = set(tris_sorted_per_vertex[i][indices[found_index_min:found_index_max]][:, 3])
            else:
                T = T.union(tris_sorted_per_vertex[i][indices[found_index_min:found_index_max]][:, 3])

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

    return deviations


def triangle_normal_deviations_naive(triangles, triangle_normals, chunk_size=100000, num_workers=4):

    # tris_shared = multiprocessing.Array('d', triangles.ravel())
    # tri_normals_shared = multiprocessing.Array('d', triangle_normals.ravel())

    if num_workers is None or num_workers <= 0:
        num_workers = multiprocessing.cpu_count()

    num_chunks = len(triangles) // chunk_size

    if len(triangles) % chunk_size:
        num_chunks += 1

    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(triangles))) for i in range(num_chunks)]
    pool = multiprocessing.Pool(processes=num_workers)
    async_results = []

    # Apply async function to each batch
    for start, stop in chunks:
        args = (start, stop, triangles, triangle_normals)
        async_result = pool.apply_async(triangle_normal_deviations_naive_batch, args=args)
        async_results.append(async_result)

    # Get the results from async results
    results = [async_result.get() for async_result in async_results]

    return np.concatenate(results)


def triangle_normal_deviations_naive_batch(start_index, stop_index, triangles, triangle_normals):
    # start_index, stop_index, triangles, triangle_normals = args

    triangle_count = len(triangles) // 3
    deviations = []
    occurrences = np.zeros(shape=(triangle_count, ))

    for tri_index in range(start_index, stop_index):
        # Since triangles can only have 3 neighbours max, we can skip any triangle we have seen 3 or more times.
        if occurrences[tri_index] >= 3:
            continue

        tri_index_corrected = tri_index * 3
        triangle_1 = triangles[tri_index_corrected:tri_index_corrected + 3]
        normal_1 = triangle_normals[tri_index_corrected:tri_index_corrected + 3]

        found_neighbours = 0

        for tri_index_2 in range(triangle_count):
            # If we have already encountered this triangle 3 times before, we don't need to check it anymore.
            if occurrences[tri_index] >= 3:
                break

            tri_index_2_corrected = tri_index_2 * 3
            triangle_2 = triangles[tri_index_2_corrected:tri_index_2_corrected + 3]

            # If the number of intersecting vertex indices is not 2, we can just ignore this.
            if len(np.intersect1d(triangle_1, triangle_2)) != 2:
                continue

            # Calculate the deviation
            normal_2 = triangle_normals[tri_index_2_corrected:tri_index_2_corrected + 3]
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

    return np.array(deviations)


def evaluate_connectivity(triangles, vertices, results: EvaluationResults, verbose: bool = True):
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

    results.connectivity_vertices_per_component = nv if isinstance(nv, list) or isinstance(nv, np.ndarray) else [nv]
    results.connectivity_triangles_per_component = nf if isinstance(nf, list) or isinstance(nf, np.ndarray) else [nf]

    if verbose:
        print(f"Connectivity: Connected Components={num_conn_comp}, largest component ratio={largest_component_ratio}")


def evaluate_point_cloud_mesh(point_cloud: PointCloud, mesh: Union[TriangleMesh, TetraMesh]) \
        -> Tuple[float, float, np.ndarray]:
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)

    # Compute Hausdorff and Chamfer distances
    hausdorff = pcu.hausdorff_distance(pcd_points, mesh_points)
    chamfer = pcu.chamfer_distance(pcd_points, mesh_points)
    print(f"Hausdorff={round(hausdorff, 4)}, Chamfer={round(chamfer, 4)}")

    distances_pts_to_mesh = mesh_utils.get_points_to_mesh_distances(pcd_points, mesh)
    utils.get_stats(distances_pts_to_mesh, "Point-to-mesh distances")
    return hausdorff, chamfer, distances_pts_to_mesh

    #distances_normalized = distances_pts_to_mesh / distance_results[0]  # Index 0 is the max distance.
    #colors = np.full(shape=(len(distances_normalized), 3), fill_value=0.0)
    #colors[:, 0] = distances_normalized
    #point_cloud.colors = open3d.utility.Vector3dVector(colors)
    # plt.hist(distances_pts_to_mesh, histtype='step', log=True, bins=100)
