import math

import numpy as np
import tqdm


def get_noise_clusters_k1(cluster_per_point, data_points_indices, kd_tree, noise_points):
    _, nearest_neighbour_indices = kd_tree.query(noise_points, k=1, workers=-1)
    new_clusters = cluster_per_point[data_points_indices[nearest_neighbour_indices]]
    return new_clusters


def _get_noise_clusters_kx_batches(cluster_per_point, data_points_indices, nearest_neighbour_indices,
                                   noise_point_indices, noise_points):
    new_clusters = np.full(shape=(noise_points.shape[0]), fill_value=-1)
    batch_size = 100
    number_of_batches = math.ceil(len(noise_points) / batch_size)
    for i in tqdm.trange(number_of_batches, desc="Assigning noise points to cluster (batches)"):
        # batch = noise_points[i * batch_size: (i + 1) * batch_size]
        batch_indices = noise_point_indices[i * batch_size:(i + 1) * batch_size]
        nearest_neighbours = nearest_neighbour_indices[batch_indices]
        nearest_neighbour_clusters = cluster_per_point[data_points_indices[nearest_neighbours]]
        bins = np.apply_along_axis(np.bincount, axis=1, arr=nearest_neighbour_clusters)
        new_clusters[batch_indices] = np.argmax(bins, axis=1)
    return new_clusters


def get_noise_clusters_kx_unsafe(neighbouring_clusters):
    u, indices = np.unique(neighbouring_clusters, return_inverse=True)
    axis = 1
    # noinspection PyTypeChecker
    x = np.apply_along_axis(np.bincount, axis, indices.reshape(neighbouring_clusters.shape), None,
                            np.max(indices) + 1)
    arg_max = np.argmax(x, axis=axis)
    return arg_max, u


def get_noise_clusters_kx(cluster_per_point, data_points_indices, kd_tree, neighbour_count, noise_point_indices,
                          noise_points):
    _, nearest_neighbour_indices = kd_tree.query(noise_points, k=neighbour_count, workers=-1)
    neighbouring_clusters = cluster_per_point[data_points_indices[nearest_neighbour_indices]]

    try:
        arg_max, u = get_noise_clusters_kx_unsafe(neighbouring_clusters)
        return u[arg_max]
    except Exception as e:
        print(f"Got exception, assigning noise using k=1: {e}")
        return get_noise_clusters_k1(cluster_per_point, data_points_indices, kd_tree, noise_points)
