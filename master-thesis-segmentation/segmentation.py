import sys
import time
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Tuple, List, Set, Dict
import multiprocessing

import fast_hdbscan
import numpy as np
import open3d
import torch
import tqdm
from scipy.spatial import KDTree

import pointnetexternal.models.pointnet2_sem_seg
import regionGrowingOctree.RegionGrowingOctreeVisualization
import utilities.pointv2_dataset
from regionGrowingOctree import RegionGrowingOctree
from utilities.HDBSCANConfig import HDBSCANConfigAndResult
from utilities.noise_clustering import get_noise_clusters_k1, get_noise_clusters_kx
from utilities.pointnetv2_utilities import convert_to_batches
from utilities.OctreeBasedRegionGrowingConfig import OctreeBasedRegionGrowingConfig
from torch.utils.data import DataLoader

CLASSES = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']

CLASS_COLORS = [("gray", np.array([0.5, 0.5, 0.5])),
                ("black", np.array([0, 0, 0])),
                ("red", np.array([1.0, 0.0, 0.0])),
                ("lime", np.array([0.0, 1.0, 0.0])),
                ("blue", np.array([0.0, 0.0, 1.0])),
                ("yellow", np.array([1.0, 1.0, 0.0])),
                ("olive", np.array([0.5, 0.5, 0.0])),
                ("green", np.array([0.0, 0.5, 0.0])),
                ("aqua", np.array([0.0, 1.0, 1.0])),
                ("teal", np.array([0.0, 0.5, 0.5])),
                ("fuchsia", np.array([1.0, 0.0, 1.0])),
                ("purple", np.array([0.5, 0.0, 0.5])),
                ("navy", np.array([0.0, 0.0, 0.5]))]


def hdbscan(points: np.ndarray,
            config: HDBSCANConfigAndResult,
            verbose: bool = False):
    """
    Parameters
    :param config: The configuration file.
    :returns: The segment index per point and their memberships strengths, and the noise indices.
    """

    # assert points.ndim == 2
    # assert points.shape[1] == 9

    if verbose:
        print(f"Clustering / segmenting using HDBScan (min cluster size {config.min_cluster_size}, "
              f"min samples {config.min_samples}, method '{config.method}')")

    start_time = time.time()

    if config.include_normals and not config.include_colors:
        if points.shape[1] <= 9:
            points = points[:, :6]
        else:
            points = np.hstack((points[:, :6], points[:, 9:]))
    elif not config.include_normals and config.include_colors:
        points = np.hstack((points[:, :3], points[:, 6:]))
    elif not config.include_normals and not config.include_colors:
        if points.shape[1] <= 9:
            points = points[:, :3]
        else:
            points = np.hstack((points[:, :3], points[:, 9:]))

    sys.setrecursionlimit(15000)
    membership_strengths: np.ndarray = None

    use_sklearn = False

    if use_sklearn:
        model = fast_hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size,
                                     min_samples=config.min_samples,
                                     cluster_selection_method=config.method,
                                     allow_single_cluster=False,
                                     cluster_selection_epsilon=0)

        cluster_per_point = model.fit_predict(points)

    else:
        try:
            results = fast_hdbscan.fast_hdbscan(points,
                                                min_samples=config.min_samples,
                                                min_cluster_size=config.min_cluster_size,
                                                cluster_selection_method=config.method,
                                                allow_single_cluster=False,
                                                cluster_selection_epsilon=0.0,
                                                return_trees=False)

            cluster_per_point, membership_strengths = results
        except Exception as e:
            print(f"HDBSCAN failed because of error. {e}")
            cluster_per_point = np.array([])

    number_of_clusters = len(np.unique(cluster_per_point))

    # if pcd.has_colors():
    #    X = np.hstack((X, np.asarray(pcd.colors)), dtype=np.float32)

    end_time = time.time()
    if verbose:
        print("Clustering done.")
        print(f"Created {number_of_clusters} clusters in {round(end_time - start_time, 4)} seconds.")

    if number_of_clusters == 0:
        return

    cluster_sizes = []
    for i in range(number_of_clusters):
        cluster = np.count_nonzero(cluster_per_point == i)
        if cluster > 0:
            cluster_sizes.append(cluster)

    if verbose:
        print(
            f"Cluster sizes (min {config.min_cluster_size}): smallest {min(cluster_sizes)} largest {max(cluster_sizes)} "
            f"mean {np.mean(cluster_sizes)} std {np.std(cluster_sizes)} median {np.median(cluster_sizes)}")

        print(f"Noise / non-clustered points: {np.count_nonzero(cluster_per_point < 0)}")

    config.noise_indices = np.nonzero(cluster_per_point < 0)[0]

    if cluster_per_point is not None and config.noise_nearest_neighbours > 0 and np.count_nonzero(
            cluster_per_point < 0) > 0:
        new_clusters_for_noise = assign_noise_nearest_neighbour_cluster(points, cluster_per_point,
                                                                        config.noise_nearest_neighbours)
        cluster_per_point[cluster_per_point < 0] = new_clusters_for_noise
        if verbose:
            print(f"Assigned non-clustered noise points. Noise remaining {np.count_nonzero(cluster_per_point < 0)}")

    config.clusters = cluster_per_point
    config.membership_strengths = membership_strengths
    config.clustering_time = round(end_time - start_time, 6)
    config.total_points = points.shape[0]

    if config.visualize and cluster_per_point is not None:
        unique_clusters = np.unique(cluster_per_point)
        rng = np.random.default_rng()
        colors_per_cluster = rng.random((len(unique_clusters), 3), dtype=np.float64)
        colors = colors_per_cluster[cluster_per_point]
        colors[cluster_per_point < 0] = np.zeros(shape=(3,))
        # if membership_strengths is not None:
        #    colors[:, 0] *= membership_strengths
        #    colors[:, 1] *= membership_strengths
        #    colors[:, 2] *= membership_strengths

        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd])


def octree_based_region_growing(data: np.ndarray,
                                config: OctreeBasedRegionGrowingConfig,
                                visualize: bool = True,
                                verbose: bool = True) -> RegionGrowingOctree.RegionGrowingOctree:
    """
    Perform octree-based region growing on a given point cloud.

    :param config: The configuration with all the parameters.
    :param data: The data to actually use during region growing (i.e. the points, colors and normals)
    :param verbose: Whether to print debug statements and progress bars.
    :param visualize: Whether to show the voxels using the standard Open3D visualizer

    :return: The resulting octree
    """

    assert data.ndim == 2
    assert data.shape[1] == 9

    if verbose:
        print("Creating octree for Octree-based-region-growing...")
    octree = RegionGrowingOctree.RegionGrowingOctree(data, root_margin=0.1)

    octree.initial_voxelization(config.initial_voxel_size)

    octree.recursive_subdivide(minimum_voxel_size=config.subdivision_minimum_voxel_size,
                               residual_threshold=config.subdivision_residual_threshold,
                               full_threshold=config.subdivision_full_threshold,
                               max_depth=9,  # A higher max depth is not recommended
                               profile=False, verbose=verbose)

    octree.grow_regions(minimum_valid_segment_size=config.minimum_valid_segment_size,
                        residual_threshold=config.region_growing_residual_threshold,
                        normal_deviation_threshold_degrees=config.growing_normal_deviation_threshold_degrees,
                        residual_threshold_is_absolute=False,
                        profile=False, verbose=verbose)

    octree.refine_regions(planar_amount_threshold=config.fast_refinement_planar_amount_threshold,
                          planar_distance_threshold=config.fast_refinement_planar_distance_threshold,
                          fast_refinement_distance_threshold=config.fast_refinement_distance_threshold,
                          buffer_zone_size=config.general_refinement_buffer_size,
                          angular_divergence_threshold_degrees=config.refining_normal_deviation_threshold_degrees,
                          verbose=verbose)

    octree.finalize()
    noise_cluster_indices = assign_noise_nearest_neighbour_cluster(data, octree.segment_index_per_point, 3)
    config.noise_points = len(noise_cluster_indices)
    octree.segment_index_per_point[octree.segment_index_per_point < 0] = noise_cluster_indices

    if visualize:
        print("Visualizing voxels...")
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_as_points(octree, True)
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_with_points(octree)
    return octree


def pointnetv2(model_checkpoint_path: str,
               points: np.ndarray,
               colors: Optional[np.ndarray],
               normals: Optional[np.ndarray],
               working_directory: Union[str, PathLike],
               visualize_raw_classifications: bool = True,
               create_segmentations: bool = True,
               segmentation_max_distance: float = 0.02) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Execute a classification (and possible segmentation) using a trained PointNet++ model.

    :param model_checkpoint_path: The path to the model checkpoint to load as the PointNet++ model.
    :param working_directory: The working directory to store the classification results (and possible visualization)
    :param visualize_raw_classifications: Whether to draw the raw classification results using Open3D
    :param create_segmentations: Whether to segment the classified pointcloud using region growing.
    :param segmentation_max_distance: The maximum distance during region growing, used during segmentation.
    :return: The classifications per point in the point cloud and optionally the clusters (i.e. sets of indices \
        of points) per label/class.
    """

    # Prepare the input
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert colors is None or colors.shape == points.shape
    assert normals is None or normals.shape == points.shape

    start_time = time.time()

    npoint = 4096

    dataset = utilities.pointv2_dataset.PointNetV2_CustomDataset(points, colors, None, npoint, 1.0, 0.5)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    #batches, batches_indices = convert_to_batches(points=points, colors=colors, normals=normals,
    #                                              point_amount=npoint, block_size=1, stride=0.5)

    batching_time = time.time() - start_time
    start_time = time.time()

    number_of_classes = len(CLASSES)  # PointNet++ is trained on the S3DIS dataset, which has 13 classes.
    channels = 9 + (0 if colors is None else 3) + (0 if normals is None else 3)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    classifier: torch.nn.Module = pointnetexternal.models.pointnet2_sem_seg.get_model(number_of_classes,
                                                                                      channels).to(device=device)
    # Load the checkpoint (i.e. the saved model)
    checkpoint = torch.load(model_checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Set the model into evaluation mode
    classifier = classifier.eval()

    # print(f"Created {len(batches)} batches.")

    # model expects input of size B x 9 x points where B is probably batches
    # index 0,1,2 are the xyz normalized by subtracting the centroid
    # index 3,4,5 are the normalized RGB values
    # index 6,7,8 are the xyz normalized by dividing by the max coordinate
    # centroid = points.mean(axis=0)
    # data[0, :3, :] = np.swapaxes(points - centroid, 0, 1)
    # data[0, 3:6, :] = np.swapaxes(colors / 255.0, 0, 1)
    # data[0, 6:9, :] = np.swapaxes(points / points.max(axis=0), 0, 1)
    # number_of_batches = math.ceil(len(points) / batch_size)
    # batches = np.array_split(data, number_of_batches, axis=2)
    number_of_votes: int = 1
    votes = np.zeros(shape=(len(points), number_of_classes), dtype=np.int32)

    for batch in tqdm.tqdm(dataloader, desc=f"Classifying batches... (votes {number_of_votes})"):
        data: torch.Tensor = torch.permute(batch[:, :, :9], (0, 2, 1)).float().cuda()
        indices: np.ndarray = batch[:, :, -1].cpu().numpy().astype(np.int32)
        for vote in range(number_of_votes):
            # Actually do the prediction!
            with torch.no_grad():
                predictions, _ = classifier(data)
            class_per_point = predictions.cpu().numpy().argmax(axis=2).squeeze().astype(np.int32)

            if indices.shape != class_per_point.shape:
                class_per_point = np.reshape(class_per_point, indices.shape)

            for i in range(indices.shape[0]):
                votes[indices[i, :], class_per_point[i, :]] += 1

    classifications: np.ndarray = votes.argmax(axis=1)

    classification_time = time.time() - start_time

    for i in range(number_of_classes):
        print(
            f"Class {CLASSES[i]} (color {CLASS_COLORS[i][0]} : {CLASS_COLORS[i][1]}) "
            f"occurred {np.count_nonzero(classifications == i)} times.")

    working_directory_path = Path(working_directory)

    current_time = str(time.time())
    classification_save_path = working_directory_path.joinpath(f"classifications-{current_time}.npy")
    np.save(classification_save_path, classifications)

    numpy_class_colors = np.array([c[1] for c in CLASS_COLORS])
    if visualize_raw_classifications:
        colors_per_point: np.ndarray = numpy_class_colors[classifications]
        visualize_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        visualize_pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
        # open3d.visualization.draw_geometries([visualize_pcd])
        pcd_colored_to_classes_path = working_directory_path.joinpath(f"classifications-{current_time}.ply")
        open3d.io.write_point_cloud(str(pcd_colored_to_classes_path), visualize_pcd)

    start_time = time.time()

    cluster_index_per_point = None
    if create_segmentations:
        print("Extracting clusters...")
        _, cluster_index_per_point = extract_clusters_from_labelled_points_multiprocess(points=points[:, :3],  # Sanity check!
                                                                                  labels_per_point=classifications,
                                                                                  max_distance=segmentation_max_distance)
        clusters_save_path = working_directory_path.joinpath(f"clusters-{current_time}.npy")
        np.save(clusters_save_path, cluster_index_per_point)

    clustering_time = time.time() - start_time

    times_path = working_directory_path.joinpath(f"times-{current_time}.txt")

    rng = np.random.default_rng()
    cluster_colors = rng.random(size=(len(cluster_index_per_point), 3))

    if visualize_raw_classifications:
        colors_per_point: np.ndarray = cluster_colors[cluster_index_per_point]
        visualize_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        visualize_pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
        # open3d.visualization.draw_geometries([visualize_pcd])
        pcd_colored_to_clusters = working_directory_path.joinpath(f"clusters-{current_time}.ply")
        open3d.io.write_point_cloud(str(pcd_colored_to_clusters), visualize_pcd)

    with open(times_path, "w") as f:
        f.write(f"batching time: {batching_time}\n")
        f.write(f"classification time: {classification_time}\n")
        f.write(f"clustering time: {clustering_time}\n")

    return classifications, cluster_index_per_point


def assign_noise_nearest_neighbour_cluster(points: np.ndarray,
                                           cluster_per_point: np.ndarray,
                                           neighbour_count: int) -> np.ndarray:
    noise_point_indices = np.nonzero(cluster_per_point < 0)[0]
    data_points_indices = np.nonzero(cluster_per_point >= 0)[0]

    assert cluster_per_point.ndim == 1
    assert points.ndim == 2

    if points.shape[1] == 3:
        data_points = points[data_points_indices]
        noise_points = points[noise_point_indices]
    elif points.shape[1] > 3:
        data_points = points[data_points_indices, :3]
        noise_points = points[noise_point_indices, :3]
    else:
        raise ValueError("Points numpy array be of shape (n_points, 3)")

    # Create a KDTree to quickly find nearest neighbours
    kd_tree = KDTree(data_points)

    if neighbour_count == 1:
        return get_noise_clusters_k1(cluster_per_point, data_points_indices, kd_tree, noise_points)
    else:
        return get_noise_clusters_kx(cluster_per_point, data_points_indices, kd_tree, neighbour_count,
                                     noise_point_indices, noise_points)


def extract_clusters_from_labelled_points(points: np.ndarray,
                                          labels_per_point: np.ndarray,
                                          max_distance: float = 0.02) -> np.ndarray:
    points_indices_per_class = [np.nonzero(labels_per_point == i)[0] for i in range(len(CLASSES))]
    cluster_indices_per_point = np.full(shape=(len(points),), fill_value=-1, dtype=np.int32)

    kd_tree = KDTree(points)
    cluster_index = 0

    for c in tqdm.trange(len(CLASSES), desc="Extracting clusters for classes..."):
        relevant_point_indices = points_indices_per_class[c]

        # there are no points in this class
        if len(relevant_point_indices) == 0:
            continue

        unassigned_point_indices = set(relevant_point_indices)
        while len(unassigned_point_indices) > 0:
            initial_seed_index = unassigned_point_indices.pop()
            frontier = [initial_seed_index]  # Basically the indices we still need to check
            current_cluster = set()
            current_cluster.add(initial_seed_index)
            while len(frontier) > 0:
                current_seed_index = frontier.pop()
                current_seed_coor = points[current_seed_index]
                neighbours_indices = kd_tree.query_ball_point(current_seed_coor, max_distance, workers=-1)
                # Filter the neighbour indices that are unassigned AND have the right class.
                neighbours_indices = set(neighbours_indices).union(unassigned_point_indices)
                if len(neighbours_indices) <= 0:
                    continue
                frontier.extend(neighbours_indices)  # Add the current neighbours to the "to check" list.
                current_cluster = current_cluster.union(neighbours_indices)
                unassigned_point_indices = unassigned_point_indices.difference(neighbours_indices)

            cluster_indices_per_point[np.array(current_cluster)] = cluster_index
            cluster_index += 1

    return cluster_indices_per_point


def extract_clusters_from_labelled_points_multicore(points: np.ndarray,
                                                    labels_per_point: np.ndarray,
                                                    max_distance: float = 0.02) -> Tuple[List[List[int]], np.ndarray]:
    print(
        f"Clustering {len(points)} points based on labels/classifications using region growing with radius {max_distance}.")
    points_indices_per_class = [np.nonzero(labels_per_point == i)[0] for i in range(len(CLASSES))]
    cluster_indices_per_point = np.full(shape=(len(points),), fill_value=-1, dtype=np.int32)
    clusters = []
    cluster_index = 0

    # leaf_size = max(int(len(points) / 100_000), 10)

    pbar = tqdm.tqdm(total=len(points), desc="Clustering points...", miniters=1, unit="points", smoothing=0.01)

    for c in range(len(CLASSES)):
        relevant_point_indices = points_indices_per_class[c]

        # there are no points in this class
        if len(relevant_point_indices) == 0:
            continue

        kd_tree = KDTree(points[relevant_point_indices], leafsize=10)
        unassigned_point_indices = relevant_point_indices.copy()

        while len(unassigned_point_indices) > 0:
            initial_seed_index = unassigned_point_indices[-1]
            unassigned_point_indices = unassigned_point_indices[:-1]
            pbar.update(1)

            frontier = np.array([points[initial_seed_index]])
            current_cluster = np.array([initial_seed_index])

            while len(frontier) > 0:
                # neighbours_indices = kd_tree.query_ball_tree(frontier, max_distance)
                neighbours_indices = kd_tree.query_ball_point(frontier, max_distance, workers=-1)
                neighbours_indices = np.concatenate(
                    [np.array(i) for i in neighbours_indices])  # https://stackoverflow.com/a/42499122
                neighbours_indices = np.unique(neighbours_indices)

                # Filter the neighbour indices that are unassigned
                global_neighbour_idx = relevant_point_indices[neighbours_indices]
                global_neighbour_idx = np.intersect1d(global_neighbour_idx, unassigned_point_indices)

                # global_neighbour_indices_set = set(global_neighbour_indices_arr).union(unassigned_point_indices)
                # global_neighbour_indices_arr = np.asarray(global_neighbour_indices_set)

                if len(global_neighbour_idx) <= 0:
                    break

                frontier = points[global_neighbour_idx]
                current_cluster = np.union1d(current_cluster, global_neighbour_idx)
                unassigned_point_indices = np.setdiff1d(unassigned_point_indices, global_neighbour_idx)
                pbar.update(len(global_neighbour_idx))

            cluster_indices_per_point[current_cluster] = cluster_index
            clusters.append(list(current_cluster))
            cluster_index += 1
            pbar.set_description(f"Clustering points... (Found {len(clusters)} clusters, done {c} classes)")

    return clusters, cluster_indices_per_point


def extract_clusters_from_labelled_points_multiprocess(points: np.ndarray,
                                                       labels_per_point: np.ndarray,
                                                       max_distance: float = 0.02) -> Tuple[List[List[int]], np.ndarray]:

    points_indices_per_class = [np.nonzero(labels_per_point == i)[0] for i in range(len(CLASSES))]
    cluster_indices_per_point_global = np.full(shape=(len(points),), fill_value=-1, dtype=np.int32)

    with multiprocessing.Pool(13) as pool:
        arguments = []
        for i in range(13):
            if np.count_nonzero(labels_per_point == i) <= 0:
                arguments.append((i, None, None, max_distance))
            else:
                indices = points_indices_per_class[i]
                relevant_points = points[indices]
                arguments.append((i, relevant_points, indices, max_distance))

        cluster_index = 0
        for result in pool.map(extract_clusters_single_process, arguments):
            cluster_indices_per_point, indices = result

            if cluster_indices_per_point is not None and len(cluster_indices_per_point) > 0:
                cluster_indices_per_point_global[indices] = cluster_indices_per_point + cluster_index
                cluster_index += np.max(cluster_indices_per_point)

    print(str(type(cluster_indices_per_point_global)))
    print(cluster_indices_per_point_global.shape)
    return None, cluster_indices_per_point_global

def extract_clusters_single_process(args):
    class_index, points, indices, max_distance = args

    if points is None or len(points) == 0:
        print(f"Class {CLASSES[class_index]} is done (had no points).")
        return None, None

    cluster_index = 0
    # clusters = []
    kd_tree = KDTree(points, leafsize=10)
    unassigned_point_indices = np.arange(len(points), dtype=np.int32)
    cluster_indices_per_point = np.full(shape=(len(points),), fill_value=-1, dtype=np.int32)

    while len(unassigned_point_indices) > 0:
        initial_seed_index = unassigned_point_indices[-1]
        unassigned_point_indices = unassigned_point_indices[:-1]
        frontier = np.array([points[initial_seed_index]])
        current_cluster = np.array([initial_seed_index])

        while len(frontier) > 0:
            # neighbours_indices = kd_tree.query_ball_tree(frontier, max_distance)
            neighbours_indices = kd_tree.query_ball_point(frontier, max_distance, workers=-1)
            neighbours_indices = np.concatenate(
                [np.array(i) for i in neighbours_indices])  # https://stackoverflow.com/a/42499122
            neighbours_indices = np.unique(neighbours_indices)

            # Filter the neighbour indices that are unassigned
            global_neighbour_idx = np.intersect1d(neighbours_indices, unassigned_point_indices)

            if len(global_neighbour_idx) <= 0:
                break

            frontier = points[global_neighbour_idx]
            current_cluster = np.union1d(current_cluster, global_neighbour_idx)
            unassigned_point_indices = np.setdiff1d(unassigned_point_indices, global_neighbour_idx)

        cluster_indices_per_point[current_cluster] = cluster_index
        # clusters.append(list(current_cluster))
        cluster_index += 1

    print(f"Class {CLASSES[class_index]} is done with {class_index + 1} clusters.")
    return cluster_indices_per_point, indices