import sys
import time
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Tuple, List, Set, Dict

import fast_hdbscan
import numpy as np
import open3d
import torch
import tqdm
from scipy.spatial import KDTree

import pointnetexternal.models.pointnet2_sem_seg
import regionGrowingOctree.RegionGrowingOctreeVisualization
from regionGrowingOctree import RegionGrowingOctree
from utilities.HDBSCANConfig import HDBSCANConfigAndResult
from utilities.noise_clustering import get_noise_clusters_k1, get_noise_clusters_kx
from utilities.pointnetv2_utilities import convert_to_batches

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

    assert points.ndim == 2
    assert points.shape[1] == 9

    if verbose:
        print(f"Clustering / segmenting using HDBScan (min cluster size {config.min_cluster_size}, "
              f"min samples {config.min_samples}, method '{config.method}')")

    start_time = time.time()

    if config.include_normals and not config.include_colors:
        points = points[:, :6]
    elif not config.include_normals and config.include_colors:
        points = np.hstack((points[:, :3], points[:, 6:]))
    elif not config.include_normals and not config.include_colors:
        points = points[:, :3]

    sys.setrecursionlimit(15000)
    membership_strengths: np.ndarray = None

    use_sklearn = False

    if use_sklearn:
        model = fast_hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size,
                                     min_samples=config.min_samples,
                                     cluster_selection_method=config.min_samples,
                                     allow_single_cluster=False,
                                     cluster_selection_epsilon=0)

        cluster_per_point = model.fit_predict(points)

    else:
        results = fast_hdbscan.fast_hdbscan(points,
                                            min_samples=config.min_samples,
                                            min_cluster_size=config.min_cluster_size,
                                            cluster_selection_method=config.method,
                                            allow_single_cluster=False,
                                            cluster_selection_epsilon=0.0,
                                            return_trees=False)

        cluster_per_point, membership_strengths = results

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

    if cluster_per_point is not None and config.noise_nearest_neighbours > 0 and np.count_nonzero(cluster_per_point < 0) > 0:
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
                                initial_voxel_size: float = 0.1,
                                minimum_valid_segment_size: int = 10,
                                subdivision_residual_threshold: float = 0.001,
                                subdivision_full_threshold: int = 10,
                                subdivision_minimum_voxel_size: float = 0.01,
                                region_growing_residual_threshold: float = 0.05,
                                growing_normal_deviation_threshold_degrees: float = 15.0,
                                refining_normal_deviation_threshold_degrees: float = 15.0,
                                fast_refinement_planar_amount_threshold: float = 0.9,
                                fast_refinement_planar_distance_threshold: float = 0.01,
                                fast_refinement_distance_threshold: float = 0.02,
                                general_refinement_buffer_size: float = 0.02,
                                visualize: bool = True,
                                verbose: bool = True):
    """
    Perform octree-based region growing on a given point cloud.

    :param visualize: Whether to show the voxels using the standard Open3D visualizer
    :param general_refinement_buffer_size: The buffer around the boundary nodes of a region / segment \
        in which points will be considered during general refinement.
    :param fast_refinement_distance_threshold: The maximum distance a point can be to the plane (defined by the \
        normal and centroid) of a region for that point to be joined with the region.
    :param fast_refinement_planar_distance_threshold: The maximum distance of points to the plane (defined by the normal \
        and centroid of a target region) for them to be considered for checking eligibility of fast refinement for \
        said target region.
    :param fast_refinement_planar_amount_threshold: The minimum fraction of points within a region that must be within \
        'refining_planar_distance_threshold' from the plane defined by the centroid and normal of the region for \
        the region to be eligible for fast refinement.
    :param refining_normal_deviation_threshold_degrees: The maximum angular deviation between neighbouring points \
        for them to join the region during (general) refinement. Used on points, not segments or nodes!
    :param growing_normal_deviation_threshold_degrees: The maximum angular deviation between normals (in degrees) of \
        neighbouring octree nodes for them to be joined with a segment. Used during growing of regions over octree \
        leaf nodes.
    :param region_growing_residual_threshold: The quantile of the residuals of the octree leaf nodes that determines \
        the actual threshold. Octree leaf nodes with a residual above the resulting threshold will not be considered \
        as seeds (for segments) for region growing.
    :param initial_voxel_size: Octree-based region growing first voxelizes the input before creating the octree. \
        This parameter controls the size of these initial voxels.
    :param minimum_valid_segment_size: During region growing, the minimum amount of octree nodes that need to be \
        in a segment for the segment to be considered valid.
    :param subdivision_residual_threshold: During subdivision, i.e. octree creation, the minimum residual value \
        to consider the octree node for subdivision.
    :param subdivision_full_threshold: How many points must be in the octree node at a minimum to consider the node \
        for subdivision.
    :param subdivision_minimum_voxel_size: The minimum size of an octree node during subdivision. The size of all \
        octree nodes will thus be equal to larger than this.
    :return:
    """

    assert data.ndim == 2
    assert data.shape[1] == 9

    # open3d.visualization.draw_geometries([ds_pcd], point_show_normal=True)
    print("Creating octree for Octree-based-region-growing...")
    octree = RegionGrowingOctree.RegionGrowingOctree(data, root_margin=0.1)
    print("Performing initial voxelization...")

    octree.initial_voxelization(initial_voxel_size)

    octree.recursive_subdivide(minimum_voxel_size=subdivision_minimum_voxel_size,
                               residual_threshold=subdivision_residual_threshold,
                               full_threshold=subdivision_full_threshold,
                               max_depth=9,  # A higher max depth is not recommended
                               profile=False, verbose=verbose)

    print("Growing regions...")
    octree.grow_regions(minimum_valid_segment_size=minimum_valid_segment_size,
                        residual_threshold=region_growing_residual_threshold,
                        normal_deviation_threshold_degrees=growing_normal_deviation_threshold_degrees,
                        residual_threshold_is_absolute=False,
                        profile=False, verbose=verbose)

    print("Refining...")
    octree.refine_regions(planar_amount_threshold=fast_refinement_planar_amount_threshold,
                          planar_distance_threshold=fast_refinement_planar_distance_threshold,
                          fast_refinement_distance_threshold=fast_refinement_distance_threshold,
                          buffer_zone_size=general_refinement_buffer_size,
                          angular_divergence_threshold_degrees=refining_normal_deviation_threshold_degrees,
                          verbose=verbose)

    octree.finalize()
    noise_cluster_indices = assign_noise_nearest_neighbour_cluster(data, octree.segment_index_per_point, 3)
    octree.segment_index_per_point[octree.segment_index_per_point < 0] = noise_cluster_indices

    if visualize:
        print("Visualizing voxels...")
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_as_points(octree, True)
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_with_points(octree)
    return octree


def pointnetv2(model_checkpoint_path: str,
               pcd: open3d.geometry.PointCloud,
               working_directory: Union[str, PathLike],
               visualize_raw_classifications: bool = True,
               create_segmentations: bool = True,
               segmentation_max_distance: float = 0.02) -> Tuple[np.ndarray, Optional[Dict[int, List[Set]]]]:

    """
    Execute a classification (and possible segmentation) using a trained PointNet++ model.

    :param model_checkpoint_path: The path to the model checkpoint to load as the PointNet++ model.
    :param pcd: The point cloud to classify.
    :param working_directory: The working directory to store the classification results (and possible visualization)
    :param visualize_raw_classifications: Whether to draw the raw classification results using Open3D
    :param create_segmentations: Whether to segment the classified pointcloud using region growing.
    :param segmentation_max_distance: The maximum distance during region growing, used during segmentation.
    :return: The classifications per point in the point cloud and optionally the clusters (i.e. sets of indices \
        of points) per label/class.
    """

    number_of_classes = len(CLASSES)  # PointNet++ is trained on the S3DIS dataset, which has 13 classes.
    channels = 9
    classifier: torch.nn.Module = pointnetexternal.models.pointnet2_sem_seg.get_model(number_of_classes,
                                                                                      channels).cuda()
    # Load the checkpoint (i.e. the saved model)
    checkpoint = torch.load(model_checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Set the model into evaluation mode
    classifier = classifier.eval()

    # Prepare the input
    points: np.ndarray = np.asarray(pcd.points)
    # colors: np.ndarray = np.asarray(pcd.colors)
    # data = np.hstack((points, colors))

    npoint = 4096
    batches, batches_indices = convert_to_batches(points=points, colors=None, normals=None,
                                                  point_amount=npoint, block_size=1, stride=0.25)
    print(f"Created {len(batches)} batches.")

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

    with torch.no_grad():
        print(f"Classifying {len(batches)} batches with {number_of_votes} votes...")
        for batch_index in tqdm.trange(len(batches), desc=f"Classifying...", miniters=1):
            batch = batches[batch_index]
            batch = np.swapaxes(batch, 2, 1)
            # batch = np.reshape(batch, newshape=(1, batch.shape[0], batch.shape[1]))
            indices = batches_indices[batch_index]
            model_input: torch.Tensor = torch.from_numpy(batch).float().cuda()
            for vote in range(number_of_votes):
                # Actually do the prediction!
                predictions, _ = classifier(model_input)
                class_per_point = predictions.cpu().numpy().argmax(axis=2).squeeze().astype(np.int32)

                for i in range(indices.shape[0]):
                    for j in range(indices.shape[1]):
                        votes[indices[i, j], class_per_point[i, j]] += 1

    classifications = votes.argmax(axis=1)
    for i in range(number_of_classes):
        print(
            f"Class {CLASSES[i]} (color {CLASS_COLORS[i][0]} : {CLASS_COLORS[i][1]}) "
            f"occurred {np.count_nonzero(classifications == i)} times.")

    working_directory_path = Path(working_directory)

    classification_save_path = working_directory_path.joinpath("classifications.npy")
    np.save(classification_save_path, classifications)

    numpy_class_colors = np.array([c[1] for c in CLASS_COLORS])
    if visualize_raw_classifications:
        colors_per_point: np.ndarray = numpy_class_colors[classifications]
        visualize_pcd = open3d.geometry.PointCloud(pcd.points)
        visualize_pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
        open3d.visualization.draw_geometries([visualize_pcd])
        pcd_colored_to_classes_path = working_directory_path.joinpath("classifications.ply")
        open3d.io.write_point_cloud(str(pcd_colored_to_classes_path), visualize_pcd)

    clusters_per_class = None
    if create_segmentations:
        clusters_per_class = extract_clusters_from_labelled_points(points=points[:, :3],  # Sanity check!
                                                                   labels_per_point=classifications,
                                                                   max_distance=segmentation_max_distance)

    return classifications, clusters_per_class


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
                                          max_distance: float = 0.02) -> Dict[int, List[Set]]:
    points_indices_per_class = [np.nonzero(labels_per_point == i)[0] for i in range(len(CLASSES))]
    clusters_per_class: dict = {}

    for c in range(len(CLASSES)):
        relevant_points = points_indices_per_class[c]
        if len(relevant_points) > 0:
            clusters = _extract_clusters_region_growing(points[relevant_points], max_distance)
            clusters_per_class[c] = clusters

    return clusters_per_class


def _extract_clusters_region_growing(points: np.ndarray, max_distance: float = 0.02) -> List[Set]:
    all_clusters = []

    unassigned_points_indices_set = set(range(len(points)))
    kd_tree = KDTree(points)

    while len(unassigned_points_indices_set) > 0:
        initial_seed = unassigned_points_indices_set.pop()
        indices_stack = [initial_seed]
        current_cluster = set()
        current_cluster.add(initial_seed)
        while len(indices_stack) > 0:
            current_seed_index = indices_stack.pop()
            current_seed_point = points[current_seed_index]

            # Get all neighbours of this current point
            _, neighbour_indices = kd_tree.query_ball_point(current_seed_point, max_distance, workers=-1)

            # Get all non-assigned neighbours. If there are none, just continue.
            neighbour_indices = set(neighbour_indices).union(unassigned_points_indices_set)
            if len(neighbour_indices) == 0:
                continue

            # Add all the neighbours to the indices stack to check later
            indices_stack.extend(neighbour_indices)

            # Mark the current neighbour indices as assigned.
            current_cluster = current_cluster.union(neighbour_indices)
            unassigned_points_indices_set = unassigned_points_indices_set.difference(neighbour_indices)

        # Add the cluster
        all_clusters.append(current_cluster)

    return all_clusters
