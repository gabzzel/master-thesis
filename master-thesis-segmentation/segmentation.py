import math
import sys
import time
from typing import Optional, Union, Tuple

import fast_hdbscan
import numpy as np
import open3d
import scipy.spatial.distance
import torch
import tqdm
from scipy.spatial import KDTree

import pointnetexternal.models.pointnet2_sem_seg
import regionGrowingOctree.RegionGrowingOctreeVisualization
from regionGrowingOctree import RegionGrowingOctree


def hdbscan(pcd: open3d.geometry.PointCloud,
            use_normals: bool = False,
            use_colors: bool = False,
            minimum_cluster_size: Union[int, str] = 10,
            minimum_samples: Optional[Union[int, str]] = None,
            cluster_selection_epsilon: Union[float, str] = 0.0,
            method: str = "eom",
            visualize: bool = True,
            use_sklearn_estimator: bool = False,
            assign_noise_to_nearest_neighbour: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    :param assign_noise_to_nearest_neighbour: Whether to assign the cluster / segment of the nearest neighbour for
        each unassigned / unsegmented / noise point.
    :param use_colors: Whether to include the color data (if available) in the clustering
    :param use_normals: Whether to include the normal data (if available) in the clustering
    :param use_sklearn_estimator: Whether to use sklearn HDBSCAN implementation. If false, uses the fast_hdbscan \
        implementation.
    :param method: Which method to use for determining best cluster set. Default is excess of mass.
    :param visualize: Whether to draw the segments to the screen using Open3D visualization.
    :param cluster_selection_epsilon: A distance threshold. Clusters below this value will be merged. \
        I probably need to keep this to 0 to keep to the original HDBSCAN method.
    :param minimum_samples: The number of samples in a neighborhood for a point to be considered as a core point. \
        This includes the point itself. When None, defaults to min_cluster_size.
    :param pcd: The point cloud to segment.
    :param minimum_cluster_size: The minimum number of samples in a group for that group to be considered a cluster; \
        groupings smaller than this size will be left as noise.
    :returns: The segment index per point and their memberships strengths.
    """

    print(f"Clustering / segmenting using HDBScan (min cluster size {minimum_cluster_size}, "
          f"min samples {minimum_samples}, method '{method}')")

    start_time = time.time()
    points = np.asarray(pcd.points)
    sys.setrecursionlimit(15000)

    if use_normals and pcd.has_normals():
        normals = np.asarray(pcd.normals)
        points = np.hstack((points, normals))

    if use_colors and pcd.has_colors():
        colors = np.asarray(pcd.colors)
        points = np.hstack((points, colors))

    cluster_per_point = None
    membership_strengths = None

    if use_sklearn_estimator:
        model = fast_hdbscan.HDBSCAN(min_cluster_size=minimum_cluster_size,
                                     min_samples=minimum_samples,
                                     cluster_selection_method=method,
                                     allow_single_cluster=False,
                                     cluster_selection_epsilon=cluster_selection_epsilon)

        cluster_per_point = model.fit_predict(points)

    else:
        results = fast_hdbscan.fast_hdbscan(points,
                                            min_samples=minimum_samples,
                                            min_cluster_size=minimum_cluster_size,
                                            cluster_selection_method=method,
                                            allow_single_cluster=False,
                                            cluster_selection_epsilon=cluster_selection_epsilon,
                                            return_trees=False)

        cluster_per_point, membership_strengths = results

    number_of_clusters = len(np.unique(cluster_per_point))

    # if pcd.has_colors():
    #    X = np.hstack((X, np.asarray(pcd.colors)), dtype=np.float32)

    end_time = time.time()
    print("Clustering done.")
    print(f"Created {number_of_clusters} clusters in {round(end_time - start_time, 4)} seconds.")

    cluster_sizes = []
    for i in range(number_of_clusters):
        cluster = np.count_nonzero(cluster_per_point == i)
        if cluster > 0:
            cluster_sizes.append(cluster)

    print(f"Cluster sizes (min {minimum_cluster_size}): smallest {min(cluster_sizes)} largest {max(cluster_sizes)} "
          f"mean {np.mean(cluster_sizes)} std {np.std(cluster_sizes)} median {np.median(cluster_sizes)}")

    print(f"Noise / non-clustered points: {np.count_nonzero(cluster_per_point < 0)}")

    if cluster_per_point is not None and assign_noise_to_nearest_neighbour:
        new_clusters_for_noise = assign_noise_nearest_neighbour_cluster(points, cluster_per_point)
        cluster_per_point[cluster_per_point < 0] = new_clusters_for_noise
        print(f"Assigned non-clustered noise points. Noise remaining {np.count_nonzero(cluster_per_point < 0)}")

    if visualize and cluster_per_point is not None:
        unique_clusters = np.unique(cluster_per_point)
        rng = np.random.default_rng()
        colors_per_cluster = rng.random((len(unique_clusters), 3), dtype=np.float64)
        colors = colors_per_cluster[cluster_per_point]
        colors[cluster_per_point < 0] = np.zeros(shape=(3,))
        # if membership_strengths is not None:
        #    colors[:, 0] *= membership_strengths
        #    colors[:, 1] *= membership_strengths
        #    colors[:, 2] *= membership_strengths

        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd])

    return cluster_per_point, membership_strengths


def octree_based_region_growing(pcd: open3d.geometry.PointCloud,
                                initial_voxel_size: float = 0.1,
                                down_sample_voxel_size: Optional[float] = 0.01,
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
                                visualize: bool = True):
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
    :param pcd: The point cloud to do the region growing on.
    :param initial_voxel_size: Octree-based region growing first voxelizes the input before creating the octree. \
        This parameter controls the size of these initial voxels.
    :param down_sample_voxel_size: The voxel size to use during voxel downsampling of the input point cloud. \
        Set to 0 or None to not downsample the input point cloud.
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

    original_number_of_points = len(pcd.points)
    r = 0.01 * math.sqrt(12.0)

    if down_sample_voxel_size is not None and down_sample_voxel_size > 0.0:
        r = down_sample_voxel_size * math.sqrt(12)
        ds_pcd = pcd.voxel_down_sample(voxel_size=down_sample_voxel_size)
        print(f"Downsampled point cloud from {original_number_of_points} to {len(ds_pcd.points)} points")
    else:
        ds_pcd = pcd

    if not ds_pcd.has_normals():
        print("Computing normals...")
        ds_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=10))
        print("Orienting normals...")
        ds_pcd.orient_normals_consistent_tangent_plane(k=10)
        ds_pcd.normalize_normals()

    # open3d.visualization.draw_geometries([ds_pcd], point_show_normal=True)
    print("Creating octree for Octree-based-region-growing...")
    octree = RegionGrowingOctree.RegionGrowingOctree(ds_pcd, root_margin=0.1)
    print("Performing initial voxelization...")

    octree.initial_voxelization(initial_voxel_size)

    octree.recursive_subdivide(minimum_voxel_size=subdivision_minimum_voxel_size,
                               residual_threshold=subdivision_residual_threshold,
                               full_threshold=subdivision_full_threshold,
                               max_depth=9,  # A higher max depth is not recommended
                               profile=True)

    print("Growing regions...")
    octree.grow_regions(minimum_valid_segment_size=minimum_valid_segment_size,
                        residual_threshold=region_growing_residual_threshold,
                        normal_deviation_threshold_degrees=growing_normal_deviation_threshold_degrees,
                        residual_threshold_is_absolute=False,
                        profile=False)

    print("Refining...")
    octree.refine_regions(planar_amount_threshold=fast_refinement_planar_amount_threshold,
                          planar_distance_threshold=fast_refinement_planar_distance_threshold,
                          fast_refinement_distance_threshold=fast_refinement_distance_threshold,
                          buffer_zone_size=general_refinement_buffer_size,
                          angular_divergence_threshold_degrees=refining_normal_deviation_threshold_degrees)

    if visualize:
        print("Visualizing voxels...")
        # octree.visualize_voxels(maximum=2000, segments=segments)
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_as_points(octree, True)
        regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_with_points(octree)
    print("Done!")


def pointnetv2(model_checkpoint_path: str,
               pcd: open3d.geometry.PointCloud):
    number_of_classes = 13  # PointNet++ is trained on the S3DIS dataset, which has 13 classes.
    classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
               'board', 'clutter']

    class_colors = [("gray", np.array([0.5, 0.5, 0.5])),
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

    classifier: torch.nn.Module = pointnetexternal.models.pointnet2_sem_seg.get_model(number_of_classes).cuda()

    # Load the checkpoint (i.e. the saved model)
    checkpoint = torch.load(model_checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Set the model into evaluation mode
    classifier = classifier.eval()

    # Prepare the input
    points: np.ndarray = np.asarray(pcd.points)
    colors: np.ndarray = np.asarray(pcd.colors)
    data = np.hstack((points, colors))

    npoint = 4096
    batches, batches_indices = convert_to_batches(data, point_amount=npoint, block_size=1, stride=0.5)
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
        print(f"Class {classes[i]} (color {class_colors[i][0]} : {class_colors[i][1]}) occurred {np.count_nonzero(classifications == i)} times.")

    numpy_class_colors = np.array([c[1] for c in class_colors])
    colors_per_point: np.ndarray = numpy_class_colors[classifications]
    visualize_pcd = open3d.geometry.PointCloud(pcd.points)
    visualize_pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
    open3d.visualization.draw_geometries([visualize_pcd])


def convert_to_batches(data,
                       block_size: float = 0.5,
                       stride: float = 0.1,
                       padding: float = 0.001,
                       point_amount: int = 4096,
                       batch_size: int = 32):

    """
    Convert point cloud data into batches that can be fed into the pointnet++ network.

    Parameters:
        data (np.ndarray): point cloud data to be converted to batches. Must be of shape (n_points, 6).
        block_size (float): the size of the 'blocks' that will be used to group points in the batches. Only applies
            to x and y dimensions. For example, of value of 1.0 will group points in 'pillars' of 1x1 meters.
        padding (float): the amount of padding to be added to each block to avoid missing points on the edge.
        stride (float): the stride size of the blocks, convolution style. This makes sure the blocks overlap and
            points are in multiple blocks / batches.
        point_amount (int): the minimum amount of points in per batch entry. If there are more points in the block than
            the points amount, the batch element will just be larger. If there are fewer points in the block than the
            point amount, random choice with replacement will be used to fill the gaps.
        batch_size: The size of each batch.
    """

    minimum_coordinates: np.ndarray = np.amin(data, axis=0)[:3]
    maximum_coordinates: np.ndarray = np.amax(data, axis=0)[:3]
    grid_x = int(np.ceil(float(maximum_coordinates[0] - minimum_coordinates[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(maximum_coordinates[1] - minimum_coordinates[1] - block_size) / stride) + 1)

    blocks_data = None
    indices = None

    print(f"Dividing point cloud into batches using (XY) blocks of size {block_size} and stride {stride}")
    for index_y in tqdm.trange(0, grid_y, desc="Dividing into blocks..."):
        for index_x in range(0, grid_x):
            s_x: float = minimum_coordinates[0] + index_x * stride
            e_x: float = min(s_x + block_size, maximum_coordinates[0])
            s_x = e_x - block_size
            s_y: float = minimum_coordinates[1] + index_y * stride
            e_y: float = min(s_y + block_size, maximum_coordinates[1])
            s_y = e_y - block_size

            point_idxs: np.ndarray = np.where((data[:, 0] >= s_x - padding) &
                                              (data[:, 0] <= e_x + padding) &
                                              (data[:, 1] >= s_y - padding) &
                                              (data[:, 1] <= e_y + padding))[0]

            if point_idxs.size == 0:
                continue

            # The amount of arrays that we need to accommodate this block, i.e. the amount of arrays we need
            # to fit all the points in the block such that each array is of size 'point_amount x C'
            num_batch: int = int(np.ceil(point_idxs.size / point_amount))

            # The amount of total points we need in order to neatly fill each array in the batch
            point_size: int = int(num_batch * point_amount)

            # Whether we need to reuse points in order to neatly fill each array
            replace: bool = point_size - point_idxs.size > point_idxs.size

            # The repeated indices such that we neatly fill the arrays and every array is of size 'point amount x C'
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))

            # Shuffle the indices and create the data
            np.random.shuffle(point_idxs)
            block_data = data[point_idxs, :]
            normalized_xyz = np.zeros((point_size, 3))
            normalized_xyz[:, 0] = block_data[:, 0] / maximum_coordinates[0]
            normalized_xyz[:, 1] = block_data[:, 1] / maximum_coordinates[1]
            normalized_xyz[:, 2] = block_data[:, 2] / maximum_coordinates[2]
            block_data[:, 0] = block_data[:, 0] - (s_x + block_size / 2.0)
            block_data[:, 1] = block_data[:, 1] - (s_y + block_size / 2.0)
            block_data[:, 3:6] /= 255.0
            block_data = np.concatenate((block_data, normalized_xyz), axis=1)

            blocks_data = np.vstack((blocks_data, block_data)) if blocks_data is not None else block_data
            indices = np.hstack((indices, point_idxs)) if indices is not None else point_idxs

    # All the blocks, neatly in a multiple of point amount
    blocks_data = blocks_data.reshape((-1, point_amount, blocks_data.shape[1]))
    indices = indices.reshape((-1, point_amount))

    number_of_batches = int(np.ceil(blocks_data.shape[0] / block_size))
    batches = []
    batches_indices = []
    for i in tqdm.trange(number_of_batches, desc="Generating batches out of the blocks..."):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, blocks_data.shape[0])
        batches.append(blocks_data[start_index:end_index])
        batches_indices.append(indices[start_index:end_index])

    return batches, batches_indices


def assign_noise_nearest_neighbour_cluster(points: np.ndarray, cluster_per_point: np.ndarray) -> np.ndarray:
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
    _, nearest_neighbour_indices = kd_tree.query(noise_points, k=1, workers=-1)
    new_clusters = cluster_per_point[data_points_indices[nearest_neighbour_indices]]
    return new_clusters
