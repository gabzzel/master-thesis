import math
import cProfile
from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN
import open3d

import regionGrowingOctree.RegionGrowingOctreeVisualization
from regionGrowingOctree import RegionGrowingOctree


def hdbscan(pcd: open3d.geometry.PointCloud):
    # The minimum number of samples in a group for that group to be considered a cluster;
    # groupings smaller than this size will be left as noise.
    min_cluster_size: int = 10

    # The number of samples in a neighborhood for a point to be considered as a core point.
    # This includes the point itself. When None, defaults to min_cluster_size.
    min_samples: int = None

    # A distance threshold. Clusters below this value will be merged.
    # I probably need to keep this to 0 to keep to the original HDBSCAN method.
    cluster_selection_epsilon: float = 0.0

    # A distance scaling parameter as used in robust single linkage. See [3] for more information.
    alpha: float = 1.0

    # Leaf size for trees responsible for fast nearest neighbour queries when a KDTree or a BallTree are used as
    # core-distance algorithms. A large dataset size and small leaf_size may induce excessive memory usage.
    # If you are running out of memory consider increasing the leaf_size parameter. Ignored for algorithm="brute".
    leaf_size: int = 10

    # Number of jobs to run in parallel to calculate distances. None means 1 unless in a joblib.parallel_backend
    # context. -1 means using all processors. See Glossary for more details.
    n_jobs: int = -1

    model = HDBSCAN(min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    max_cluster_size=None,  # Allow large clusters
                    metric="euclidean",
                    metric_params=None,  # Not needed.
                    alpha=alpha,
                    algorithm="auto",  # uses KD-Tree if possible, else BallTree
                    leaf_size=leaf_size,
                    n_jobs=n_jobs,
                    cluster_selection_method="eom",  # Use the standard Excess Of Mass metric
                    allow_single_cluster=False,  # Having a single cluster probably means something has gone wrong...
                    store_centers=None,  # We don't need centroids or mediods
                    copy=True)  # Just to be safe.

    X: np.ndarray = np.asarray(pcd.points)
    # if pcd.has_colors():
    #    X = np.hstack((X, np.asarray(pcd.colors)), dtype=np.float32)

    return model.fit_predict(X)


def octree_based_region_growing(pcd: open3d.geometry.PointCloud,
                                initial_voxel_size: float = 0.1,
                                down_sample_voxel_size: Optional[float] = 0.01):
    original_number_of_points = len(pcd.points)
    if down_sample_voxel_size is not None and down_sample_voxel_size > 0.0:
        ds_pcd = pcd.voxel_down_sample(voxel_size=down_sample_voxel_size)
        print(f"Downsampled point cloud from {original_number_of_points} to {len(ds_pcd.points)} points")
    else:
        ds_pcd = pcd

    if not ds_pcd.has_normals():
        print("Computing normals...")
        r = down_sample_voxel_size * math.sqrt(12.0)
        ds_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=10))
        print("Orienting normals...")
        ds_pcd.orient_normals_consistent_tangent_plane(k=10)
        ds_pcd.normalize_normals()

    # open3d.visualization.draw_geometries([pcd, ds_pcd], point_show_normal=True)
    print("Creating octree for Octree-based-region-growing...")
    octree = RegionGrowingOctree.RegionGrowingOctree(ds_pcd, root_margin=0.1)
    print("Performing initial voxelization...")

    octree.initial_voxelization(initial_voxel_size)

    residual_threshold = 0.001
    octree.recursive_subdivide(minimum_voxel_size=0.001,
                               residual_threshold=residual_threshold,
                               full_threshold=10,
                               max_depth=9,  # A higher max depth is not recommended
                               profile=True)

    print("Growing regions...")
    octree.grow_regions(minimum_valid_segment_size=4,
                        residual_threshold=0.05,
                        normal_deviation_threshold_degrees=90.0,
                        residual_threshold_is_absolute=False,
                        profile=False)

    print("Refining...")
    octree.refine_regions(planar_amount_threshold=0.9,
                          planar_distance_threshold=0.01,
                          fast_refinement_distance_threshold=0.01,
                          buffer_zone_size=0.02,
                          angular_divergence_threshold_degrees=30.0)

    print("Visualizing voxels...")
    # octree.visualize_voxels(maximum=2000, segments=segments)
    # regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_as_points(octree, True)
    regionGrowingOctree.RegionGrowingOctreeVisualization.visualize_segments_with_points(octree)
    print("Done!")
