import math
import cProfile
from typing import Optional

import numpy as np
from sklearn.cluster import HDBSCAN
import fast_hdbscan
import open3d

import regionGrowingOctree.RegionGrowingOctreeVisualization
from regionGrowingOctree import RegionGrowingOctree


def hdbscan(pcd: open3d.geometry.PointCloud,
            minimum_cluster_size: int = 10,
            minimum_samples: Optional[int] = None,
            cluster_selection_epsilon: float = 0.0,
            visualize: bool = True):
    """

    :param visualize: Whether to draw the segments to the screen using Open3D visualization.
    :param cluster_selection_epsilon: A distance threshold. Clusters below this value will be merged. \
        I probably need to keep this to 0 to keep to the original HDBSCAN method.
    :param minimum_samples: The number of samples in a neighborhood for a point to be considered as a core point. \
        This includes the point itself. When None, defaults to min_cluster_size.
    :param pcd: The point cloud to segment.
    :param minimum_cluster_size: The minimum number of samples in a group for that group to be considered a cluster; \
        groupings smaller than this size will be left as noise.
    :return:
    """

    points = np.asarray(pcd.points)

    model = fast_hdbscan.HDBSCAN(min_cluster_size=minimum_cluster_size,
                                 min_samples=minimum_samples,
                                 cluster_selection_method="eom",
                                 allow_single_cluster=False,
                                 cluster_selection_epsilon=cluster_selection_epsilon)

    cluster_per_point = model.fit_predict(points)

    # if pcd.has_colors():
    #    X = np.hstack((X, np.asarray(pcd.colors)), dtype=np.float32)

    print("Clustering done.")

    if visualize:
        unique_clusters = np.unique(cluster_per_point)
        rng = np.random.default_rng()
        colors_per_cluster = rng.random((len(unique_clusters), 3), dtype=np.float64)
        colors = colors_per_cluster[cluster_per_point]
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd])


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
        for region growing.
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
