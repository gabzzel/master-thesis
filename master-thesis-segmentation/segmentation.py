import math
import os
from typing import Optional, List, Union, Union

import fast_hdbscan
import numpy as np
import open3d
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import regionGrowingOctree.RegionGrowingOctreeVisualization
from pointnetPlusPlus.CustomDataset import S3DIS_Dataset_Area
from pointnetPlusPlus.pointnetv2 import PointNet2Segmentation
from regionGrowingOctree import RegionGrowingOctree


def hdbscan(pcd: open3d.geometry.PointCloud,
            minimum_cluster_size: Union[int, str] = 10,
            minimum_samples: Optional[Union[int, str]] = None,
            cluster_selection_epsilon: Union[float, str] = 0.0,
            method : str = "eom",
            visualize: bool = True,
            use_sklearn_estimator: bool = False):
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

    print(f"Clustering / segmenting using HDBScan (min cluster size {minimum_cluster_size}, min samples {minimum_samples}, method '{method}')")

    points = np.asarray(pcd.points)
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

    print("Clustering done.")
    print(f"Created {number_of_clusters} clusters.")

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


def pointnet_train(dataset_path: Union[str, os.PathLike],
                   number_of_classes: int,
                   batch_size: int = 64,
                   learning_rate: float = 0.001,
                   num_epochs: int = 10):
    print("Starting training... ")
    # Initialize dataset and dataloaders
    train_dataset = S3DIS_Dataset_Area(s3dis_root_dir=dataset_path, train=True, number_of_classes=number_of_classes)
    print("Initialized training dataset.")
    val_dataset = S3DIS_Dataset_Area(s3dis_root_dir=dataset_path, train=False, number_of_classes=number_of_classes)
    print("Initialized validation dataset.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Initialized training data loader.")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    print("Prepared data.")
    # Initialize model
    model = PointNet2Segmentation(set_abstraction_ratio_1=0.5,
                                  set_abstraction_radius_1=0.02,
                                  set_abstraction_ratio_2=0.5,
                                  set_abstraction_radius_2=0.02,
                                  dropout=0.1,
                                  number_of_classes=number_of_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Training the model...")
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            inputs, labels = data  # Load inputs and labels from DataLoader
            optimizer.zero_grad()  # Zero gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

        # Validate the model
        model.eval()
        with torch.no_grad(): # Disable gradients
            total_loss = 0.0
            for data in val_loader:
                inputs, labels = data  # Load inputs and labels from DataLoader
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                total_loss += loss.item() * inputs.size(0)
            val_loss = total_loss / len(val_dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'pointnet_segmentation_model.pth')


def pointnet_segment(model: PointNet2Segmentation,
                     pcd: open3d.geometry.PointCloud,
                     visualize: bool = True):

    points = torch.tensor(pcd.points)

    if pcd.has_normals():
        normals = torch.tensor(pcd.normals)
        points = torch.cat([points, normals])

    if pcd.has_colors():
        colors = torch.tensor(pcd.colors)
        points = torch.cat([points, colors])

    model.eval()
    with torch.no_grad():
        outputs = model(points)

    if visualize:
        visualization_pcd = open3d.geometry.PointCloud(pcd.points)
        rng = np.random.default_rng()
        label_colors = rng.random(size=(len(outputs), 3))
        color_per_point = label_colors[np.asarray(outputs)]
        visualization_pcd.colors = open3d.utility.Vector3dVector(color_per_point)
        open3d.visualization.draw_geometries([visualization_pcd])

    return outputs