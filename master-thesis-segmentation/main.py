import sys

import numpy as np
import open3d

import segmentation


def execute():
    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud_preprocessed.ply"

    print("Loading point cloud...")
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(point_cloud_path)
    downsample: bool = True
    if downsample:
        voxel_size = 0.01
        print(f"Downsampling point cloud using voxel size {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # pointnet_checkpoint_path = ("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal"
    #                            "\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\pretrained_original.pth")

    # segmentation.pointnetv2(pointnet_checkpoint_path, pcd)

    cluster_per_points, membership_strengths = segmentation.hdbscan(pcd,
                                                                    minimum_cluster_size=100,
                                                                    minimum_samples=200,
                                                                    visualize=True,
                                                                    use_sklearn_estimator=False,
                                                                    use_colors=True,
                                                                    use_normals=True)

    return

    segmentation.octree_based_region_growing(pcd,
                                             initial_voxel_size=0.1,
                                             down_sample_voxel_size=None,

                                             # Subdivision parameters
                                             subdivision_residual_threshold=0.001,
                                             subdivision_full_threshold=4,
                                             subdivision_minimum_voxel_size=0.01,

                                             # Region growing parameters
                                             minimum_valid_segment_size=20,
                                             region_growing_residual_threshold=0.95,
                                             growing_normal_deviation_threshold_degrees=90,

                                             # Region refining / refinement parameter
                                             refining_normal_deviation_threshold_degrees=30,
                                             general_refinement_buffer_size=0.02,
                                             fast_refinement_planar_distance_threshold=0.02,
                                             fast_refinement_distance_threshold=0.05,
                                             fast_refinement_planar_amount_threshold=0.8,
                                             visualize=True)


if __name__ == '__main__':
    execute()
