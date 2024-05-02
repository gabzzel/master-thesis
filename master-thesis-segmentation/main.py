import math
import sys
import utilities
from pathlib import Path

import open3d
import numpy as np

import segmentation
import utilities.s3dis_reader

def segment_with_hdbscan(pcd):
    
    sys.setrecursionlimit(15000)

    # Office (downsampled 0.01, minimum cluster size 300 and minimum samples None, method EOM) = 43 GB RAM
    # Office (downsampled 0.01, minimum cluster size 400 and minimum samples None, method EOM) = 56 GB RAM
    # Office (downsampled 0.01, minimum cluster size 500 and minimum samples None, method EOM) = 68 GB RAM

    segmentation.hdbscan(pcd,
                        minimum_cluster_size=300,
                        minimum_samples=100,
                        cluster_selection_epsilon=0.0,
                        method="eom",
                        visualize=True)


def execute():
    use_downsampled: bool = True
    point_cloud_path = "C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-reconstruction\\data\\etvr\\training_complex.pcd"

    print("Loading point cloud...")
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(point_cloud_path)
    pcd = pcd.voxel_down_sample(0.01)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    pointnet_checkpoint_path = ("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal"
                                "\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\pretrained_original.pth")

    segmentation.pointnetv2(pointnet_checkpoint_path, pcd)

    return
    print("Clustering...")
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
                                             growing_normal_deviation_threshold_degrees=45,

                                             # Region refining / refinement parameter
                                             refining_normal_deviation_threshold_degrees=30,
                                             general_refinement_buffer_size=0.02,
                                             fast_refinement_planar_distance_threshold=0.01,
                                             fast_refinement_distance_threshold=0.05,
                                             fast_refinement_planar_amount_threshold=0.8,
                                             visualize=True)


if __name__ == '__main__':
    execute()

