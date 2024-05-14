import sys
from pathlib import Path

import numpy as np
import open3d
import tqdm

import segmentation
import utilities.HDBSCANConfig


def execute():
    point_cloud_path = "C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud_preprocessed.ply"

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

    hdbscan_config_path = "C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-segmentation\\results\\training-complex\\hdbscan\\config.json"
    hdbscan_config_path = Path(hdbscan_config_path)
    result_path = hdbscan_config_path.parent.joinpath("results.csv")
    hdbscan_configs = utilities.HDBSCANConfig.read_from_file_multiple(hdbscan_config_path)

    for i in tqdm.trange(len(hdbscan_configs), desc="Executing HDBSCANs..."):
        segmentation.hdbscan(pcd, hdbscan_configs[i], verbose=False)

    utilities.HDBSCANConfig.write_multiple(hdbscan_configs, result_path, delimiter=";")


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
