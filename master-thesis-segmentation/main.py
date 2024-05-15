import copy
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import open3d
import tqdm

import segmentation
import utilities.HDBSCANConfig


def get_points_and_labels(data_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if data_path.suffix in [".ply", ".pcd"]:
        print("Loading point cloud...")
        pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(str(data_path))
        downsample: bool = True
        if downsample:
            voxel_size = 0.01
            print(f"Downsampling point cloud using voxel size {voxel_size}")
            pcd = pcd.voxel_down_sample(voxel_size)
        print(f"Loaded point cloud with {len(pcd.points)} points.")

        assert pcd.has_normals()
        assert pcd.has_colors()
        return np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors))), None

    elif data_path.suffix == ".npy":
        print("Loading npy data file")
        data = np.load(data_path)
        assert data.shape[1] == 10
        return data[:, :9], data[:, 9]

    print("Loading failed.")

def execute():

    data_path = "C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-segmentation\\data\\s3dis_npy_incl_normals"

    for npy_file in tqdm.tqdm(os.listdir(data_path), desc="Clustering S3DIS point clouds..."):
        current_file_path = Path(data_path).joinpath(str(npy_file))
        if not current_file_path.suffix == ".npy":
            continue

        points, labels = get_points_and_labels(current_file_path)
        execute_hdbscan_on_data(segmentation.CLASS_COLORS, labels, points, str(current_file_path.stem))

    # pointnet_checkpoint_path = ("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal"
    #                            "\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\pretrained_original.pth")

    # segmentation.pointnetv2(pointnet_checkpoint_path, pcd)

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


def execute_hdbscan_on_data(class_colors: list,
                            labels: Optional[np.ndarray],
                            points: np.ndarray,
                            dataset_name_override: str = ""):
    normalize_coordinates = False
    if normalize_coordinates:
        min_coords = np.min(points[:, :3], axis=0)
        points[:, :3] -= min_coords
        max_coords = np.max(points[:, :3], axis=0)
        points[:, :3] /= max_coords
    hdbscan_config_path = ("C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-segmentation\\results"
                           "\\training-complex\\hdbscan\\config.json")
    hdbscan_config_path = Path(hdbscan_config_path)
    result_path = hdbscan_config_path.parent.joinpath("results.csv")
    hdbscan_configs = utilities.HDBSCANConfig.read_from_file_multiple(hdbscan_config_path, dataset_name_override)
    for i in tqdm.trange(len(hdbscan_configs), desc="Executing HDBSCANs..."):
        config = hdbscan_configs[i]
        segmentation.hdbscan(points, config, verbose=False)
        if labels is not None:
            cluster_label_map = config.assign_labels_to_clusters(labels=labels)
            if config.visualize:
                pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
                class_colors_np = np.array([a[1] for a in class_colors])
                colors = class_colors_np[cluster_label_map[config.clusters]]
                pcd.colors = open3d.utility.Vector3dVector(colors)
                open3d.visualization.draw_geometries([pcd])
    utilities.HDBSCANConfig.write_multiple(hdbscan_configs, result_path, delimiter=";")


if __name__ == '__main__':
    execute()
