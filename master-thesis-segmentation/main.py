import copy
import os
import time
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import open3d
import tqdm

import segmentation
import utilities.HDBSCANConfig
import utilities.OctreeBasedRegionGrowingConfig
from utilities.OctreeBasedRegionGrowingConfig import OctreeBasedRegionGrowingConfig


def get_points_and_labels(data_path: Path, down_sample_voxel_size: float = 0.01) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if data_path.suffix in [".ply", ".pcd"]:
        print("Loading point cloud...")
        pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(str(data_path))
        if down_sample_voxel_size > 0:
            voxel_size = 0.01
            print(f"Downsampling point cloud using voxel size {voxel_size}")
            pcd = pcd.voxel_down_sample(down_sample_voxel_size)
        print(f"Loaded point cloud with {len(pcd.points)} points.")

        assert pcd.has_normals()
        assert pcd.has_colors()
        return np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors))), None

    elif data_path.suffix == ".npy":
        print(f"Loading npy data file {data_path.name}")
        data = np.load(data_path)
        assert data.shape[1] == 10
        return np.hstack((data[:, :3], data[:, 6:9], data[:, 3:6])), data[:, 9]

    print("Loading failed.")


def execute():
    # execute_hdbscan_on_S3DIS()
    # execute_obrg_on_S3DIS()

    pointnet_checkpoint_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\pointnetexternal\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\best_model.pth"
    pointcloud_path = Path("E:\\etvr_datasets\\enfsi-2023_reduced_cloud_preprocessed.ply")
    result_directory = Path("C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\results\\pointnetv2")

    points, _ = get_points_and_labels(pointcloud_path, down_sample_voxel_size=0.0)
    segmentation.pointnetv2(model_checkpoint_path=pointnet_checkpoint_path,
                            points=points,
                            working_directory=result_directory,
                            visualize_raw_classifications=True,
                            create_segmentations=True,
                            segmentation_max_distance=0.02)


def execute_hdbscan_on_S3DIS():
    data_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\data"

    start_index = 0
    all_files = sorted(os.listdir(data_path))

    area = 1

    for npy_file in tqdm.tqdm(all_files[start_index:], desc=f"Clustering S3DIS area {area}"):
        current_file_path = Path(data_path).joinpath(str(npy_file))
        if not current_file_path.suffix == ".npy":
            continue

        if area is not None and f"Area_{area}" not in current_file_path.name:
            continue

        points, labels = get_points_and_labels(current_file_path)
        execute_hdbscan_on_data(segmentation.CLASS_COLORS, labels, points, str(current_file_path.stem))


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
    hdbscan_config_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\results\\config.json"
    hdbscan_config_path = Path(hdbscan_config_path)

    if not hdbscan_config_path.exists():
        raise FileNotFoundError(hdbscan_config_path)

    result_path = hdbscan_config_path.parent.joinpath("results.csv")
    hdbscan_configs = utilities.HDBSCANConfig.read_from_file_multiple(hdbscan_config_path, dataset_name_override)
    for i in tqdm.trange(len(hdbscan_configs), desc="Executing HDBSCANs..."):
        config = hdbscan_configs[i]
        segmentation.hdbscan(points, config, verbose=True)
        if labels is not None:
            cluster_label_map = config.assign_labels_to_clusters(labels=labels)
            if config.visualize:
                pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
                class_colors_np = np.array([a[1] for a in class_colors])
                colors = class_colors_np[cluster_label_map[config.clusters]]
                pcd.colors = open3d.utility.Vector3dVector(colors)
                open3d.visualization.draw_geometries([pcd])
    utilities.HDBSCANConfig.write_multiple(hdbscan_configs, result_path, delimiter=";")


def execute_obrg_on_S3DIS():
    data_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\data\\"
    result_dir_path = Path("E:\\thesis-results\\segmentation\\obrg")
    config = utilities.OctreeBasedRegionGrowingConfig.read_from_file(result_dir_path.joinpath("config.json"))
    config.segments_save_path = result_dir_path
    start_index = 0
    end_index = None

    files = sorted(os.listdir(data_path))
    if end_index is None:
        end_index = len(files)
    files = files[start_index:end_index]


    configs = []
    for npy_file in tqdm.tqdm(files, desc="Clustering S3DIS point clouds..."):
        current_file_path = Path(data_path).joinpath(str(npy_file))
        if not current_file_path.suffix == ".npy":
            continue

        points, labels = get_points_and_labels(current_file_path)
        current_config = copy.copy(config)
        current_config.data_set = str(current_file_path.stem)
        configs.append(current_config)
        execute_obrg_on_data(points, labels, current_config, visualize=True, verbose=True)

    results_path = result_dir_path.joinpath("results.csv")
    utilities.OctreeBasedRegionGrowingConfig.write_results_to_file_multiple(results_path, configs)


def execute_obrg_on_data(points: np.ndarray,
                         labels: Optional[np.ndarray],
                         config: Union[str, os.PathLike, OctreeBasedRegionGrowingConfig],
                         visualize: bool = True,
                         verbose: bool = True):

    assert points.ndim == 2
    assert points.shape[1] == 9
    assert points.shape[0] > 0

    if isinstance(config, (str, Path)):
        config = utilities.OctreeBasedRegionGrowingConfig.read_from_file(Path(config))

    start_time = time.time()
    octree = segmentation.octree_based_region_growing(points,
                                                      config,
                                                      visualize=visualize,
                                                      verbose=verbose)

    end_time = time.time()
    config.total_time = end_time - start_time

    if labels is None:
        return

    cluster_label_map = octree.assign_labels_to_clusters(classes=segmentation.CLASSES, config=config, labels=labels)

    if visualize:
        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
        class_colors_np = np.array([a[1] for a in segmentation.CLASS_COLORS])
        colors = class_colors_np[cluster_label_map[octree.segment_index_per_point]]
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd])

    if config.segments_save_path is None:
        return

    segment_save_path = config.segments_save_path if isinstance(config.segments_save_path, Path) else Path(config.segments_save_path)
    np.savetxt(segment_save_path.joinpath(f"segments-{time.time()}.txt"), octree.segment_index_per_point)




if __name__ == '__main__':
    execute()
