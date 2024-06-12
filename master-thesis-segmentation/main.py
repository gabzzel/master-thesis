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


def get_points_and_labels(data_path: Path,
                          down_sample_voxel_size: float = 0.01) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:

    if data_path.suffix in [".ply", ".pcd"]:
        print("Loading point cloud...")
        pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(str(data_path))
        assert len(pcd.points) > 0, "No point cloud found or invalid point cloud."

        if down_sample_voxel_size > 0:
            voxel_size = 0.01
            print(f"Downsampling point cloud using voxel size {voxel_size}")
            pcd = pcd.voxel_down_sample(down_sample_voxel_size)
        print(f"Loaded point cloud with {len(pcd.points)} points.")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        return points, colors, normals, None

    elif data_path.suffix == ".npy":
        print(f"Loading npy data file {data_path.name}")
        data = np.load(data_path)
        assert data.shape[1] == 10
        points = data[:, :3]
        colors = data[:, 3:6]
        normals = data[:, 6:9]
        labels = data[:, 9]
        return points, colors, normals, labels

    print("Loading failed.")


def execute():
    # extract_clusters()

    # execute_hdbscan_on_S3DIS()
    #execute_obrg_on_S3DIS()

    # pointnet_checkpoint_path = "C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\pretrained_original_coords_colors.pth"
    # pointcloud_path = Path("E:\\etvr_datasets\\ruimte_ETVR-preprocessed-lower-only.ply")
    # result_directory = Path("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\pointnetv2\\office")

    # points, colors, normals, _ = get_points_and_labels(pointcloud_path, down_sample_voxel_size=0.0)
    # segmentation.pointnetv2(model_checkpoint_path=pointnet_checkpoint_path,
    #                         points=points,
    #                         normals=None,
    #                        colors=colors,
    #                        working_directory=result_directory,
    #                        visualize_raw_classifications=True,
    #                        create_segmentations=False,
    #                        segmentation_max_distance=0.02)

    config = utilities.HDBSCANConfig.HDBSCANConfigAndResult(
        pcd_path="C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-reconstruction\\data\\etvr\\training_complex_downsampled_001_incl_oriented_normals.ply",
        min_cluster_size=125,
        min_samples=200,
        include_normals=True,
        include_colors=False,
        visualize=True
    )

    classifications = np.load("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\pointnext\\training-complex\\training_complex_downsampled_001_incl_oriented_normals_1718011697.0712814_classifications.npy")
    n_values = np.max(classifications) + 1
    classifications_one_hot = np.eye(n_values)[classifications]
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(config.pcd_path)
    points = np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors), classifications_one_hot))
    segmentation.hdbscan(points, config, verbose=True)
    results_folder = Path("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\hdbscan_incl_pointnext")
    utilities.HDBSCANConfig.write_multiple([config], results_folder.joinpath("result.txt"), delimiter="\n")
    np.save(results_folder.joinpath("cluster_per_point.npy"), config.clusters)
    print("Done!")


def extract_clusters():
    folder_path = Path("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\pointnext")
    file_path = "ruimte_ETVR-preprocessed_1718021719.7522366_classifications.npy"
    classifications = np.load(folder_path.joinpath(file_path))
    pcd_file = file_path.replace("classifications.npy", "pcd.ply")
    pcd = open3d.io.read_point_cloud(str(folder_path.joinpath(pcd_file)))
    points = np.asarray(pcd.points)

    start_time = time.time()
    clusters, cluster_per_point_raw = segmentation.extract_clusters_from_labelled_points_multicore(points, classifications,
                                                                                               max_distance=0.05)
    clusters_dest = folder_path.joinpath(file_path.replace("classifications.npy", "clusters.npy"))
    np.save(clusters_dest, cluster_per_point_raw)

    cluster_sizes = [len(i) for i in clusters]
    print(f"Cluster sizes: min {np.min(cluster_sizes)}, max {np.max(cluster_sizes)}, avg {np.average(cluster_sizes)}, median {np.median(cluster_sizes)}")
    print("Saved clusters. Total time taken: {:.2f}".format(time.time() - start_time))

    stats_path = folder_path.joinpath(file_path.replace("classifications.npy", "stats.txt"))
    with open(stats_path, "a") as f:
        f.write("Clustering time:" + str(time.time() - start_time) + "\n")

    rng = np.random.default_rng()
    colors = rng.random(size=(len(clusters), 3))
    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    pcd.colors = open3d.utility.Vector3dVector(colors[cluster_per_point_raw])
    open3d.visualization.draw_geometries([pcd])

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

        points, colors, normals, labels = get_points_and_labels(current_file_path)
        data = np.hstack((points, normals, colors))
        execute_hdbscan_on_data(segmentation.CLASS_COLORS, labels, data, str(current_file_path.stem))


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
    data_path = "C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal\\data\\s3dis_npy_incl_normals\\"
    result_dir_path = Path("E:\\thesis-results\\segmentation\\obrg")
    config = utilities.OctreeBasedRegionGrowingConfig.read_from_file(result_dir_path.joinpath("config.json"))
    config.segments_save_path = result_dir_path
    start_index = 0
    end_index = None

    files = sorted(os.listdir(data_path))
    if end_index is None:
        end_index = len(files)
    files = files[start_index:end_index]

    for npy_file in tqdm.tqdm(files, desc="Clustering S3DIS point clouds..."):
        current_file_path = Path(data_path).joinpath(str(npy_file))
        if not current_file_path.suffix == ".npy":
            continue

        points, colors, normals, labels = get_points_and_labels(current_file_path)
        current_config = copy.copy(config)
        current_config.data_set = str(current_file_path.stem)

        data = np.hstack((points, normals, colors))
        execute_obrg_on_data(data, labels, current_config, visualize=False, verbose=True)
        results_path = result_dir_path.joinpath("results.csv")
        utilities.OctreeBasedRegionGrowingConfig.write_results_to_file_multiple(results_path, [current_config])


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
    np.savetxt(segment_save_path.joinpath(f"segments-{time.time()}.txt"), octree.segment_index_per_point, fmt="%0i")




if __name__ == '__main__':
    execute()
