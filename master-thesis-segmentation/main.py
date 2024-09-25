import argparse
import copy
import math
import os
import sys
import time
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
        downsample: bool = False
        if downsample:
            voxel_size = 0.01
            print(f"Downsampling point cloud using voxel size {voxel_size}")
            pcd = pcd.voxel_down_sample(voxel_size)
        print(f"Loaded point cloud with {len(pcd.points)} points at {data_path}")

        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros(shape=(len(pcd.points), 3))
        return np.hstack((np.asarray(pcd.points), normals, np.asarray(pcd.colors))), None

    elif data_path.suffix == ".npy":
        print(f"Loading npy data file {data_path.name}")
        data = np.load(data_path)
        assert data.shape[1] == 10
        return data[:, :9], data[:, 9]

    print("Loading failed.")


def execute():
    args = parse_args()

    if args.method == "hdbscan":
        execute_hdbscan_with_args(args)
    elif args.method == "pointnetv2":
        execute_pointnetv2_with_args(args)
    else:
        print("Invalid method")


def execute_hdbscan_with_args(args):
    config = utilities.HDBSCANConfig.HDBSCANConfigAndResult(
        pcd_path=args.point_cloud_path,
        min_cluster_size=args.hdbscan_min_cluster_size,
        min_samples=args.hdbscan_min_samples,
        include_normals=args.include_normals,
        include_colors=args.include_colors,
        visualize=False
    )
    config.noise_nearest_neighbours = min(max(args.hdbscan_noise_nearest_neighbours, 0), 1000)
    if args.verbose:
        print("Loading point cloud...")
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(str(config.pcd_path))

    if args.verbose and args.downsampling_method is not None and args.downsampling_method != "none":
        print("Downsampling point cloud...")

    if args.downsampling_method == "random":
        pcd = pcd.random_down_sample(args.downsampling_param)
    elif args.downsampling_method == "voxel":
        pcd = pcd.voxel_down_sample(args.downsampling_param)

    if args.verbose and (args.normal_estimation_radius > 0 or args.normal_estimation_neighbours > 0):
        print("Estimating normals...")

    if args.normal_estimation_radius > 0 and args.normal_estimation_neighbours > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_estimation_radius,
                                                                     max_nn=args.normal_estimation_neighbours))
    elif args.normal_estimation_radius > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamRadius(args.normal_estimation_radius))
    elif args.normal_estimation_neighbours > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(args.normal_estimation_neighbours))

    if pcd.has_normals() and args.normal_orientation_neighbours > 0:
        if args.verbose:
            print("Orienting normals... (Can take a while)")

        pcd.orient_normals_consistent_tangent_plane(k=args.normal_orientation_neighbours)

    points = np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors)))
    if args.verbose:
        print("Executing HDBSCAN...")
    segmentation.hdbscan(points, config, verbose=args.verbose)

    if args.verbose:
        print("Saving results...")
    results_path = args.result_path.joinpath(f"{args.point_cloud_path.stem}_clusters.npy")
    np.save(results_path, config.clusters)


def execute_pointnetv2_with_args(args):

    if args.verbose:
        print("Loading point cloud...")
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(str(args.point_cloud_path))

    if args.verbose and args.downsampling_method is not None and args.downsampling_method != "none":
        print("Downsampling point cloud...")

    if args.downsampling_method == "random":
        pcd = pcd.random_down_sample(args.downsampling_param)
    elif args.downsampling_method == "voxel":
        pcd = pcd.voxel_down_sample(args.downsampling_param)

    if args.verbose and (args.normal_estimation_radius > 0 or args.normal_estimation_neighbours > 0):
        print("Estimating normals...")

    if args.normal_estimation_radius > 0 and args.normal_estimation_neighbours > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_estimation_radius,
                                                                     max_nn=args.normal_estimation_neighbours))
    elif args.normal_estimation_radius > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamRadius(args.normal_estimation_radius))
    elif args.normal_estimation_neighbours > 0:
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(args.normal_estimation_neighbours))

    if pcd.has_normals() and args.normal_orientation_neighbours > 0:
        if args.verbose:
            print("Orienting normals... (Can take a while)")

        pcd.orient_normals_consistent_tangent_plane(k=args.normal_orientation_neighbours)

    if args.verbose:
        print("Classifying with PointNetV2...")

    classifications, cluster_indexes = segmentation.pointnetv2(str(args.pointnetv2_checkpoint_path),
                                                               points=np.asarray(pcd.points),
                                                               colors=np.asarray(pcd.colors) if args.include_colors else None,
                                                               normals=np.asarray(pcd.normals) if args.include_normals else None,
                                                               working_directory=args.point_cloud_path.parent,
                                                               visualize_raw_classifications=False,
                                                               create_segmentations=args.do_segmentation,
                                                               segmentation_max_distance=args.segmentation_max_distance)

    classifications_path = args.result_path.joinpath(f"{args.point_cloud_path.stem}_classifications.npy")
    np.save(classifications_path, classifications)

    if args.do_segmentation:
        clusters_path = args.result_path.joinpath(f"{args.point_cloud_path.stem}_clusters.npy")
        np.save(clusters_path, cluster_indexes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('point_cloud_path', type=Path, help='Path to the point cloud file',
                        action='store')
    parser.add_argument('method', type=str, help='Which segmentation method to use.', action='store',
                        choices=['pointnetv2', 'hdbscan'])
    parser.add_argument('-result_path', type=Path, help='Folder where the result will be saved',
                        required=False, default=None)
    parser.add_argument('-downsampling_method', type=str, help='Which downsampling method to use.',
                        action='store', choices=['voxel', 'random', 'none', None], default='none')
    parser.add_argument('-downsampling_param', type=float, action='store', default=0.1,
                        help="Downsampling parameter. Represents voxel size if downsampling method is 'voxel' "
                             "and sampling ratio if downsampling method is 'random'")
    parser.add_argument('-normal_estimation_radius', type=float, action='store', default=0.1,
                        help="The radius within which neighbours are used around a target point during normal "
                             "estimation. Recommended value is 0.1. If using voxel downsampling, recommended to use"
                             " approximately 3.5 * voxel size")
    parser.add_argument('-normal_estimation_neighbours', type=int, action='store', default=10,
                        help="The number of neighbours around a target point during normal estimation.")
    parser.add_argument('-normal_orientation_neighbours', type=int, action='store', default=10,
                        help="The number of neighbours used around a target point during normal orientation.")
    parser.add_argument('-hdbscan_min_cluster_size', type=int, action='store', default=125,
                        help="The minimum number of points in a hdbscan cluster.")
    parser.add_argument('-hdbscan_min_samples', type=int, action='store', default=200,
                        help="The minimum number of core points that need to be around a point for that point"
                             " to be considered (in) a hdbscan cluster.")
    parser.add_argument('-include_colors', action='store', default=False, type=bool,
                        help="whether to include color-data during clustering. (Not recommended)")
    parser.add_argument('-include_normals', action='store', default=True, type=bool,
                        help="Whether to include normal data during clustering (highly recommended!)")
    parser.add_argument('-verbose', action='store', default=True, type=bool,
                        help="whether to print progress and result information.")
    parser.add_argument('-pointnetv2_checkpoint_path', type=Path, action='store',
                        help="Path to the checkpoint for pointnetv2 (i.e. the trained model).")
    parser.add_argument('-do_segmentation', type=bool, action='store', default=False,
                        help="whether to segment the point cloud after classification.")
    parser.add_argument('-segmentation_max_distance', type=float, action='store', default=0.2,
                        help="When creating the segmentation for pointnetv2 classifications, this is the maximum "
                             "distance between points such that they are considered part of the same cluster.")
    parser.add_argument('-hdbscan_noise_nearest_neighbours', type=int, action='store', default=3,
                        help="When HDBSCAN fails to assign points to a cluster (i.e. noise points), these point can "
                             "be forcefully assigned to the most occuring cluster of its nearest neighbours. This"
                             "parameter determines the amount of neighbours considered for this operation. Default is 3."
                             " Set to 0 to disable this automatic assignment of noise points.")
    args = parser.parse_args()
    return args


def cluster_manual():
    paths = [
        "E:\\thesis-results\\segmentation\\pointnext\\office\\ruimte_ETVR-preprocessed_1719156546.4426675_classifications.npy",
        "E:\\thesis-results\\segmentation\\pointnext\\office (cleaned)\\ruimte_ETVR-preprocessed-lower-cleaned_1719161060.2401881_classifications.npy",
        "E:\\thesis-results\\segmentation\\pointnext\\tc\\training_complex_downsampled_001_incl_oriented_normals_1719147859.9874692_classifications.npy",
        "E:\\thesis-results\\segmentation\\pointnext\\zuidberg\\Zuidberg-preprocessed_1719161201.6034386_classifications.npy",
        "E:\\thesis-results\\segmentation\\pointnext\\zuidberg (cleaned)\\Zuidberg-preprocessed-clean_1719161305.087092_classifications.npy"
    ]

    for string_path in paths:
        classification_path = Path(string_path)
        pcd_path = string_path.replace("_classifications.npy", "_pcd.ply")
        pcd = open3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        labels_per_point = np.load(classification_path)
        start_time = time.time()
        _, clusters_per_point = segmentation.extract_clusters_from_labelled_points_multiprocess(points=points,
                                                                                                labels_per_point=labels_per_point,
                                                                                                max_distance=0.02)
        cluster_time = time.time() - start_time
        stats_path = string_path.replace("_classifications.npy", "_stats.txt")
        with open(stats_path, "a") as f:
            f.write(f"Clustering Time={cluster_time}\n")
        cluster_path = string_path.replace("_classifications.npy", "_clusters.npy")
        np.save(cluster_path, clusters_per_point)

        rng = np.random.default_rng()
        colors = rng.random((np.max(clusters_per_point) + 1, 3))
        colors_per_point = colors[clusters_per_point]
        new_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        new_pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
        new_pcd_path = string_path.replace("_classifications.npy", "_clusters.ply")
        open3d.io.write_point_cloud(new_pcd_path, new_pcd)


def execute_hdbscan_manual():
    datasets_path = Path("E:\\etvr_datasets")
    datasets_names = [
        "enfsi-2023_reduced.pcd",
        "ruimte_ETVR.ply",
        "Zuidberg.ply"
    ]
    for dataset_name in datasets_names:
        min_cluster_size = 125
        min_samples = 200

        config = utilities.HDBSCANConfig.HDBSCANConfigAndResult(
            pcd_path=str(datasets_path.joinpath(dataset_name)),
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            include_normals=True,
            include_colors=False,
            visualize=False
        )

        start_time = time.time()
        pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(config.pcd_path)
        pcd = pcd.voxel_down_sample(0.01)
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=math.sqrt(12) * 0.01, max_nn=10))
        pcd.orient_normals_consistent_tangent_plane(k=10)
        points = np.hstack((np.asarray(pcd.points), np.asarray(pcd.normals), np.asarray(pcd.colors)))
        loading_time = time.time() - start_time

        start_time = time.time()
        segmentation.hdbscan(points, config, verbose=False)
        segmentation_time = time.time() - start_time

        print(f"{dataset_name};{min_samples};{min_cluster_size};{loading_time};{segmentation_time}")

        # results_folder = Path("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\hdbscan")
        # utilities.HDBSCANConfig.write_multiple([config], results_folder.joinpath("result.txt"), delimiter="\n")
        # np.save(results_folder.joinpath("cluster_per_point.npy"), config.clusters)


def execute_pointnetv2_manual():
    pointnet_checkpoint_path = "C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\pointnetexternal\\log\\sem_seg\\pointnet2_sem_seg\\checkpoints\\pretrained_original_coords_colors.pth"

    dataset_path = Path("E:\\etvr_datasets")

    datasets_names = [
        "Zuidberg-cleaned.ply"
    ]

    for dataset_name in datasets_names:
        pcd_path = dataset_path.joinpath(dataset_name)
        result_directory = Path("E:\\thesis-results\\segmentation\\pointnetv2\\all")
        start_time = time.time()
        points, labels = get_points_and_labels(pcd_path)
        loading_time = time.time() - start_time
        print(loading_time)

        segmentation.pointnetv2(model_checkpoint_path=pointnet_checkpoint_path,
                                points=points[:, :3],
                                normals=None,
                                colors=points[:, 6:9],
                                working_directory=result_directory,
                                visualize_raw_classifications=True,
                                create_segmentations=True,
                                segmentation_max_distance=0.02)


def extract_clusters():
    folder_path = Path("C:\\Users\\admin\\gabriel-master-thesis\\master-thesis-segmentation\\results\\pointnext")
    file_path = "ruimte_ETVR-preprocessed_1718021719.7522366_classifications.npy"
    classifications = np.load(folder_path.joinpath(file_path))
    pcd_file = file_path.replace("classifications.npy", "pcd.ply")
    pcd = open3d.io.read_point_cloud(str(folder_path.joinpath(pcd_file)))
    points = np.asarray(pcd.points)

    start_time = time.time()
    clusters, cluster_per_point_raw = segmentation.extract_clusters_from_labelled_points_multicore(points,
                                                                                                   classifications,
                                                                                                   max_distance=0.05)
    clusters_dest = folder_path.joinpath(file_path.replace("classifications.npy", "clusters.npy"))
    np.save(clusters_dest, cluster_per_point_raw)

    cluster_sizes = [len(i) for i in clusters]
    print(
        f"Cluster sizes: min {np.min(cluster_sizes)}, max {np.max(cluster_sizes)}, avg {np.average(cluster_sizes)}, median {np.median(cluster_sizes)}")
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
    data_path = "C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-segmentation\\data\\s3dis_npy_incl_normals"

    start_index = 48
    all_files = sorted(os.listdir(data_path))

    area = None

    if area is not None:
        all_files = [i for i in all_files if f"Area_{area}" in i]

    for npy_file in tqdm.tqdm(all_files[start_index:], desc=f"Clustering S3DIS area {area}"):
        current_file_path = Path(data_path).joinpath(str(npy_file))
        if not current_file_path.suffix == ".npy":
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
    hdbscan_config_path = "C:\\Users\\ETVR\\Documents\\gabriel-master-thesis\\master-thesis-segmentation\\results\\office\\hdbscan\\config.json"
    hdbscan_config_path = Path(hdbscan_config_path)

    if not hdbscan_config_path.exists():
        raise FileNotFoundError(hdbscan_config_path)

    result_path = hdbscan_config_path.parent.joinpath("results.csv")
    hdbscan_configs = utilities.HDBSCANConfig.read_from_file_multiple(hdbscan_config_path, dataset_name_override)
    for i in tqdm.trange(len(hdbscan_configs), desc="Executing HDBSCANs..."):
        current_time = time.time()
        config = hdbscan_configs[i]
        segmentation.hdbscan(points, config, verbose=False)
        np.save(result_path.parent.joinpath(f"cluster_per_point-{current_time}.npy"), config.clusters)
        if labels is not None and config.clusters is not None:
            cluster_label_map = config.assign_labels_to_clusters(labels=labels, verbose=False)
            if config.visualize:
                pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
                class_colors_np = np.array([a[1] for a in class_colors])
                colors = class_colors_np[cluster_label_map[config.clusters]]
                pcd.colors = open3d.utility.Vector3dVector(colors)
                open3d.visualization.draw_geometries([pcd])

            np.save(result_path.parent.joinpath(f"label_per_point-{current_time}.npy"),
                    cluster_label_map[config.clusters])

    utilities.HDBSCANConfig.write_multiple(hdbscan_configs, result_path, delimiter=";")


def execute_obrg_on_S3DIS():
    data_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-segmentation\\data\\"
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

        points, labels = get_points_and_labels(current_file_path)
        execute_obrg_on_data(points, labels, visualize=True)


def execute_obrg_on_data(points: np.ndarray, labels: Optional[np.ndarray], visualize: bool = True):
    assert points.ndim == 2
    assert points.shape[1] == 9
    assert points.shape[0] > 0

    octree = segmentation.octree_based_region_growing(points,
                                                      initial_voxel_size=0.1,
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

    if labels is not None:
        cluster_label_map = octree.assign_labels_to_clusters(classes=segmentation.CLASSES, labels=labels)
        if visualize:
            pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[:, :3]))
            class_colors_np = np.array([a[1] for a in segmentation.CLASS_COLORS])
            colors = class_colors_np[cluster_label_map[octree.segment_index_per_point]]
            pcd.colors = open3d.utility.Vector3dVector(colors)
            open3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    print("Initializing...")
    execute()
