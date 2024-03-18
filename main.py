import argparse
import time
from typing import Optional, Tuple
import pathlib
import os

import open3d

import evaluation
from utilities import io, mesh_cleaning, pcd_utils
import surface_reconstruction
from utilities.run_configuration import RunConfiguration
from utilities.evaluation_results import EvaluationResults


def execute(config_file: Optional[str]):
    verbose = True
    draw = False

    if config_file is None:
        args: argparse.Namespace = io.parse_args()
        if args is None:
            print(f"Parsing arguments failed. Cancelling")
            time.sleep(5)
            return
        run_configs = io.get_run_configurations_from_args(args)

    elif pathlib.Path(config_file).is_file():
        run_configs, verbose, draw, copy = io.get_run_configurations_from_json(pathlib.Path(config_file))

    else:
        print(f"Invalid config file path: {config_file}.")
        return

    results_path = pathlib.Path(config_file).parent

    for i in range(len(run_configs)):
        print(f"Starting run {i + 1}/{len(run_configs)}")
        run_result_path = results_path.joinpath(f"result{i}")
        if not run_result_path.exists():
            os.makedirs(run_result_path)

        execute_run(run_configs[i], results_path=run_result_path, verbose=verbose, draw=draw)


def execute_run(run_config: RunConfiguration, results_path: pathlib.Path, verbose: bool = True, draw: bool = False):
    results = EvaluationResults(name="results")

    print("\n ============= Step 1 : Loading & Preprocessing =============")
    raw_pcd, pcd = load_point_cloud(run_config, results, verbose=verbose)

    if run_config.store_preprocessed_pointcloud:
        original_path = run_config.point_cloud_path
        preprocessed_pcd_file_name = original_path.stem + "_preprocessed" + original_path.suffix
        pcd_path = os.path.join(results_path, preprocessed_pcd_file_name)
        open3d.io.write_point_cloud(filename=str(pcd_path), pointcloud=pcd, print_progress=verbose)

    print("\n ============= Step 2 : Surface Reconstruction =============")
    mesh = surface_reconstruction.run(pcd=pcd, results=results, config=run_config, verbose=verbose)

    print("\n ============= Step 3 : Cleaning =============")
    aspect_ratios = mesh_cleaning.run_mesh_cleaning(mesh, run_config, results, verbose=verbose)

    # If we have aspect ratios return from the mesh cleaning, we want the remaining after-cleaning aspect ratios
    if aspect_ratios is not None:
        aspect_ratios = aspect_ratios[1]

    print("\n ============= Step 4 : Evaluation =============")
    # Raw point cloud is used here, since we want to evaluate against the original, not the preprocessed.
    evaluation.evaluate(mesh, raw_pcd, run_config, results, precomputed_aspect_ratios=aspect_ratios, verbose=verbose)

    results.save_to_file(results_path)

    if run_config.store_mesh:
        original_path = run_config.point_cloud_path
        mesh_name = original_path.stem + "_mesh.ply"
        mesh_path = os.path.join(results_path, mesh_name)
        open3d.io.write_triangle_mesh(filename=mesh_path, mesh=mesh)

    if draw:
        open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def load_point_cloud(config: RunConfiguration,
                     results: EvaluationResults,
                     verbose: bool = True) \
        -> Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]:
    start_time = time.time()
    raw_pcd, pcd = pcd_utils.load_point_cloud(config.point_cloud_path,
                                              results=results,
                                              down_sample_method=config.down_sample_method,
                                              down_sample_param=config.down_sample_params,
                                              verbose=verbose)

    pcd_utils.estimate_normals(pcd,
                               max_nn=config.normal_estimation_neighbours,
                               radius=config.normal_estimation_radius,
                               orient=config.orient_normals,
                               normalize=not config.skip_normalizing_normals,
                               verbose=verbose)

    results.loading_and_preprocessing_time = time.time() - start_time
    return raw_pcd, pcd


if __name__ == "__main__":
    config_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\run_configs\\config.json"
    execute(config_path)

    # point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"

    # pcd2_tree = open3d.geometry.KDTreeFlann(pcd2)
    # [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd2.points[idx_b], 50)
    # np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    # open3d.visualization.draw_geometries([pcd, pcd2])

    # write_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_open3D_normals.pcd"
    # open3d.io.write_point_cloud(write_path)

    # radii = [voxel_size, voxel_size*2, voxel_size*3]
    # mesh = BPA(pcd, radii)
    # mesh = SPSR(pcd, octree_max_depth=9, density_quantile_threshold=0.1)
    # mesh = AlphaShapes(pcd, alpha=0.1)
    # open3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)

    # DBSCAN_clustering(pcd, eps=0.05, min_points=15, verbose=True)
    # open3d.visualization.draw_geometries([pcd])
