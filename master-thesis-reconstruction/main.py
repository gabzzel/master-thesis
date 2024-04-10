import argparse
import sys
import time
from typing import Optional, Tuple
import pathlib
import os

import open3d

import evaluation
import utilities.enumerations
from utilities import io, mesh_cleaning, pcd_utils
import surface_reconstruction
from utilities.enumerations import SurfaceReconstructionMethod
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

    reused_point_clouds: Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud] = None

    for i in range(len(run_configs)):
        print(f"\n================== Starting Run {i + 1} (of {len(run_configs)}) ==================")
        prev_run_config = run_configs[i - 1] if i > 0 else None
        run_config = run_configs[i]
        run_config.name = str(time.time()) + "_run_" + str(i)

        run_result_path = results_path.joinpath(run_config.name)
        if not run_result_path.exists():
            os.makedirs(run_result_path)

        # If we have no previous run config, or we may not reuse the previous pointcloud, set this to None to force
        # recalculation
        may_reuse = prev_run_config is not None and prev_run_config.reuse_pointcloud
        if not may_reuse or not run_config.eligible_for_pointcloud_reuse(prev_run_config):
            reused_point_clouds = None

        reused_point_clouds = execute_run(run_config,
                                          reused_point_clouds=reused_point_clouds,
                                          results_path=run_result_path,
                                          verbose=verbose,
                                          draw=draw)


def execute_run(run_config: RunConfiguration,
                reused_point_clouds: Optional[Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]],
                results_path: pathlib.Path,
                verbose: bool = True,
                draw: bool = False) -> Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]:
    results = EvaluationResults(name="results")

    print("\n============= Step 1 : Loading & Preprocessing =============")
    if reused_point_clouds is not None:
        raw_pcd, pcd = reused_point_clouds
        print("Reused point clouds from previous run.")
    else:
        raw_pcd, pcd = load_point_cloud(run_config, results, verbose=verbose)

    if run_config.store_preprocessed_pointcloud:
        original_path = run_config.point_cloud_path
        preprocessed_pcd_file_name = original_path.stem + "_preprocessed" + original_path.suffix
        pcd_path = os.path.join(results_path, preprocessed_pcd_file_name)
        open3d.io.write_point_cloud(filename=str(pcd_path), pointcloud=pcd, print_progress=verbose)
        run_config.preprocessed_pointcloud_path = pcd_path
        print(f"Stored preprocessed point cloud at {pcd_path}")

    print("\n============= Step 2 : Surface Reconstruction =============")
    # Densities array is None if not applicable
    mesh, densities = surface_reconstruction.run(pcd=pcd, results=results, config=run_config, verbose=verbose)

    print("\n============= Step 3 : Cleaning =============")
    aspect_ratios = mesh_cleaning.run_mesh_cleaning(mesh, run_config, results, densities=densities, verbose=verbose)

    # If we have aspect ratios return from the mesh cleaning, we want the remaining after-cleaning aspect ratios
    if aspect_ratios is not None:
        aspect_ratios = aspect_ratios[1]

    print("\n============= Step 4 : Evaluation =============")
    # Raw point cloud is used here, since we want to evaluate against the original, not the preprocessed.
    evaluation.evaluate(mesh, raw_pcd, run_config, results, precomputed_aspect_ratios=aspect_ratios, verbose=verbose)

    print("\n============= Step 5 : Saving Results =============")
    start_time = time.time()
    results.save_to_file(run_config=run_config, folder_path=results_path, pcd=pcd, raw_pcd=raw_pcd)
    print(f"Saved results to {results_path}. [{round(time.time() - start_time, 3)}s]")

    if run_config.store_mesh:
        start_time = time.time()
        original_path = run_config.point_cloud_path
        mesh_name = original_path.stem + "_mesh.ply"
        mesh_path = os.path.join(results_path, mesh_name)
        open3d.io.write_triangle_mesh(filename=mesh_path, mesh=mesh)
        print(f"Saved mesh to {mesh_path}. [{round(time.time() - start_time, 3)}s]")

    if draw:
        open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    return raw_pcd, pcd


def load_point_cloud(config: RunConfiguration,
                     results: EvaluationResults,
                     verbose: bool = True) \
        -> Tuple[open3d.geometry.PointCloud, open3d.geometry.PointCloud]:
    start_time = time.time()

    if config.point_cloud_path.is_absolute():
        pcd_path = config.point_cloud_path
    else:
        pcd_path = pathlib.Path(sys.argv[0]).parent.joinpath("data", config.point_cloud_path)

    raw_pcd, pcd = pcd_utils.load_point_cloud(pcd_path,
                                              results=results,
                                              down_sample_method=config.down_sample_method,
                                              down_sample_param=config.down_sample_params,
                                              verbose=verbose)

    if config.surface_reconstruction_method != SurfaceReconstructionMethod.SCREENED_POISSON_SURFACE_RECONSTRUCTION and \
            config.surface_reconstruction_method != SurfaceReconstructionMethod.BALL_PIVOTING_ALGORITHM:
        print("Skipped normal estimation since surface reconstruction method is neither SPSR or BPA.")
    else:
        pcd_utils.estimate_normals(pcd,
                                   max_nn=config.normal_estimation_neighbours,
                                   radius=config.normal_estimation_radius,
                                   orient=config.orient_normals,
                                   normalize=not config.skip_normalizing_normals,
                                   verbose=verbose)

    results.loading_and_preprocessing_time = time.time() - start_time
    return raw_pcd, pcd


def run_with_test_cube():
    mesh = open3d.geometry.TriangleMesh()
    size = 10
    alpha = 1
    mesh: open3d.geometry.TriangleMesh = mesh.create_mobius()
    pcd = mesh.sample_points_poisson_disk(number_of_points=int(size / alpha * 6))
    mesh2 = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    open3d.visualization.draw_geometries([pcd, mesh2])


if __name__ == "__main__":
    #point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    #raw_pcd, pcd = pcd_utils.load_point_cloud(point_cloud_path,
    #                                          results=EvaluationResults("test"),
    #                                          down_sample_method=None,
    #                                          down_sample_param=None,
    #                                          verbose=True)
    #open3d.visualization.draw_geometries([pcd])

    main_script_path = pathlib.Path(sys.argv[0])

    if not main_script_path.exists():
        config_path = None
        print("Cannot find config.")

    if main_script_path.is_dir():
        config_path = main_script_path.joinpath("run_configs", "config.json")
        execute(str(config_path))

    elif main_script_path.is_file():
        config_path = main_script_path.parent.joinpath("run_configs", "config.json")
        execute(str(config_path))


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
