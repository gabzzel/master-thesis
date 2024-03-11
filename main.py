import argparse
import time

import open3d

import evaluation
from utilities import io, mesh_cleaning, pcd_utils
import surface_reconstruction
from utilities.run_configuration import RunConfiguration


def execute():
    args: argparse.Namespace = io.parse_args()

    if args is None:
        print(f"Parsing arguments failed. Cancelling")
        time.sleep(5)
        return

    run_configs = io.get_run_configurations_from_args(args)

    for i in range(len(run_configs)):
        print(f"Starting run {i + 1}/{len(run_configs)}")
        execute_run(run_configs[i], args)


def execute_run(run_config: RunConfiguration, args: argparse.Namespace):
    verbose = args.verbose
    pcd = load_point_cloud(config=run_config, verbose=verbose)
    mesh = surface_reconstruction.run(pcd=pcd, config=run_config, verbose=verbose)

    aspect_ratios = mesh_cleaning.run_mesh_cleaning(mesh=mesh, config=run_config, verbose=verbose)

    # If we have aspect ratios return from the mesh cleaning, we want the remaining after-cleaning aspect ratios
    if aspect_ratios is not None:
        aspect_ratios = aspect_ratios[1]

    evaluation.evaluate_mesh(mesh=mesh, config=run_config, precomputed_aspect_ratios=aspect_ratios, verbose=verbose)

    if args.draw:
        open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def load_point_cloud(config: RunConfiguration, verbose: bool = True) -> open3d.geometry.PointCloud:
    pcd = pcd_utils.load_point_cloud(config.point_cloud_path,
                                     down_sample_method=config.down_sample_method,
                                     down_sample_param=config.down_sample_params,
                                     verbose=verbose)

    pcd_utils.estimate_normals(pcd,
                               max_nn=config.normal_estimation_neighbours,
                               radius=config.normal_estimation_radius,
                               orient=config.orient_normals,
                               normalize=not config.skip_normalizing_normals,
                               verbose=verbose)
    return pcd


if __name__ == "__main__":
    execute()

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
