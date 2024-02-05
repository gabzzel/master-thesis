import pclpy  # https://github.com/davidcaron/pclpy
from pclpy import pcl

import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)

import open3d
from open3d.geometry import TriangleMesh, KDTreeSearchParamHybrid, KDTreeSearchParamKNN, KDTreeSearchParamRadius
import numpy as np
import matplotlib.pyplot as plt
import time
import random


def load_point_cloud(path, voxel_down_sample=True, voxel_down_sample_size=0.05, verbose=True):
    if verbose:
        print("Loading point cloud...")

    start_time = time.time()
    pcd = open3d.io.read_point_cloud(path, print_progress=True)
    end_time = time.time()

    if verbose:
        num_pts = format_number(len(pcd.points))
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Loaded point cloud with {num_pts} points [{elapsed_time}s]")

    if not voxel_down_sample:
        return pcd

    num_points_original = len(pcd.points)
    npof = format_number(num_points_original)  # Number Points Original Formatted
    start_time = time.time()
    pcd = pcd.voxel_down_sample(voxel_size=voxel_down_sample_size)
    end_time = time.time()

    if verbose:
        elapsed = str(round(end_time - start_time, 2))  # The number of seconds elapsed during downsampling operation
        num_pts = format_number(len(pcd.points))
        ratio = str(round(float(len(pcd.points)) / float(num_points_original) * 100))
        print(f"Downsampled {npof} pts -> {num_pts} pts ({ratio}%) (voxel size {voxel_down_sample_size}) [{elapsed}s]")

    return pcd


def estimate_normals(point_cloud, max_nn=None, radius=None, orient=None, normalize=True, verbose=True):
    start_time = time.time()
    if verbose:
        print("Estimating normals...")

    params_str = "Invalid Parameters"
    max_nn_valid = max_nn is not None and isinstance(max_nn, int) and max_nn > 0
    radius_valid = radius is not None and isinstance(radius, float) and radius > 0.0

    if not max_nn_valid and not radius_valid:
        print(f"WARNING: Both max_nn ({max_nn}) and radius ({radius}) values are invalid. Using default max_nn=30.")
        print("If this is not desired behaviour, please check the entered values and re-run.")
        max_nn = 30
        max_nn_valid = True

    if max_nn_valid and radius_valid:
        params_str = f"Max NN={max_nn}, radius={radius}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    elif max_nn_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamKNN(max_nn))
    elif radius_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamRadius(radius))
    else:
        print("Point cloud normal estimation failed, parameters invalid.")
        return

    if normalize:
        point_cloud.normalize_normals()

    end_time = time.time()
    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Estimated normals ({params_str}) (Normalized={normalize}) [{elapsed_time}s]")

    if orient is None or not isinstance(orient, int) or orient <= 0:
        return

    if verbose:
        print("Orienting normals w.r.t. tangent plane...")
    start_time = time.time()
    point_cloud.orient_normals_consistent_tangent_plane(orient)
    end_time = time.time()
    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Oriented normals (KNN={orient}) [{elapsed_time}s]")


def dbscan_clustering(point_cloud, eps=0.02, min_points=10, verbose=True, add_colors=True):
    start_time = time.time()
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=verbose))
    end_time = time.time()

    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Clustered points using DBSCAN (ε={eps} , min_points={min_points}) [{elapsed_time}s]")

    colors = None
    if add_colors:
        max_label = labels.max()
        if verbose:
            print(f"point cloud has {max_label + 1} clusters")

        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        point_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
    return point_cloud, labels, colors


def format_number(number, digits=1):
    if number >= 1000000000:
        return str(round(number / 1000000000, digits)) + "B"

    if number >= 1000000:
        return str(round(number / 1000000, digits)) + "M"

    if number >= 1000:
        return str(round(number / 1000, digits)) + "K"

    return str(number)


def BPA(point_cloud, radii, verbose=True):
    start_time = time.time()
    mesh = TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, open3d.utility.DoubleVector(radii))
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number Triangles Formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: BPA ({ntf} triangles, radii {radii}) [{elapsed_time}s]")

    return mesh


def SPSR(point_cloud, octree_max_depth=8, density_quantile_threshold=0.1, verbose=True):
    start_time = time.time()

    # Densities is by how many vertices the other vertex is supported
    (mesh, densities) = TriangleMesh.create_from_point_cloud_poisson(point_cloud,
                                                                     depth=octree_max_depth,
                                                                     width=0,
                                                                     scale=1.1,
                                                                     linear_fit=False,
                                                                     n_threads=-1)

    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: SPSR ({ntf} tris, max octree depth {octree_max_depth}) [{elapsed_time}s]")

    density_quantile_threshold = min(1.0, max(density_quantile_threshold, 0.0))
    if density_quantile_threshold <= 0.0:
        return mesh

    vertices_to_remove = densities < np.quantile(densities, density_quantile_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if verbose:
        nrv = format_number(np.sum(vertices_to_remove))  # Number of Removed Vertices
        rem_tris = format_number(len(mesh.triangles))  # Remaining Triangles
        print(f"Removed {nrv} verts in {density_quantile_threshold} density quantile, tris remaining: {rem_tris}.")

    return mesh


def AlphaShapes(point_cloud, alpha=0.02, verbose=True):
    start_time = time.time()
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    end_time = time.time()
    if verbose:
        ntf = format_number(len(mesh.triangles))  # Number of triangles formatted
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh: Alpha Shapes ({ntf} tris, α={alpha}) [{elapsed_time}s]")

    return mesh


if __name__ == "__main__":

    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    voxel_size = 0.02
    pcd = load_point_cloud(point_cloud_path, voxel_down_sample=True, voxel_down_sample_size=voxel_size, verbose=True)
    pcd2 = load_point_cloud(point_cloud_path, voxel_down_sample=True, voxel_down_sample_size=voxel_size, verbose=True)
    angle = random.random() * 30
    rot_matrix = open3d.geometry.get_rotation_matrix_from_xyz(rotation=np.array([0.0, angle, 0.0], np.float64))
    pcd2.rotate(rot_matrix)

    # estimate_normals(pcd, max_nn=30, radius=0.4, orient=30)
    # open3d.visualization.draw_geometries([pcd], point_show_normal=True)

    pcd_points = np.asarray(pcd.points)
    pcd2_points = np.asarray(pcd2.points)

    h_dist, idx_a, idx_b = pcu.hausdorff_distance(pcd_points, pcd2_points, return_index=True)
    chamfer_dist = pcu.chamfer_distance(pcd_points, pcd2_points)
    print(f"Hausdorff distance={round(h_dist,3)}, Chamfer distance={round(chamfer_dist, 3)}")

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
