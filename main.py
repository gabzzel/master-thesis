import pclpy  # https://github.com/davidcaron/pclpy
from pclpy import pcl

import open3d
import numpy as np
import matplotlib.pyplot as plt
import time


def load_point_cloud(path, voxel_down_sample=True, voxel_down_sample_size=0.05, verbose=True):

    if verbose: print("Loading point cloud...")

    start_time = time.time()
    pcd = open3d.io.read_point_cloud(path, print_progress=True)
    end_time = time.time()

    if verbose: print("Loaded point cloud with " + format_number(len(pcd.points)) + " points in " + str(round(end_time - start_time, 2)) + " seconds.")

    if not voxel_down_sample: return pcd

    start_time = time.time()
    num_points_original = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_down_sample_size)
    end_time = time.time()

    if verbose: 
        elapsed_time = str(round(end_time - start_time, 2))
        num_pts_formatted = format_number(len(pcd.points))
        ratio = str(round(float(len(pcd.points)) / float(num_points_original) * 100))
        print("Downsampled point cloud to " + num_pts_formatted + " points (" + ratio + "% of original, voxel size " + 
            str(voxel_down_sample_size) + ") in " + elapsed_time + " seconds.")

    return pcd

def DBSCAN_clustering(point_cloud, eps=0.02, min_points=10, verbose=True, add_colors=True):
    start_time = time.time()
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=verbose))
    end_time = time.time()

    if verbose: 
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Clustered points using DBSCAN (ε={str(eps)} , min_points={str(min_points)}) in {elapsed_time} seconds.")

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
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud, open3d.utility.DoubleVector(radii))
    end_time = time.time()
    if verbose:
        num_triangles_formatted = format_number(len(mesh.triangles))
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh using BPA ({num_triangles_formatted} triangles, radii {str(radii)}) in {elapsed_time} seconds.")

    return mesh

def SPSR(point_cloud, octree_max_depth=8, density_quantile_threshold=0.1, verbose=True):
    start_time = time.time()

    # Densities is by how many vertices the other vertex is supported
    (mesh, densities) = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=octree_max_depth, width=0, scale=1.1, linear_fit=False, n_threads=-1)
    
    end_time = time.time()
    if verbose:
        num_triangles_formatted = format_number(len(mesh.triangles))
        elapsed_time = str(round(end_time - start_time, 2))
        print("Created mesh using SPSR (" + num_triangles_formatted + " triangles, at max octree depth " + str(octree_max_depth) + ") in " + elapsed_time + " seconds.")

    density_quantile_threshold = min(1.0, max(density_quantile_threshold, 0.0))
    if density_quantile_threshold <= 0.0: return mesh

    vertices_to_remove = densities < np.quantile(densities, density_quantile_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if verbose:
        num_removed_vertices = format_number(np.sum(vertices_to_remove))
        triangles_left = format_number(len(mesh.triangles))
        print(f"Removed {num_removed_vertices} vertices in {str(density_quantile_threshold)} density quantile, triangles remaining: {triangles_left}.")

    return mesh

def AlphaShapes(point_cloud, alpha=0.02, verbose=True):
    start_time = time.time()
    mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha)
    end_time = time.time()
    if verbose:
        num_triangles_formatted = format_number(len(mesh.triangles))
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Created mesh using Alpha Shapes ({num_triangles_formatted} triangles, α={str(alpha)}) in {elapsed_time} seconds.")

    return mesh

if __name__ == "__main__":

    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    voxel_size = 0.01
    pcd = load_point_cloud(point_cloud_path, voxel_down_sample=False, voxel_down_sample_size=voxel_size, verbose=True)

    start_time = time.time()
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    end_time = time.time()
    print("Estimated normals in " + str(round(end_time - start_time, 2)) + " seconds.")
    open3d.io.write_point_cloud("C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud_incl_open3D_normals.pcd")

    #radii = [voxel_size, voxel_size*2, voxel_size*3]
    #mesh = BPA(pcd, radii)
    #mesh = SPSR(pcd, octree_max_depth=9, density_quantile_threshold=0.1)
    #mesh = AlphaShapes(pcd, alpha=0.1)
    #open3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)

    #DBSCAN_clustering(pcd, eps=0.05, min_points=15, verbose=True)
    #open3d.visualization.draw_geometries([pcd])