import numpy as np
import open3d

import evaluation
import surface_reconstruction
import utils
from basic_point_cloud_ops import load_point_cloud, estimate_normals

if __name__ == "__main__":

    # point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\dummy\\stanford-dragon\\dragon_recon\\dragon_vrip.ply"
    voxel_size = 0.05
    prefer_triangle_mesh = True
    pcd = load_point_cloud(point_cloud_path, down_sample_method=None, down_sample_param=0.01, verbose=True)
    estimate_normals(pcd, max_nn=30, radius=0.4, orient=None, normalize=True)
    mesh = surface_reconstruction.Delaunay(pcd, as_tris=prefer_triangle_mesh)
    # mesh = surface_reconstruction.SPSR(pcd, octree_max_depth=8)

    # Clean the mesh and return the aspect ratios (if calculated, which is done when aspect ratio threshold are > 0)
    _, aspect_ratios_clean = utils.clean_mesh(mesh)
    #if prefer_triangle_mesh:
        #evaluation.evaluate_mesh(mesh, aspect_ratios=aspect_ratios_clean)
    # evaluation.evaluate_point_cloud_mesh(pcd, mesh)
    open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

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
