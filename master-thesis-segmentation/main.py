import numpy as np
from open3d.cpu.pybind.geometry import KDTreeSearchParamHybrid

import segmentation
import open3d

if __name__ == '__main__':
    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud_preprocessed.ply"
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(point_cloud_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    #voxel_size = 0.05
    #print("Estimating normals...")
    #pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #print("Estimated normals")
    #pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    #print(f"Downsampled point cloud to {len(pcd.points)} points.")

    print("Clustering...")
    segmentation.octree_based_region_growing(pcd,
                                             initial_voxel_size=0.1,
                                             down_sample_voxel_size=None,

                                             # Subdivision parameters
                                             subdivision_residual_threshold=0.001,
                                             subdivision_full_threshold=10,
                                             subdivision_minimum_voxel_size=0.01,

                                             # Region growing parameters
                                             minimum_valid_segment_size=3,
                                             region_growing_residual_threshold=0.99,
                                             growing_normal_deviation_threshold_degrees=30,

                                             # Region refining / refinement parameter
                                             refining_normal_deviation_threshold_degrees=45,
                                             general_refinement_buffer_size=0.02,
                                             fast_refinement_planar_distance_threshold=0.01,
                                             fast_refinement_distance_threshold=0.02,
                                             fast_refinement_planar_amount_threshold=0.9)

    #cluster_per_point = segmentation.hdbscan(pcd)
    #print("Clustering done.")
    #unique_clusters = np.unique(cluster_per_point)
    #rng = np.random.default_rng()
    #colors_per_cluster = rng.random((len(unique_clusters), 3), dtype=np.float64)
    #colors = colors_per_cluster[cluster_per_point]
    #pcd.colors = open3d.utility.Vector3dVector(colors)
    #open3d.visualization.draw_geometries([pcd])