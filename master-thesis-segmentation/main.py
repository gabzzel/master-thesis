import numpy as np
from open3d.cpu.pybind.geometry import KDTreeSearchParamHybrid

import segmentation
import open3d

if __name__ == '__main__':
    point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(point_cloud_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    #voxel_size = 0.05
    #print("Estimating normals...")
    #pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #print("Estimated normals")
    #pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    #print(f"Downsampled point cloud to {len(pcd.points)} points.")

    print("Clustering...")
    segmentation.octree_based_region_growing(pcd)

    cluster_per_point = segmentation.hdbscan(pcd)
    print("Clustering done.")
    unique_clusters = np.unique(cluster_per_point)
    rng = np.random.default_rng()
    colors_per_cluster = rng.random((len(unique_clusters), 3), dtype=np.float64)
    colors = colors_per_cluster[cluster_per_point]
    pcd.colors = open3d.utility.Vector3dVector(colors)
    open3d.visualization.draw_geometries([pcd])