import sys

import open3d

import segmentation


def segment_with_hdbscan(pcd):
    sys.setrecursionlimit(15000)
    segmentation.hdbscan(pcd,
                         minimum_cluster_size=50,
                         minimum_samples=None,
                         cluster_selection_epsilon=0.0,
                         visualize=True)


def execute():
    # use_downsampled: bool = True
    # point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud_preprocessed.ply"
    # if not use_downsampled:
    #     point_cloud_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis-reconstruction\\data\\etvr\\enfsi-2023_reduced_cloud.pcd"
    # print("Loading point cloud...")
    # pcd: open3d.geometry.PointCloud = open3d.io.read_point_cloud(point_cloud_path)
    # print(f"Loaded point cloud with {len(pcd.points)} points.")
    # print("Clustering / segmenting using HDBScan...")
    # segment_with_hdbscan()

    s3dis_root = "E:\\etvr_datasets\\Stanford3dDataset_v1.2"
    segmentation.pointnet_train(s3dis_root, 13)

    return
    print("Clustering...")
    segmentation.octree_based_region_growing(pcd,
                                             initial_voxel_size=0.1,
                                             down_sample_voxel_size=None,

                                             # Subdivision parameters
                                             subdivision_residual_threshold=0.001,
                                             subdivision_full_threshold=4,
                                             subdivision_minimum_voxel_size=0.01,

                                             # Region growing parameters
                                             minimum_valid_segment_size=20,
                                             region_growing_residual_threshold=0.95,
                                             growing_normal_deviation_threshold_degrees=45,

                                             # Region refining / refinement parameter
                                             refining_normal_deviation_threshold_degrees=30,
                                             general_refinement_buffer_size=0.02,
                                             fast_refinement_planar_distance_threshold=0.01,
                                             fast_refinement_distance_threshold=0.05,
                                             fast_refinement_planar_amount_threshold=0.8)


if __name__ == '__main__':
    execute()

