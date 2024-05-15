import open3d
import numpy as np
import tqdm

from regionGrowingOctree.RegionGrowingOctree import RegionGrowingOctree


def visualize_segments_with_points(octree: RegionGrowingOctree):
    rng = np.random.default_rng()
    original_points = octree.points[:, :3]
    colors = rng.random(size=(len(np.unique(octree.segment_index_per_point)), 3))
    colors_per_point = colors[octree.segment_index_per_point]
    pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(original_points))
    pcd.colors = open3d.utility.Vector3dVector(colors_per_point)
    open3d.visualization.draw_geometries([pcd])


def visualize_segments_as_points(octree: RegionGrowingOctree, show_normals: bool = True):
    points = np.zeros(shape=(len(octree.segments), 3))
    normals = np.zeros(shape=(len(octree.segments), 3))
    rng = np.random.default_rng()
    colours = np.zeros(shape=(len(octree.segments), 3))

    for i, segment in enumerate(octree.segments):
        points[i] = segment.centroid
        normals[i] = segment.normal
        colours[i] = rng.random(size=(3, ))

    pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
    pcd.colors = open3d.utility.Vector3dVector(colours)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    open3d.visualization.draw_geometries([pcd], point_show_normal=show_normals)

