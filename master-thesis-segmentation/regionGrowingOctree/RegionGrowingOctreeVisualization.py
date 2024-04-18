import open3d
import numpy as np
import tqdm

from regionGrowingOctree.RegionGrowingOctree import RegionGrowingOctree


def visualize_segments_with_points(octree: RegionGrowingOctree):
    rng = np.random.default_rng()
    original_points = np.asarray(octree.origin_point_cloud.points)
    points = []
    colors = []
    for segment in octree.segments:
        current_segment_color = rng.random(size=(3,))
        for node in segment.nodes:
            for point_index in node.vertex_indices:
                points.append(original_points[point_index])
                colors.append(current_segment_color)

    pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
    pcd.colors = open3d.utility.Vector3dVector(colors)
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

