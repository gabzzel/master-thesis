import open3d
from open3d.geometry import TriangleMesh, PointCloud
from open3d.t.geometry import RaycastingScene
from open3d.core import Tensor
# import pclpy  # https://github.com/davidcaron/pclpy
# from pclpy import pcl
import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)
import numpy as np
import time
import matplotlib.pyplot as plt
import utils


def evaluate_point_clouds(point_cloud_a: PointCloud, point_cloud_b: PointCloud):
    print("Evaluating...")
    pcd_a_points = np.asarray(point_cloud_a.points)
    pcd_b_points = np.asarray(point_cloud_b.points)
    start_time = time.time()
    hausdorff = pcu.hausdorff_distance(pcd_a_points, pcd_b_points)
    chamfer = pcu.chamfer_distance(pcd_a_points, pcd_b_points)
    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer} [{elapsed_time}s]")


def evaluate_point_cloud_mesh(point_cloud: PointCloud, mesh: TriangleMesh):
    # Computer Chamfer and Hausdorff distances.
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)
    start_time = time.time()
    hausdorff = round(pcu.hausdorff_distance(pcd_points, mesh_points), 4)
    chamfer = round(pcu.chamfer_distance(pcd_points, mesh_points), 4)

    # Compute point cloud to mesh surfaces distances.
    rcs = RaycastingScene()
    tensor_mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rcs.add_triangles(tensor_mesh)
    pts = Tensor(np.array(pcd_points).astype(dtype=np.float32))
    distances = rcs.compute_distance(query_points=pts).numpy().astype(dtype=np.float32)
    max_distance = np.max(distances)
    distances_norm = distances / max_distance
    colors = np.asarray(point_cloud.colors)
    colors[:, 0] = distances_norm
    colors[:, 1] = np.full_like(distances_norm, 0.9)
    colors[:, 2] = np.full_like(distances_norm, 0.9)

    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer}, Max Dist={max_distance} [{elapsed_time}s]")

    plt.hist(distances, histtype='step', log=True, bins=100)


def evaluate_mesh(mesh: TriangleMesh, aspect_ratios=None):
    edge_lengths = utils.get_mesh_edge_lengths(mesh)
    utils.get_stats(edge_lengths, name="Edge Lengths", print_only=True)

    # If we do not get the aspect ratios, calculate them ourselves.
    if aspect_ratios is None:
        aspect_ratios = utils.get_mesh_aspect_ratios(mesh)

    utils.get_stats(aspect_ratios, name="Aspect Ratios", print_only=True)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # cv is the index of the connected component of each vertex
    # nv is the number of vertices per component
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    cv, nv, cf, nf = pcu.connected_components(vertices, triangles)
    num_conn_comp = len(nf)  # Number of connected components (according to triangles)
    largest_component_ratio = round(np.max(nf) / np.sum(nf) * 100.0, 3)
    print(f"Connectivity: Connected Components={num_conn_comp}, largest component ration={largest_component_ratio}")

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    triangle_normals = np.asarray(mesh.triangle_normals)

    triangles_sorted = np.sort(triangles, axis=1)
    triangle_count = len(triangles_sorted)
    triangle_indices = np.arange(triangle_count)

    # Add the indices to the sorted triangles
    triangles_sorted = np.hstack((triangle_indices[:, np.newaxis], triangles_sorted))

    print(f"Calculating normal deviations")
    dots = []

    for i in range(triangle_count):
        current_triangle = triangles_sorted[i]
        to_compare_against = triangles_sorted[i+1:triangle_count]
        compare = triangles_sorted[i] == to_compare_against
        sums = np.sum(compare, axis=1)
        adjacent = to_compare_against[sums == 2]
        if len(adjacent) == 0:
            continue

        current_triangle_normal = triangle_normals[current_triangle[0]]
        adjacent_normals = triangle_normals[adjacent[:, 0]]
        dots += np.clip(np.dot(adjacent_normals, current_triangle_normal), -1.0, 1.0).tolist()


    angles = np.degrees(np.arccos(dots))
    utils.get_stats(angles, name="Normal Deviations")

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()
