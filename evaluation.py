import time

import matplotlib.pyplot as plt
import numpy as np
import open3d
import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)
from open3d.core import Tensor
from open3d.geometry import TriangleMesh, PointCloud
from open3d.t.geometry import RaycastingScene

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
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Expose everything we need.
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Edge length statistics
    edge_lengths = utils.get_mesh_edge_lengths(vertices, triangles)
    utils.get_stats(edge_lengths, name="Edge Lengths", print_only=True)

    # Aspect Ratio statistics
    if aspect_ratios is None:
        aspect_ratios = utils.get_mesh_aspect_ratios(vertices, triangles)
    utils.get_stats(aspect_ratios, name="Aspect Ratios", print_only=True)

    # cv is the index of the connected component of each vertex
    # nv is the number of vertices per component
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    cv, nv, cf, nf = pcu.connected_components(vertices, triangles)
    num_conn_comp = len(nf)  # Number of connected components (according to triangles)
    largest_component_ratio = round(np.max(nf) / np.sum(nf) * 100.0, 3)
    print(f"Connectivity: Connected Components={num_conn_comp}, largest component ration={largest_component_ratio}")

    curvatures = utils.get_mesh_discrete_curvature(vertices, vertex_normals, triangles, triangle_normals, sample_ratio=0.01, radius=0.1)
    utils.get_stats(curvatures, "Curvature", print_only=True)

    deviations = utils.get_mesh_triangle_normal_deviations(triangles, triangle_normals)
    utils.get_stats(deviations, name="Normal Deviations")

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()
