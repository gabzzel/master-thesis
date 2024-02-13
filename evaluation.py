import time
import cProfile
import matplotlib.pyplot as plt
import numpy as np
import open3d
import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)
from open3d.geometry import TriangleMesh, PointCloud

import utils
import mesh_quality


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
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)

    # Compute Hausdorff and Chamfer distances
    hausdorff = round(pcu.hausdorff_distance(pcd_points, mesh_points), 4)
    chamfer = round(pcu.chamfer_distance(pcd_points, mesh_points), 4)
    print(f"Hausdorff={hausdorff}, Chamfer={chamfer}")

    distances_pts_to_mesh = utils.get_points_to_mesh_distances(pcd_points, mesh)
    distance_results = utils.get_stats(distances_pts_to_mesh, "Point-to-mesh distances", return_results=True)

    distances_normalized = distances_pts_to_mesh / distance_results[0]  # Index 0 is the max distance.
    colors = np.full(shape=(len(distances_normalized), 3), fill_value=0.0)
    colors[:, 0] = distances_normalized
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    plt.hist(distances_pts_to_mesh, histtype='step', log=True, bins=100)


def evaluate_mesh(mesh: TriangleMesh, aspect_ratios=None):
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_adjacency_list():
        mesh.compute_adjacency_list()

    # Expose everything we need.
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Edge length statistics
    edge_lengths = utils.get_mesh_edge_lengths(vertices, triangles)
    utils.get_stats(edge_lengths, name="Edge Lengths", print_results=True)

    # Aspect Ratio statistics
    if aspect_ratios is None:
        aspect_ratios = mesh_quality.aspect_ratios(vertices, triangles)
    utils.get_stats(aspect_ratios, name="Aspect Ratios", print_results=True)

    # cv is the index of the connected component of each vertex
    # nv is the number of vertices per component
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    cv, nv, cf, nf = pcu.connected_components(vertices, triangles)
    num_conn_comp = len(nf)  # Number of connected components (according to triangles)
    largest_component_ratio = round(np.max(nf) / np.sum(nf) * 100.0, 3)
    print(f"Connectivity: Connected Components={num_conn_comp}, largest component ratio={largest_component_ratio}")

    # Curvatures (Discrete)
    #curvatures = mesh_quality.discrete_curvature(vertices, vertex_normals, triangles, triangle_normals, sample_ratio=0.01, radius=0.1)
    #utils.get_stats(curvatures, "Discrete Curvature", print_results=True)

    # Normal Deviations
    deviations = mesh_quality.triangle_normal_deviations_adjacency(mesh.adjacency_list, triangles, triangle_normals)
    utils.get_stats(deviations, name="Normal Deviations", print_results=True)

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()
