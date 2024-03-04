import time
import cProfile
import pstats
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import open3d
import point_cloud_utils # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)

import utils
import mesh_quality


def evaluate_point_clouds(point_cloud_a: open3d.geometry.PointCloud, point_cloud_b: open3d.geometry.PointCloud):
    print("Evaluating...")
    pcd_a_points = np.asarray(point_cloud_a.points)
    pcd_b_points = np.asarray(point_cloud_b.points)
    start_time = time.time()
    hausdorff = point_cloud_utils.hausdorff_distance(pcd_a_points, pcd_b_points)
    chamfer = point_cloud_utils.chamfer_distance(pcd_a_points, pcd_b_points)
    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer} [{elapsed_time}s]")


def evaluate_point_cloud_mesh(point_cloud: open3d.geometry.PointCloud,
                              mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh]):
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)

    # Compute Hausdorff and Chamfer distances
    hausdorff = round(point_cloud_utils.hausdorff_distance(pcd_points, mesh_points), 4)
    chamfer = round(point_cloud_utils.chamfer_distance(pcd_points, mesh_points), 4)
    print(f"Hausdorff={hausdorff}, Chamfer={chamfer}")

    distances_pts_to_mesh = utils.get_points_to_mesh_distances(pcd_points, mesh)
    distance_results = utils.get_stats(distances_pts_to_mesh, "Point-to-mesh distances", return_results=True)

    distances_normalized = distances_pts_to_mesh / distance_results[0]  # Index 0 is the max distance.
    colors = np.full(shape=(len(distances_normalized), 3), fill_value=0.0)
    colors[:, 0] = distances_normalized
    point_cloud.colors = open3d.utility.Vector3dVector(colors)
    plt.hist(distances_pts_to_mesh, histtype='step', log=True, bins=100)


def evaluate_mesh(mesh: open3d.geometry.TriangleMesh, aspect_ratios=None, el=True, ar=True, co=True, nd=True, dc=True):
    """
    Evaluate a mesh on its quality and print the results.

    :param mesh: The mesh to evaluate.
    :param aspect_ratios: Precomputed aspect ratios, if available. None otherwise.
    :param el: Whether to evaluate the edge lengths of the mesh.
    :param ar: Whether to evaluate the aspect ratios of the mesh.
    :param co: Whether to evaluate the connectivity of the mesh.
    :param nd: Whether to evaluate the triangle normal deviations of the mesh.
    :param dc: Whether to evaluate the discrete curvature of the mesh.
    :return: None
    """

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_adjacency_list():
        mesh.compute_adjacency_list()

    mesh.normalize_normals()

    # Expose everything we need.
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    # Edge length statistics
    if el:
        edge_lengths = utils.get_edge_lengths_flat(vertices, triangles)
        utils.get_stats(edge_lengths, name="Edge Lengths", print_results=True)

    if ar:
        # Aspect Ratio statistics
        if aspect_ratios is None:
            aspect_ratios = utils.aspect_ratios(vertices, triangles)
        utils.get_stats(aspect_ratios, name="Aspect Ratios", print_results=True)

    if co:
        evaluate_connectivity(triangles, vertices)

    if dc:
        evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices)

    if nd:
        evaluate_normal_deviations(mesh.adjacency_list, triangle_normals, triangles)

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()


def evaluate_normal_deviations(adjacency_list, triangle_normals, triangles):
    pr = cProfile.Profile()
    pr.enable()
    deviations = mesh_quality.triangle_normal_deviations_adjacency(adjacency_list.copy(),
                                                                   triangles,
                                                                   triangle_normals)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    utils.get_stats(deviations, name="Normal Deviations", print_results=True)


def evaluate_connectivity(triangles, vertices):
    # cv is the index of the connected component of each vertex
    # nv is the number of vertices per component
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    cv, nv, cf, nf = point_cloud_utils.connected_components(vertices, triangles)
    # If nf is a 0-dimensional array or has length 1, there is only a single component
    if nf.ndim == 0 or len(nf) == 1:
        num_conn_comp = 1
        largest_component_ratio = 1.0
    else:
        num_conn_comp = len(nf)  # Number of connected components (according to triangles)
        largest_component_ratio = round(np.max(nf) / np.sum(nf), 3)
    print(f"Connectivity: Connected Components={num_conn_comp}, largest component ratio={largest_component_ratio}")


def evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices):
    curvatures = mesh_quality.discrete_curvature(vertices,
                                                 vertex_normals,
                                                 triangles,
                                                 triangle_normals,
                                                 sample_ratio=0.01,
                                                 radius=0.1)
    utils.get_stats(curvatures, "Discrete Curvature", print_results=True)
