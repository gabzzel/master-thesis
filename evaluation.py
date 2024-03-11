import time
import cProfile
import pstats

import numpy as np
import open3d

from utilities import mesh_utils, utils
import point_cloud_utils  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)

import mesh_quality
from mesh_quality import evaluate_connectivity
from utilities.enumerations import MeshEvaluationMetric as MQM
from utilities.run_configuration import RunConfiguration


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


def evaluate_mesh(mesh: open3d.geometry.TriangleMesh,
                  config: RunConfiguration,
                  precomputed_aspect_ratios=None,
                  verbose: bool = True):
    """
    Evaluate a mesh on its quality and print the results.

    :param verbose: Whether to print progress and results.
    :param config: The configuration that determines what will be evaluated
    :param mesh: The mesh to evaluate.
    :param precomputed_aspect_ratios: Precomputed aspect ratios, if available. None otherwise.
    :return: None
    """

    if config.mesh_evaluation_metrics is None:
        if verbose:
            print(f"Evaluating mesh skipped. No mesh evaluation metrics specified in run configuration.")
        return

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

    if MQM.EDGE_LENGTHS in config.mesh_evaluation_metrics:
        edge_lengths = mesh_utils.get_edge_lengths_flat(vertices, triangles)
        utils.get_stats(edge_lengths, name="Edge Lengths", print_results=True)

    if MQM.TRIANGLE_ASPECT_RATIOS in config.mesh_evaluation_metrics:
        if precomputed_aspect_ratios is None:
            precomputed_aspect_ratios = mesh_utils.aspect_ratios(vertices, triangles)
        utils.get_stats(precomputed_aspect_ratios, name="Aspect Ratios", print_results=True)

    if MQM.CONNECTIVITY in config.mesh_evaluation_metrics:
        evaluate_connectivity(triangles, vertices)

    if MQM.DISCRETE_CURVATURE in config.mesh_evaluation_metrics:
        evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices)

    if MQM.TRIANGLE_NORMAL_DEVIATIONS in config.mesh_evaluation_metrics:
        evaluate_normal_deviations(mesh.adjacency_list, triangle_normals, triangles)

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()


def evaluate_normal_deviations(adjacency_list, triangle_normals, triangles, profile: bool = False):
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    deviations = mesh_quality.triangle_normal_deviations_adjacency(adjacency_list.copy(),
                                                                   triangles,
                                                                   triangle_normals)

    if profile:
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()

    utils.get_stats(deviations, name="Normal Deviations", print_results=True)


def evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices):
    curvatures = mesh_quality.discrete_curvature(vertices,
                                                 vertex_normals,
                                                 triangles,
                                                 triangle_normals,
                                                 sample_ratio=0.01,
                                                 radius=0.1)
    utils.get_stats(curvatures, "Discrete Curvature", print_results=True)
