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
from utilities.evaluation_results import EvaluationResults
from utilities.enumerations import TriangleNormalDeviationMethod


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


def evaluate(mesh: open3d.geometry.TriangleMesh,
             original_point_cloud: open3d.geometry.PointCloud,
             config: RunConfiguration,results: EvaluationResults,
             precomputed_aspect_ratios=None,
             verbose: bool = True):
    """
    Evaluate a mesh on its quality and print the results.

    :param results: The EvaluationResults instance to store the results in.
    :param original_point_cloud: The original point cloud from which the mesh is constructed.
    :param verbose: Whether to print progress and results.
    :param config: The configuration that determines what will be evaluated
    :param mesh: The mesh to evaluate.
    :param precomputed_aspect_ratios: Precomputed aspect ratios, if available. None otherwise.
    :return: None
    """

    if config.mesh_quality_metrics is None:
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

    start_time = time.time()

    if MQM.EDGE_LENGTHS in config.mesh_quality_metrics:
        edge_lengths = mesh_utils.get_edge_lengths_flat(vertices, triangles)
        results.edge_lengths = edge_lengths
        utils.get_stats(edge_lengths, name="Edge Lengths", print_results=True)

    if MQM.TRIANGLE_ASPECT_RATIOS in config.mesh_quality_metrics:
        if precomputed_aspect_ratios is None:
            precomputed_aspect_ratios = mesh_utils.aspect_ratios(vertices, triangles)
        results.aspect_ratios = precomputed_aspect_ratios
        utils.get_stats(precomputed_aspect_ratios, name="Aspect Ratios", print_results=True)

    if MQM.CONNECTIVITY in config.mesh_quality_metrics:
        evaluate_connectivity(triangles, vertices, results=results)

    if MQM.DISCRETE_CURVATURE in config.mesh_quality_metrics:
        evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices, results=results)

    if MQM.TRIANGLE_NORMAL_DEVIATIONS in config.mesh_quality_metrics:
        evaluate_normal_deviations(config, mesh.adjacency_list, triangle_normals, triangles, results=results)

    hausdorff, chamfer, pc2md = mesh_quality.evaluate_point_cloud_mesh(original_point_cloud, mesh, config)

    if MQM.HAUSDORFF_DISTANCE in config.mesh_quality_metrics:
        results.hausdorff_distance = hausdorff
    if MQM.CHAMFER_DISTANCE in config.mesh_quality_metrics:
        results.chamfer_distance = chamfer
    if MQM.MESH_TO_CLOUD_DISTANCE in config.mesh_quality_metrics:
        results.point_cloud_to_mesh_distances = pc2md

    results.evaluation_time = time.time() - start_time

    # print(f"Principal Curvatures: Magnitudes Min={k1}, Max={k2}. Directions {d1} and {d2}")
    # plt.hist(aspect_ratios, histtype='step', log=True, bins=100, label="Aspect Ratios")
    # plt.show()


def evaluate_normal_deviations(config: RunConfiguration,
                               adjacency_list,
                               triangle_normals,
                               triangles,
                               results: EvaluationResults,
                               profile: bool = False):
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    if config.triangle_normal_deviation_method == TriangleNormalDeviationMethod.ADJANCENCY:
        deviations = mesh_quality.triangle_normal_deviations_adjacency(adjacency_list.copy(),
                                                                       triangles,
                                                                       triangle_normals,
                                                                       num_workers=config.processes,
                                                                       chunk_size=config.chunk_size)

    elif config.triangle_normal_deviation_method == TriangleNormalDeviationMethod.NAIVE:
        deviations = mesh_quality.triangle_normal_deviations_naive(triangles,
                                                                   triangle_normals,
                                                                   num_workers=config.processes,
                                                                   chunk_size=config.chunk_size)

    else:
        print(f"Triangle normal deviation method {config.triangle_normal_deviation_method} unknown.")
        results.normal_deviations = []
        return

    results.normal_deviations = deviations

    if profile:
        pr.disable()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()

    utils.get_stats(deviations, name="Normal Deviations", print_results=True)


def evaluate_discrete_curvatures(triangle_normals, triangles, vertex_normals, vertices, results: EvaluationResults):
    curvatures = mesh_quality.discrete_curvature(vertices,
                                                 vertex_normals,
                                                 triangles,
                                                 triangle_normals,
                                                 sample_ratio=0.01,
                                                 radius=0.1)
    results.discrete_curvatures = curvatures
    utils.get_stats(curvatures, "Discrete Curvature", print_results=True)
