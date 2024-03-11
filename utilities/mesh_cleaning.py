import time
from typing import Union, Optional, Tuple
import numpy as np
import open3d

from utilities import mesh_utils, run_configuration
from utilities.enumerations import MeshCleaningMethod


def get_cleaning_type(cleaning_type_text: str) -> Optional[MeshCleaningMethod]:
    t = cleaning_type_text.lower().strip()
    t = t.replace(" ", "_")

    if t == "s" or t == "simple":
        return MeshCleaningMethod.SIMPLE
    elif "edge" in t or t == "el":
        return MeshCleaningMethod.EDGE_LENGTHS
    elif "aspect" in t or t == "ar":
        return MeshCleaningMethod.ASPECT_RATIOS
    elif t == "all":
        return MeshCleaningMethod.ALL

    return None


def run_mesh_cleaning(mesh: open3d.geometry.TriangleMesh,
                      config: run_configuration.RunConfiguration,
                      verbose: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not config.mesh_cleaning_methods:
        return None

    if MeshCleaningMethod.SIMPLE in config.mesh_cleaning_methods:
        clean_mesh_simple(mesh=mesh, verbose=verbose)

    if MeshCleaningMethod.EDGE_LENGTHS in config.mesh_cleaning_methods:
        clean_mesh_metric(mesh=mesh,
                          metric=MeshCleaningMethod.EDGE_LENGTHS,
                          quantile=config.edge_length_cleaning_portion,
                          verbose=True)

    ar = ar_clean = None
    if MeshCleaningMethod.ASPECT_RATIOS in config.mesh_cleaning_methods:
        ar, ar_clean = clean_mesh_metric(mesh=mesh,
                                         metric=MeshCleaningMethod.ASPECT_RATIOS,
                                         quantile=config.aspect_ratio_cleaning_portion,
                                         verbose=verbose)

    return ar, ar_clean


def clean_mesh_simple(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh], verbose: bool = True):
    """
    Clean a mesh using simple mesh cleaning actions:
    - Remove duplicated vertices
    - Remove unreferenced vertices (i.e. vertices not in any triangle/tetra)
    - Remove duplicated triangles/tetras
    - Remove degenerate triangles/tetras (i.e. triangles/tetras that reference the same vertex multiple times)

    These actions are executed in the order displayed above.

    :param mesh: The mesh to clean.
    :param verbose: Whether to print the results.
    :return: Nothing.
    """
    is_triangle_mesh = mesh_utils.check_mesh_type(mesh=mesh)

    if verbose:
        print(f"Cleaning mesh (Simple)...")

    start_time = time.time()
    nvo, nto = mesh_utils.get_mesh_verts_and_tris(mesh=mesh)

    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if is_triangle_mesh:
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
    else:
        mesh.remove_duplicated_tetras()
        mesh.remove_degenerate_tetras()

    nvc, ntc = mesh_utils.get_mesh_verts_and_tris(mesh=mesh)
    end_time = time.time()
    if verbose:
        elapsed = round(end_time - start_time, 3)
        form = "tris" if is_triangle_mesh else "tetras"
        print(f"Cleaned mesh (simple) ({nvo} -> {nvc} verts, {nto} -> {ntc} {form}) [{elapsed}s]")


def clean_mesh_metric(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh],
                      metric: MeshCleaningMethod = MeshCleaningMethod.ASPECT_RATIOS,
                      quantile: float = 0.95,
                      absolute: float = 1000,
                      verbose: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clean a mesh using threshold on a certain metric.

    :param mesh: The mesh to clean.
    :param metric: Which metric to clean the mesh up with. Can be either "aspect_ratio" / "ar" or "edge_length" / "el"
    :param quantile: The quantile to calculate the threshold on. The lower value of this and the absolute value is \
     used. Set to 0 to disable.
    :param absolute: The absolute threshold. The lower value of this and the quantile value is used. Set to 0 to disable.
    :param verbose: Whether to print the progress and result.
    :return: A tuple of numpy arrays containing the metric calculated for all vertices, edges or triangles at index 0 \
     and the cleaned subset/subarray at index 1. Returns None if any error occurs.
    """

    is_triangle_mesh = mesh_utils.check_mesh_type(mesh=mesh)
    if not is_triangle_mesh:
        print(f"Cannot clean a mesh that is not an instance of {type(open3d.geometry.TriangleMesh)}. Returning")
        return None

    if quantile <= 0.0 and absolute <= 0.0:
        print("Both quantile and absolute value are 0. Returning None.")
        return None

    if metric is not MeshCleaningMethod.EDGE_LENGTHS or metric is not MeshCleaningMethod.ASPECT_RATIOS:
        print(f"Invalid specified metric {metric}. Must be one either {MeshCleaningMethod.ASPECT_RATIOS}. "
              f"or {MeshCleaningMethod.EDGE_LENGTHS}. Defaulting to aspect ratios.")
        return None

    if verbose:
        print(f"Cleaning mesh ({metric})...  Thresholds: Quantile={quantile}, Absolute={absolute})")

    start_time = time.time()
    nvo, nto = mesh_utils.get_mesh_verts_and_tris(mesh=mesh)
    quantile = min(1.0, max(quantile, 0.0))

    metric_all = None
    metric_cleaned = None
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    edge_lengths = mesh_utils.get_edge_lengths_per_triangle(vertices=vertices, triangles=triangles)

    if metric == MeshCleaningMethod.ASPECT_RATIOS:
        metric_all = mesh_utils.aspect_ratios_edge_lengths(edge_lengths=edge_lengths)
    elif metric == MeshCleaningMethod.EDGE_LENGTHS:
        metric_all = edge_lengths

    if quantile > 0 and absolute > 0:
        threshold = min(np.quantile(metric_all.flatten(), quantile), absolute)
    elif quantile > 0:
        threshold = np.quantile(metric_all.flatten(), quantile)
    else:
        threshold = absolute

    print(f"Actual threshold: {threshold}")

    if metric == MeshCleaningMethod.ASPECT_RATIOS:
        mesh.remove_triangles_by_mask(metric_all > threshold)
        metric_cleaned = metric_all[metric_all <= threshold]
    elif metric == MeshCleaningMethod.EDGE_LENGTHS:
        mesh.remove_triangles_by_mask(np.any(a=metric_all > threshold, axis=1))
        metric_cleaned = metric_all[np.any(a=metric_all <= threshold, axis=1)]

    mesh.remove_unreferenced_vertices()

    if verbose:
        print_cleaning_result(mesh=mesh, start_time=start_time, vertices_before=nvo, simplices_before=nto)

    return metric_all, metric_cleaned


def print_cleaning_result(mesh: Union[open3d.geometry.TriangleMesh, open3d.geometry.TetraMesh],
                          start_time: float,
                          vertices_before: str,
                          simplices_before: str):
    is_triangle_mesh = mesh_utils.check_mesh_type(mesh=mesh)

    nvc, ntc = mesh_utils.get_mesh_verts_and_tris(mesh=mesh)
    end_time = time.time()
    elapsed = round(end_time - start_time, 3)
    form = "tris" if is_triangle_mesh else "tetras"
    print(f"Cleaned mesh ({vertices_before} -> {nvc} verts, {simplices_before} -> {ntc} {form}) [{elapsed}s]")
