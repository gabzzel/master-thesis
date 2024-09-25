import argparse
import json
import math
from pathlib import Path
from typing import Optional, List, Tuple, Union, Set

import numpy as np

import mesh_quality
import surface_reconstruction
from utilities import mesh_cleaning, pcd_utils
from utilities.enumerations import DownSampleMethod as DSM
from utilities.enumerations import MeshCleaningMethod as MCM
from utilities.enumerations import MeshEvaluationMetric as MEM
from utilities.enumerations import SurfaceReconstructionMethod as SRM
from utilities.enumerations import SurfaceReconstructionParameters as SRP
from utilities.enumerations import TriangleNormalDeviationMethod as TNDM
from utilities.run_configuration import RunConfiguration


def parse_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser()

    # The paths to the point cloud, the segments and classifications
    parser.add_argument('point_cloud_path', type=Path, help="The path to the point cloud file.")

    parser.add_argument('result_path', type=Path, help="The path where the results will be stored.")

    parser.add_argument('-segments_path', type=Path, default=None, help="The path to the segments file.")
    parser.add_argument('-classifications_path', type=Path, default=None, help="The path to the classification file.")

    parser.add_argument('-down_sample_method', '-dsm',
                        type=str, default=None, action='store',
                        help="The down-sampling method(s) to use. ",
                        choices=[None, 'none', 'voxel', 'random'])

    parser.add_argument('-down_sample_params', '-dsp',
                        type=float, default=0.01, action='store',
                        help="When doing voxel down-sampling, this parameter will be the voxel size. When random "
                             "down-sampling, this parameter will be the amount of sample ratio.")

    parser.add_argument('-normal_estimation_neighbours', '-nen',
                        type=int, default=10, action='store',
                        help="The number of neighbours to use during normal estimation. Clipped between 0 and 1000.")

    parser.add_argument('-normal_estimation_radius', '-ner',
                        type=float, default=0.5, action='store',
                        help="The radius around the point to use during normal estimation. Clipped between"
                             " 0 and 1000")

    parser.add_argument('-skip_normalizing_normals', '-skip_nornor', '-snn',
                        action='store_true',
                        help="Whether to skip normalizing the normals after estimation. Not recommended.")

    parser.add_argument('-orient_normals', '-on', '-ornor', type=int, action='store', default=0,
                        help="Whether to orient the normals after estimation. Expensive, but might lead to better "
                             "results. Clipped between 0 and 1000.")

    parser.add_argument('-verbose', '-v', action='store_true',
                        help='Whether to print progress and results in the console.')

    parser.add_argument('-surface_reconstruction_algorithm', '-sra', '-sr_alg',
                        type=str, action='store', default='spsr',
                        help="The surface reconstruction algorithm to run. Ball Pivoting = 'bpa', 'b' or 'B' \n"
                             "Screened Poisson = 'spsr', 'SPSR', 'sp', 'SP', 's' or 'S'",
                        choices=['ball_pivoting', 'bpa', 'bp', 'b', 'poisson', 'spsr', 'P', 'p',
                                 'alpha_shapes', 'alpha', 'a', 'A'])

    parser.add_argument('-alpha', '-a', type=float, action='store', default=0.02,
                        help="The alpha value(s) to run Alpha Shapes with.")

    parser.add_argument('-ball_pivoting_radii', '-bpa_radii', '-bpar',
                        default=[0.1], action='append',
                        help="The ball radii to for ball pivoting.")

    parser.add_argument('-poisson_density_quantile', '-pdq',
                        type=float, action='store', default=0.1,
                        help="The lower quantile/portion of the points sorted by density / support is removed."
                             "Used to clean mesh after reconstruction using Screened Poisson Surface Reconstruction.")

    parser.add_argument('-poisson_octree_max_depth', '-pomd',
                        type=int, action='store', default=0,
                        help="The maximum depth of the octree when using Poisson Surface Reconstruction. Lower values"
                             " correspond to lower resoluation and thus less detailed meshes, but reduces "
                             "computing time. Must be between 3 and 100 (both inclusive). Recommended is between 8"
                             " and 15 depending on the size of the scene.")

    parser.add_argument('-poisson_octree_min_width', '-pomw',
                        type=float, action='store', default=0,
                        help="The minimum size of the octree cells when using Poisson Surface Reconstruction. Overrides"
                             " the max depth if used.")

    parser.add_argument('-mesh_clean_methods', '-mcm',
                        required=False, action='append',
                        choices=['simple', 's', 'edge_length', 'aspect_ratio', 'el', 'ar'])

    parser.add_argument('-edge_length_clean_portion', '-elcp',
                        required=False, action='store', type=float, default=0.95,
                        help="The portion of the edge lengths to keep when cleaning. Must be between 0 and 1."
                             "Set to 1 to not remove any edges.")

    parser.add_argument('-aspect_ratio_clean_portion', '-arcp',
                        required=False, action='store', type=float, default=0.95,
                        help="The portion of triangles, sorted by aspect ratio, to keep when cleaning. Must be "
                             "between 0 and 1. Set to 1 to not remove any triangles.")

    parser.add_argument('-mesh_quality_metrics', '-mqm',
                        required=False, action='append',
                        help="Which metrics to use during evaluation of the reconstructed surface /  mesh.",
                        choices=['all', 'edge_lengths', 'el', 'aspect_ratios', 'ar', 'connectivity', 'co',
                                 'discrete_curvature', 'dc', 'normal_deviations', 'nd'])

    parser.add_argument('-mesh_to_cloud_metrics', '-m2cm',
                        required=False, action='append',
                        help="Which metrics to use to evaluate to surface reconstruction to the point cloud.",
                        choices=['all', 'chamfer', 'c', 'hausdorff', 'h', 'distances', 'distance', 'd'])

    parser.add_argument('-result_path', action='store', type=Path, required=False,
                        help="Path to the folder where to write the evaluation results.")

    parser.add_argument('-store_mesh', '-save_mesh', action='store_true')

    parser.add_argument('-mesh_output_format', action='store', type=str, default='.gltf',
                        choices=['.gltf', '.obj', '.ply'])

    parser.add_argument('-processes', action='store', type=int, default=-1,
                        help="The number of processes to use. Set to -1 to automatically determine. Clipped between"
                             " -1 and 100 (both inclusive). Set to 0 or 1 to run single-threaded.")

    parser.add_argument('-draw',
                        action='store_true',
                        help="Whether to draw the mesh after reconstruction")

    args = parser.parse_args()

    return args


def get_surface_reconstruction_configs(args: argparse.Namespace) -> List[Tuple]:
    if args.surface_reconstruction_algorithms is None:
        print(f"No surface reconstruction algorithms specified. Using default surface reconstruction method: "
              f"{SRM.DELAUNAY_TRIANGULATION}")
        unique_desired_algorithms = {SRM.DELAUNAY_TRIANGULATION}
    else:
        unique_desired_algorithms = set([alg.lower().strip() for alg in args.surface_reconstruction_algorithms])

    # Get all the names for the algorithms we want to run
    if 'all' in unique_desired_algorithms:
        unique_desired_algorithms = set([method for method in SRM])
    else:
        unique_desired_algorithms = set([surface_reconstruction.get_surface_reconstruction_method(i) for i in
                                         unique_desired_algorithms])

    result = []

    # Add the alpha shapes configurations
    alphas = set(args.alpha) if isinstance(args.alpha, type(iter)) else args.alpha
    if SRM.ALPHA_SHAPES in unique_desired_algorithms:
        for alpha in alphas:
            result.append((SRM.ALPHA_SHAPES, {'alpha': alpha}))

    if SRM.BALL_PIVOTING_ALGORITHM in unique_desired_algorithms:
        radii_incl_cutoffs = []
        radii = []
        cutoff_index = 0
        for i in range(len(args.ball_pivoting_radii)):
            radius = args.ball_pivoting_radii[i]
            radii.append(radius)
            cutoff = args.bpa_radii_cutoff[cutoff_index]
            if len(radii) >= cutoff:
                cutoff_index = min(cutoff_index + 1, len(args.bpa_radii_cutoff) - 1)
                radii_incl_cutoffs.append(radii.copy())
                radii.clear()
        result.append((SRM.BALL_PIVOTING_ALGORITHM, {SRP.BPA_RADII: radii}))

    if SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION in unique_desired_algorithms:
        for max_octree_depth in args.poisson_octree_max_depth:
            params = {SRP.POISSON_OCTREE_MAX_DEPTH: max_octree_depth,
                      SRP.POISSON_DENSITY_QUANTILE_THRESHOLD: args.poisson_density_quantile}
            result.append((SRM.SCREENED_POISSON_SURFACE_RECONSTRUCTION, params))

    if SRM.DELAUNAY_TRIANGULATION in unique_desired_algorithms:
        result.append((SRM.DELAUNAY_TRIANGULATION, {}))

    return result


def get_run_configurations_from_args(args: argparse.Namespace) -> List[RunConfiguration]:

    config = RunConfiguration()

    # 1. Important file paths. We can directly copy this, it will be checked for validity by the parser AND later.
    config.point_cloud_path = args.point_cloud_path
    config.segments_path = args.segments_path
    config.classifications_path = args.classifications_path

    # 2. Downsampling parameters
    config.down_sample_method = pcd_utils.get_down_sample_method(args.down_sample_method)
    config.down_sample_params = args.down_sample_params

    # Point cloud Normal Estimation Settings
    config.normal_estimation_neighbours = min(max(0, args.normal_estimation_neighbours), 1000)
    config.normal_estimation_radius = min(max(0, args.normal_estimation_radius), 1000)
    config.skip_normalizing_normals = args.skip_normalizing_normals
    config.orient_normals = min(max(args.orient_normals, 0), 1000)

    # Surface reconstruction settings.
    config.surface_reconstruction_method = surface_reconstruction.get_surface_reconstruction_method(args.surface_reconstruction_algorithm)
    config.alpha = min(max(args.alpha, 0), 100)

    if isinstance(args.ball_pivoting_radii, float):
        config.ball_pivoting_radii = [args.ball_pivoting_radii]
    elif isinstance(config.ball_pivoting_radii, list):
        config.ball_pivoting_radii = set(args.ball_pivoting_radii)
    config.poisson_density_quantile = min(max(args.poisson_density_quantile, 0), 1)

    config.poisson_octree_min_width = min(max(args.poisson_octree_min_width, 0), 100)
    config.poisson_octree_max_depth = min(max(args.poisson_octree_max_depth, 0), 100)
    if config.poisson_octree_min_width == 0 and config.poisson_octree_max_depth < 2:
        config.poisson_octree_max_depth = 3

    config.poisson_adaptive = False # ?

    # Mesh cleaning and evaluation
    if args.mesh_clean_methods is not None and len(args.mesh_clean_methods) > 0:
        config.mesh_cleaning_methods = set([mesh_cleaning.get_cleaning_type(m) for m in args.mesh_clean_methods])
    config.edge_length_cleaning_portion = min(0.0, max(args.edge_length_clean_portion, 1.0))
    config.aspect_ratio_cleaning_portion = min(0.0, max(args.aspect_ratio_clean_portion, 1.0))

    if not args.mesh_quality_metrics:
        config.mesh_quality_metrics = None
    elif args.mesh_quality_metrics is 'all' or 'all' in [i.lower().strip() for i in args.mesh_quality_metrics]:
        config.mesh_quality_metrics = [i for i in mesh_quality.MeshEvaluationMetric]
    else:
        config.mesh_quality_metrics = set([mesh_quality.get_mesh_quality_metric(i) for i in args.mesh_quality_metrics])

    config.triangle_normal_deviation_method = TNDM.ADJANCENCY

    config.store_mesh = True  # Always store the mesh for ETVR

    config.processes = max(min(args.processes, 100), -1)
    if config.processes == 0:
        config.processes = 1

    configs: List[RunConfiguration] = [config]
    return configs


def get_run_configurations_from_json(file_name: Path) -> Tuple[List[RunConfiguration], bool, bool, bool]:
    verbose = True
    draw = False
    copy = False  # Whether to copy all parameters of a previous run and overwrite what is specified

    configs = []

    with open(file=file_name, mode='r') as f:
        text = f.read()
        json_data_raw = json.loads(text)

        if not ("runs" in json_data_raw) or not ("point_cloud_path" in json_data_raw):
            print(f"No runs found in {file_name} OR no file path found.")
            return [], verbose, draw, copy

        if "verbose" in json_data_raw:
            verbose = bool(json_data_raw["verbose"])
        if "draw" in json_data_raw:
            draw = bool(json_data_raw["draw"])
        if "copy" in json_data_raw:
            copy = bool(json_data_raw["copy"])

        pcd_path = Path(json_data_raw['point_cloud_path'])
        if "segments_path" in json_data_raw:
            segments_path = json_data_raw['segments_path']
        else:
            segments_path = None

        classifications_path = Path(json_data_raw['classifications_path']) if 'classifications_path' in json_data_raw else None

        for i in range(len(json_data_raw["runs"])):
            run_config_raw = json_data_raw["runs"][i]
            try:
                base_config = None
                if i > 0 and copy:
                    base_config = configs[i - 1]  # If we want to copy the previous config
                config = get_run_config_from_json(run_config_raw,
                                                  pcd_path=pcd_path,
                                                  base_config=base_config,
                                                  segments_path=segments_path,
                                                  classifications_path=classifications_path)
                configs.append(config)
            except json.JSONDecodeError as e:
                print(f"Could not parse run {i} json config file {file_name}: {e}")
            except Exception as e:
                raise e

    return configs, verbose, draw, copy


def get_run_config_from_json(data,
                             pcd_path: Union[Path, str],
                             base_config: RunConfiguration = None,
                             segments_path: Union[Path, str] = None,
                             classifications_path: Union[Path, str] = None) \
        -> RunConfiguration:

    config = RunConfiguration()
    if base_config is not None:
        config = base_config.copy()
        print("Config copies settings from base config.")

    config.point_cloud_path = Path(pcd_path) if isinstance(pcd_path, str) else pcd_path
    config.segments_path = Path(segments_path) if isinstance(segments_path, str) else segments_path
    config.classifications_path = Path(classifications_path) if isinstance(classifications_path, str) else classifications_path

    config.set_setting(data, "down_sample_method", default=None, cast_method=pcd_utils.get_down_sample_method)
    config.set_setting(data, "down_sample_params", default=0, cast_method=float)
    config.set_setting(data, "normal_estimation_neighbours", default=0, cast_method=int)

    if "normal_estimation_radius" in data and data["normal_estimation_radius"] == "auto":
        config.normal_estimation_radius = config.down_sample_params * math.sqrt(12)
        print(f"Set normal estimation to auto: {config.normal_estimation_radius}")
    else:
        config.set_setting(data, "normal_estimation_radius", default=config.down_sample_params * math.sqrt(12), cast_method=float)
    config.set_setting(data, "skip_normalizing_normals", default=False, cast_method=bool)
    config.set_setting(data, "orient_normals", default=0, cast_method=int)

    config.set_setting(data, "surface_reconstruction_method",
                       default=SRM.DELAUNAY_TRIANGULATION,
                       cast_method=surface_reconstruction.get_surface_reconstruction_method)

    # Surface reconstruction parameters
    config.set_setting(data, "alpha", default=0.5, cast_method=float)
    config.set_setting(data, "ball_pivoting_radii", default=[0.1, 0.2, 0.3], cast_method=float, force_unique=True)
    config.set_setting(data, "poisson_density_quantile", default=0.1, cast_method=float)
    config.set_setting(data, "poisson_octree_max_depth", default=8, cast_method=int)
    config.set_setting(data, "poisson_adaptive", default=False, cast_method=bool)

    # Mesh cleaning
    config.set_setting(data, "mesh_cleaning_methods", default=None, force_unique=True,
                       cast_method=mesh_cleaning.get_cleaning_type, handle_all_value=True, special_all_value=MCM.ALL,
                       special_all_values_getter=MCM)
    config.set_setting(data, "edge_length_cleaning_portion", default=0.9, cast_method=float)
    config.set_setting(data, "aspect_ratio_cleaning_portion", default=0.9, cast_method=float)

    # Evaluation
    config.set_setting(data, "mesh_quality_metrics", default=None, handle_all_value=True,
                       special_all_value=MEM.ALL, special_all_values_getter=MEM,
                       cast_method=mesh_quality.get_mesh_quality_metric)
    config.set_setting(data, "triangle_normal_deviation_method", default=TNDM.ADJANCENCY,
                       cast_method=mesh_quality.get_normal_deviation_method)

    # Utility
    config.set_setting(data, "store_mesh", default=False, cast_method=bool)
    config.set_setting(data, "store_preprocessed_pointcloud", default=False, cast_method=bool)
    config.set_setting(data, "processes", default=1, cast_method=int)
    config.set_setting(data, "chunk_size", default=1000_000, cast_method=int)
    config.set_setting(data, "reuse_pointcloud", default=True, cast_method=bool)
    return config


def get_metric_to_list(data, name: str, default_single: Union[int, float], verbose: bool = True):
    t = type(default_single)
    if name in data:
        d = data[name]
        if isinstance(d, list) or isinstance(d, tuple):
            return [t(i) for i in data[name]]
        elif isinstance(d, t):
            return [d]

    if verbose:
        print(f"Metric {name} not found. Defaulting to {default_single}.")

    return [default_single]


def get_setting(data: dict, name: str, default, cast_type: Optional[type]):
    if not data or not (name in data):
        return default

    return data[name] if cast_type is not None else cast_type(data[name])


def extract_quality_metric(data, metric_name: str, everything: Set[MEM]) -> Optional[Set[MEM]]:
    mem = None
    if metric_name in data:
        m = data[metric_name]

        if isinstance(m, list) or isinstance(m, tuple) or isinstance(m, np.ndarray):
            mem = set([mesh_quality.get_mesh_quality_metric(i.lower().strip()) for i in m])
        elif isinstance(m, str):
            mem = {mesh_quality.get_mesh_quality_metric(m)}
    if mem and MEM.ALL in mem:
        mem = everything
    elif mem and None in mem:  # Reduce to None if it's only a set containing None
        mem = None
    return mem
