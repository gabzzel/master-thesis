import argparse
import json
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
from utilities.run_configuration import RunConfiguration


def parse_args() -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser()
    parser.add_argument('point_cloud_path', type=Path, help="The path to the point cloud file.")

    parser.add_argument('-down_sample_methods', '-dsm',
                        type=str, default=None, action='append',
                        help="The down-sampling method(s) to use. ",
                        choices=[None, 'none', 'voxel', 'random'])

    parser.add_argument('-down_sample_params', '-dsp',
                        type=float, default=0.01, action='append',
                        help="When doing voxel down-sampling, this parameter will be the voxel size. When random "
                             "down-sampling, this parameter will be the amount of sample ratio.")

    parser.add_argument('-normal_estimation_neighbours', '-nen',
                        type=int, default=30, action='store',
                        help="The number of neighbours to use during normal estimation.")

    parser.add_argument('-normal_estimation_radius', '-ner',
                        type=float, default=0.5, action='store',
                        help="The radius around the point to use during normal estimation.")

    parser.add_argument('-skip_normalizing_normals', '-skip_nornor', '-snn',
                        action='store_true',
                        help="Whether to skip normalizing the normals after estimation.")

    parser.add_argument('-orient_normals', '-on', '-ornor', type=int, action='store', default=0,
                        help="Whether to orient the normals after estimation. Expensive, but might lead to better "
                             "results.")

    parser.add_argument('-verbose', '-v', action='store_true',
                        help='Whether to print progress and results in the console.')

    parser.add_argument('-surface_reconstruction_algorithms', '-sra', '-sr_alg',
                        type=str, action='append',
                        help="The surface reconstruction algorithms to run. Use 'all' to run all 4. \n "
                             "Delaunay = 'd' or 'D' \n"
                             "Ball Pivoting = 'bpa', 'b' or 'B' \n"
                             "Alpha Shapes = 'alpha', 'a' or 'A' \n"
                             "Screened Poisson = 'spsr', 'SPSR', 'sp', 'SP', 's' or 'S'",
                        choices=['delaunay', 'D', 'd', 'ball_pivoting', 'bpa', 'bp', 'b', 'poisson', 'spsr', 'P', 'p',
                                 'alpha_shapes', 'alpha', 'a', 'A'])

    parser.add_argument('-alpha', '-a', type=float, action='append', default=0.02,
                        help="The alpha value(s) to run Alpha Shapes with.")

    parser.add_argument('-ball_pivoting_radii', '-bpa_radii', '-bpar',
                        default=0.1, type=float, action='append',
                        help="The ball radii to for ball pivoting.")

    parser.add_argument('-bpa_radii_cutoff', '-bparc',
                        default=0, type=int, action='append',
                        help="After how many radii a new execution of BPA should begin.")

    parser.add_argument('-poisson_density_quantile', '-pdq',
                        type=float, action='store', default=0.1,
                        help="The lower quantile/portion of the points sorted by density / support is removed."
                             "Used to clean mesh after reconstruction using Screened Poisson Surface Reconstruction.")

    parser.add_argument('-poisson_octree_max_depth', '-pomd',
                        type=int, action='append', default=8,
                        help="The maximum depth of the octree when using Poisson Surface Reconstruction. Lower values"
                             " correspond to lower resoluation and thus less detailed meshes, but reduces "
                             "computing time.")

    parser.add_argument('-mesh_clean_methods', '-mcm',
                        required=False, action='append', type=str,
                        choices=['simple', 's', 'edge_length', 'aspect_ratio', 'el', 'ar'])

    parser.add_argument('-edge_length_clean_portion', '-elcp',
                        required=False, action='store', type=float, default=0.95,
                        help="The portion of the edge lengths to keep when cleaning.")

    parser.add_argument('-aspect_ratio_clean_portion', '-arcp',
                        required=False, action='store', type=float, default=0.95,
                        help="The portion of triangles, sorted by aspect ratio, to keep when cleaning.")

    parser.add_argument('-mesh_quality_metrics', '-mqm',
                        required=False, action='append', type=str,
                        help="Which metrics to use during evaluation of the reconstructed surface /  mesh.",
                        choices=['all', 'edge_lengths', 'el', 'aspect_ratios', 'ar', 'connectivity', 'co',
                                 'discrete_curvature', 'dc', 'normal_deviations', 'nd'])

    parser.add_argument('-mesh_to_cloud_metrics', '-m2cm',
                        required=False, action='append', type=str,
                        help="Which metrics to use to evaluate to surface reconstruction to the point cloud.",
                        choices=['all', 'chamfer', 'c', 'hausdorff', 'h', 'distances', 'distance', 'd'])

    parser.add_argument('-result_path', action='store', type=Path, required=False,
                        help="Path to the folder where to write the evaluation results.")

    parser.add_argument('-store_mesh', '-save_mesh', action='store_true')

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
    if not args.mesh_clean_methods:
        mesh_cleaning_methods = None
    else:
        mesh_cleaning_methods = set([mesh_cleaning.get_cleaning_type(m) for m in args.mesh_clean_methods])

    edge_length_clean_portion = min(0.0, max(args.edge_length_clean_portion, 1.0))
    aspect_ratio_clean_portion = min(0.0, max(args.aspect_ratio_clean_portion, 1.0))

    # Not a set, since we can have the same sampling method multiple times.
    if not args.down_sample_methods:
        point_cloud_samplings = None
    else:
        point_cloud_samplings = [pcd_utils.get_down_sample_method(m) for m in args.down_sample_methods]

    # Parse the mesh quality metrics
    if not args.mesh_quality_metrics:
        mesh_quality_metrics = None
    elif args.mesh_quality_metrics is 'all' or 'all' in [i.lower().strip() for i in args.mesh_quality_metrics]:
        mesh_quality_metrics = [i for i in mesh_quality.MeshEvaluationMetric]
    else:
        mesh_quality_metrics = set([mesh_quality.get_mesh_quality_metric(i) for i in args.mesh_quality_metrics])

    configs: List[RunConfiguration] = []

    for i in range(len(point_cloud_samplings)):
        down_sample_method: DSM = point_cloud_samplings[i]

        # Loop through the down sample parameters if we have multiple.
        # If we only have a single down sample parameter, use that every time.
        down_sample_param: float = args.down_sample_params[i % len(args.down_sample_params)] \
            if isinstance(args.down_sample_params, type(iter)) \
            else args.down_sample_params

        for sr_config in get_surface_reconstruction_configs(args):
            config = RunConfiguration(pcd_path=args.point_cloud_path,
                                      down_sample_method=down_sample_method,
                                      down_sample_params=down_sample_param,
                                      surface_reconstruction_method=sr_config[0],
                                      surface_reconstruction_params=sr_config[1],
                                      mesh_cleaning_methods=mesh_cleaning_methods,
                                      edge_length_cleaning_portion=edge_length_clean_portion,
                                      aspect_ratio_cleaning_portion=aspect_ratio_clean_portion,
                                      orient_normals=args.orient_normals,
                                      skip_normalizing_normals=args.skip_normalizing_normals,
                                      normal_estimation_radius=args.normal_estimation_radius,
                                      normal_estimation_neighbours=args.normal_estimation_neighbours,
                                      mesh_evaluation_metrics=mesh_quality_metrics)
            configs.append(config)

    return configs


def get_run_configurations_from_json(file_name: Path) -> Tuple[List[RunConfiguration], bool, bool]:
    verbose = True
    draw = False

    configs = []

    with open(file=file_name, mode='r') as f:
        text = f.read()
        json_data_raw = json.loads(text)

        if not ("runs" in json_data_raw) or not ("point_cloud_path" in json_data_raw):
            print(f"No runs found in {file_name} OR no file path found.")
            return [], verbose, draw

        if "verbose" in json_data_raw:
            verbose = bool(json_data_raw["verbose"])
        if "draw" in json_data_raw:
            draw = bool(json_data_raw["draw"])

        pcd_path = Path(json_data_raw['point_cloud_path'])

        for i in range(len(json_data_raw["runs"])):
            run_config_raw = json_data_raw["runs"][i]
            try:
                config = run_config_from_json(run_config_raw, pcd_path=pcd_path)
                configs.append(config)
            except json.JSONDecodeError as e:
                print(f"Could not parse run {i} json config file {file_name}: {e}")
            except Exception as e:
                print(f"{e}")

    return configs, verbose, draw


def run_config_from_json(data, pcd_path: Union[Path, str]) -> RunConfiguration:
    config = RunConfiguration()
    config.point_cloud_path = Path(pcd_path) if isinstance(pcd_path, str) else pcd_path

    if "down_sample_method" in data:
        config.down_sample_method = pcd_utils.get_down_sample_method(data["down_sample_method"])
    else:
        print(f"No down_sample_method in config. Using default {None}")
        config.down_sample_method = None

    config.down_sample_params = get_setting(data, "down_sample_param", 0, type(float))
    config.normal_estimation_neighbours = get_setting(data, "normal_estimation_neighbours", 0, type(int))
    config.normal_estimation_radius = get_setting(data, "normal_estimation_radius", 0.0, type(float))
    config.skip_normalizing_normals = get_setting(data, "skip_normalizing_normals", False, type(bool))
    config.orient_normals = get_setting(data, "orient_normals", 0, type(int))

    config.surface_reconstruction_method = SRM.DELAUNAY_TRIANGULATION
    if "surface_reconstruction_algorithm" in data:
        sra_input = data["surface_reconstruction_algorithm"]
        config.surface_reconstruction_method = surface_reconstruction.get_surface_reconstruction_method(sra_input)
    else:
        print(f"No surface reconstruction algorithm found in config. Using default {SRM.DELAUNAY_TRIANGULATION}")

    # Surface reconstruction parameters
    alpha = get_metric_to_list(data, name="alpha", default_single=0.5)
    bpa_radii = get_metric_to_list(data, name="ball_pivoting_radii", default_single=0.5)
    poisson_density_quantile = get_setting(data, "poisson_density_quantile", 0.1, type(float))
    poisson_octree_max_depth = get_setting(data, "poisson_octree_max_depth", 8, type(int))
    config.surface_reconstruction_params = {
        SRP.ALPHA: alpha,
        SRP.BPA_RADII: bpa_radii,
        SRP.POISSON_DENSITY_QUANTILE_THRESHOLD: poisson_density_quantile,
        SRP.POISSON_OCTREE_MAX_DEPTH: poisson_octree_max_depth
    }

    mcm = None
    if "mesh_cleaning_methods" in data:
        if isinstance(data["mesh_cleaning_methods"], list):
            mcm = set([mesh_cleaning.get_cleaning_type(i.lower().strip()) for i in data["mesh_cleaning_methods"]])
        elif isinstance(data["mesh_cleaning_methods"], str):
            mcm = {mesh_cleaning.get_cleaning_type(data["mesh_cleaning_methods"])}
    if mcm and MCM.ALL in mcm:
        mcm = set([i for i in MCM])
    config.mesh_cleaning_methods = mcm

    config.edge_length_cleaning_portion = get_setting(data, "edge_length_clean_portion", 0.9, type(float))
    config.aspect_ratio_cleaning_portion = get_setting(data, "aspect_ratio_clean_portion", 0.9, type(float))

    mem = extract_quality_metric(data, metric_name="mesh_quality_metrics", everything={MEM.DISCRETE_CURVATURE,
                                                                                       MEM.TRIANGLE_NORMAL_DEVIATIONS,
                                                                                       MEM.EDGE_LENGTHS,
                                                                                       MEM.TRIANGLE_ASPECT_RATIOS,
                                                                                       MEM.CONNECTIVITY})
    m2cm = extract_quality_metric(data, metric_name="mesh_to_cloud_metrics", everything={MEM.CHAMFER_DISTANCE,
                                                                                         MEM.HAUSDORFF_DISTANCE,
                                                                                         MEM.MESH_TO_CLOUD_DISTANCE})
    if mem is not None and m2cm is not None:
        mem = mem.union(m2cm)
    elif mem is None and m2cm is not None:
        mem = m2cm

    config.mesh_evaluation_metrics = mem
    config.store_mesh = get_setting(data, "store_mesh", False, type(bool))
    config.store_preprocessed_pointcloud = get_setting(data, "store_preprocessed_pointcloud", False, type(bool))
    return config


def get_metric_to_list(data, name: str, default_single: Union[int, float], verbose: bool = True):
    t = type(default_single)
    if name in data:
        d = data[name]
        if isinstance(d, list) or isinstance(d, tuple):
            return [t(i) for i in data["ball_pivoting_radii"]]
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
