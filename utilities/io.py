import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import surface_reconstruction

from utilities.run_configuration import RunConfiguration
from utilities import mesh_cleaning, pcd_utils
from utilities.enumerations import SurfaceReconstructionMethod as SRM
from utilities.enumerations import SurfaceReconstructionParameters as SRP
from utilities.enumerations import DownSampleMethod as DSM
from utilities.enumerations import CleaningType


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
                             "Used to clean a mesh after reconstruction using Screened Poisson Surface Reconstruction.")

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
    if args.mesh_clean_methods is None or CleaningType.NONE in args.mesh_clean_methods:
        mesh_cleaning_methods = {CleaningType.NONE}
    else:
        mesh_cleaning_methods = set([mesh_cleaning.get_cleaning_type(m) for m in args.mesh_clean_methods])

    edge_length_clean_portion = min(0.0, max(args.edge_length_clean_portion, 1.0))
    aspect_ratio_clean_portion = min(0.0, max(args.aspect_ratio_clean_portion, 1.0))

    # Not a set, since we can have the same sampling method multiple times.
    if args.down_sample_methods is None or None in args.down_sample_methods or DSM.NONE in args.down_sample_methods:
        point_cloud_samplings: List[DSM] = [DSM.NONE]
    else:
        point_cloud_samplings: List[DSM] = [pcd_utils.get_down_sample_method(m) for m in args.down_sample_methods]

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
                                      normal_estimation_neighbours=args.normal_estimation_neighbours)
            configs.append(config)

    return configs
