import time
from pathlib import Path
from typing import Optional, Union

import open3d

from utilities import utils
from utilities.enumerations import DownSampleMethod
from utilities.evaluation_results import EvaluationResults


def get_down_sample_method(down_sample_method_string: str) -> Optional[DownSampleMethod]:
    t = down_sample_method_string.lower().strip()
    t = t.replace(" ", "_")
    if t == "voxel" or t == "v":
        return DownSampleMethod.VOXEL
    elif t == "random" or t == "rand" or t == "r":
        return DownSampleMethod.RANDOM
    return None


def load_point_cloud(path: Union[Path, str],
                     results: EvaluationResults,
                     down_sample_method: DownSampleMethod = None,
                     down_sample_param: Optional[Union[float, int]] = None,
                     verbose=True) -> open3d.geometry.PointCloud:
    """
    Load a point cloud into Open3D format.

    :param verbose: Whether to print progress and status to the console.
    :param path: The path to the file from which to load the point cloud.
    :param down_sample_method: Either None, 'voxel' or 'random'.
    :param down_sample_param: Depending on the down sample method:
    Either the ratio of random points that will be kept [0-1] when random down sampling or
    The size of the voxels used during down sampling
    :returns: The loaded point cloud, in Open3D format.
    """

    if verbose:
        print("Loading point cloud...")

    start_time = time.time()
    pcd = open3d.io.read_point_cloud(filename=str(path),
                                     remove_nan_points=True,
                                     remove_infinite_points=True,
                                     print_progress=True)
    end_time = time.time()

    if verbose:
        num_pts = utils.format_number(len(pcd.points))
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Loaded point cloud with {num_pts} points [{elapsed_time}s]")

    if down_sample_method is None or not (down_sample_method in DownSampleMethod):
        if verbose:
            print(f"Skipping downsampling: got None or invalid downsampling method {down_sample_method}")
        return pcd

    num_points_original = len(pcd.points)
    npof = utils.format_number(num_points_original)  # Number Points Original Formatted
    start_time = time.time()

    if down_sample_method == DownSampleMethod.VOXEL:
        pcd = pcd.voxel_down_sample(voxel_size=down_sample_param)
    elif down_sample_method == DownSampleMethod.RANDOM:
        down_sample_param = max(0, min(1, down_sample_param))
        pcd = pcd.random_down_sample(sampling_ratio=down_sample_param)

    end_time = time.time()
    npad = len(pcd.points)

    if verbose:
        elapsed = str(round(end_time - start_time, 2))  # The number of seconds elapsed during downsampling operation
        ratio = str(round(float(len(pcd.points)) / float(num_points_original) * 100))
        print(f"Downsampled {npof} pts -> {utils.format_number(npad)} pts ({ratio}%) "
              f"({down_sample_method} @ {down_sample_param}) [{elapsed}s]")

    results.number_of_vertices_original = num_points_original
    results.number_of_vertices_after_downsampling = npad
    return pcd


def estimate_normals(point_cloud: open3d.geometry.PointCloud,
                     max_nn: int = None,
                     radius: float = None,
                     orient: int = None,
                     normalize: bool = True,
                     verbose: bool = True):
    """
    Estimate the normals for a point cloud. Functions as a wrapper for Open3D methods.

    :param point_cloud: The point cloud for which to estimate the normals.
    :param max_nn: The maximum amount of neighbours used to estimate the normal of a given point.
    :param radius: The maximum radius in which neighbours will be used to estimate the normal of a given point.
    :param orient: The amount of points used around a given point to align / orient normals consistently. Costly,
    but can improve the quality of surface reconstruction or other calculations downstream. Set to None or 0 to ignore.
    :param normalize: Whether to normalize the normals after calculation. [Recommended = True]
    :param verbose: Whether to print the progress.
    """

    start_time = time.time()
    if verbose:
        print("Estimating normals...")

    max_nn_valid = max_nn is not None and isinstance(max_nn, int) and max_nn > 0

    # From Open3D docs: "neighbors search radius parameter to use HybridSearch. [Recommended ~1.4x voxel size]"
    radius_valid = radius is not None and isinstance(radius, float) and radius > 0.0

    if not max_nn_valid and not radius_valid:
        if verbose:
            print(f"Skipping normal estimation. Both max_nn {max_nn} and radius {radius} values are 0, None or invalid.")
        return

    if max_nn_valid and radius_valid:
        params_str = f"Max NN={max_nn}, radius={radius}"
        point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    elif max_nn_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(max_nn))
    elif radius_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamRadius(radius))
    else:
        if verbose:
            print("Point cloud normal estimation failed, parameters invalid.")
        return

    if normalize:
        point_cloud.normalize_normals()

    end_time = time.time()
    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Estimated normals ({params_str}) (Normalized={normalize}) [{elapsed_time}s]")

    if orient is None or not isinstance(orient, int) or orient <= 0:
        return

    if verbose:
        print("Orienting normals w.r.t. tangent plane...")
    start_time = time.time()
    point_cloud.orient_normals_consistent_tangent_plane(orient)
    end_time = time.time()
    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Oriented normals (KNN={orient}) [{elapsed_time}s]")
