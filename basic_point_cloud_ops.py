import open3d
from open3d.geometry import KDTreeSearchParamHybrid, KDTreeSearchParamKNN, KDTreeSearchParamRadius
import time
from utils import format_number


def load_point_cloud(path, down_sample_method=None, down_sample_param=None, verbose=True):
    """
    Load a point cloud into Open3D format.

    :param path: The path to the file from which to load the point cloud.
    :param down_sample_method: Either None, 'voxel' or 'random'.
    :param down_sample_param: Depending on the down sample method:
    Either the ratio of random points that will be kept [0-1] when random down sampling or
    The size of the voxels used during down sampling
    """

    if verbose:
        print("Loading point cloud...")

    start_time = time.time()
    pcd = open3d.io.read_point_cloud(path, print_progress=True)
    end_time = time.time()

    if verbose:
        num_pts = format_number(len(pcd.points))
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Loaded point cloud with {num_pts} points [{elapsed_time}s]")

    if down_sample_method is None:
        return pcd

    if not isinstance(down_sample_method, str) or down_sample_method not in ['voxel', 'random']:
        print(f"Invalid downsampling method {down_sample_method}. Cancelling downsampling.")
        return pcd

    num_points_original = len(pcd.points)
    npof = format_number(num_points_original)  # Number Points Original Formatted
    start_time = time.time()

    if down_sample_method == 'voxel':
        pcd = pcd.voxel_down_sample(voxel_size=down_sample_param)
    elif down_sample_method == 'random':
        down_sample_param = max(0, min(1, down_sample_param))
        pcd = pcd.random_down_sample(sampling_ratio=down_sample_param)

    end_time = time.time()

    if verbose:
        elapsed = str(round(end_time - start_time, 2))  # The number of seconds elapsed during downsampling operation
        num_pts = format_number(len(pcd.points))
        ratio = str(round(float(len(pcd.points)) / float(num_points_original) * 100))
        print(f"Downsampled {npof} pts -> {num_pts} pts ({ratio}%) ({down_sample_method} @ {down_sample_param}) [{elapsed}s]")

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

    params_str = "Invalid Parameters"
    max_nn_valid = max_nn is not None and isinstance(max_nn, int) and max_nn > 0

    # From Open3D docs: "neighbors search radius parameter to use HybridSearch. [Recommended ~1.4x voxel size]"
    radius_valid = radius is not None and isinstance(radius, float) and radius > 0.0

    if not max_nn_valid and not radius_valid:
        print(f"WARNING: Both max_nn ({max_nn}) and radius ({radius}) values are invalid. Using default max_nn=30.")
        print("If this is not desired behaviour, please check the entered values and re-run.")
        max_nn = 30
        max_nn_valid = True

    if max_nn_valid and radius_valid:
        params_str = f"Max NN={max_nn}, radius={radius}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    elif max_nn_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamKNN(max_nn))
    elif radius_valid:
        params_str = f"Max NN={max_nn}"
        point_cloud.estimate_normals(search_param=KDTreeSearchParamRadius(radius))
    else:
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
