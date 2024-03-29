import numpy as np
from sklearn.cluster import HDBSCAN
import open3d


def hdbscan(pcd: open3d.geometry.PointCloud):
    # The minimum number of samples in a group for that group to be considered a cluster;
    # groupings smaller than this size will be left as noise.
    min_cluster_size: int = 10

    # The number of samples in a neighborhood for a point to be considered as a core point.
    # This includes the point itself. When None, defaults to min_cluster_size.
    min_samples: int = None

    # A distance threshold. Clusters below this value will be merged.
    cluster_selection_epsilon: float = 0.0

    # A distance scaling parameter as used in robust single linkage. See [3] for more information.
    alpha: float = 1.0

    # Leaf size for trees responsible for fast nearest neighbour queries when a KDTree or a BallTree are used as
    # core-distance algorithms. A large dataset size and small leaf_size may induce excessive memory usage.
    # If you are running out of memory consider increasing the leaf_size parameter. Ignored for algorithm="brute".
    leaf_size: int = 100

    # Number of jobs to run in parallel to calculate distances. None means 1 unless in a joblib.parallel_backend
    # context. -1 means using all processors. See Glossary for more details.
    n_jobs: int = -1

    model = HDBSCAN(min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    max_cluster_size=None,  # Allow large clusters
                    metric="euclidean",
                    metric_params=None,  # Not needed.
                    alpha=alpha,
                    algorithm="auto",  # uses KD-Tree if possible, else BallTree
                    leaf_size=leaf_size,
                    n_jobs=n_jobs,
                    cluster_selection_method="eom",  # Use the standard Excess Of Mass metric
                    allow_single_cluster=False,  # Having a single cluster probably means something has gone wrong...
                    store_centers=None,  # We don't need centroids or mediods
                    copy=True)  # Just to be safe.

    X: np.ndarray = np.asarray(pcd.points)
    if pcd.has_colors():
        X = np.hstack((X, np.asarray(pcd.colors)), dtype=np.float32)

    return model.fit_predict(X)
