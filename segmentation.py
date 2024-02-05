import numpy as np
import time
import open3d
import matplotlib.pyplot as plt


def dbscan_clustering(point_cloud, eps=0.02, min_points=10, verbose=True, add_colors=True):
    start_time = time.time()
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=verbose))
    end_time = time.time()

    if verbose:
        elapsed_time = str(round(end_time - start_time, 2))
        print(f"Clustered points using DBSCAN (ε={eps} , min_points={min_points}) [{elapsed_time}s]")

    colors = None
    if add_colors:
        max_label = labels.max()
        if verbose:
            print(f"point cloud has {max_label + 1} clusters")

        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        point_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])
    return point_cloud, labels, colors
