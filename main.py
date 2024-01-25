import pclpy  # https://github.com/davidcaron/pclpy
from pclpy import pcl

import open3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dragon_path = "C:\\Users\\Gabi\\master-thesis\\master-thesis\\data\\dummy\\stanford-dragon\\dragon_recon\\dragon_vrip.ply"
    point_cloud = open3d.io.read_point_cloud(dragon_path)
    labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    open3d.visualization.draw_geometries([point_cloud])
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    point_cloud.colors = open3d.utility.Vector3dVector(colors[:, :3])

    print("Hello world")
