import open3d
from open3d.geometry import TriangleMesh, PointCloud
from open3d.t.geometry import RaycastingScene
from open3d.core import Tensor
import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)
import numpy as np
import time
import matplotlib.pyplot as plt


def evaluate_point_clouds(point_cloud_a: PointCloud, point_cloud_b: PointCloud):
    print("Evaluating...")
    pcd_a_points = np.asarray(point_cloud_a.points)
    pcd_b_points = np.asarray(point_cloud_b.points)
    start_time = time.time()
    hausdorff = pcu.hausdorff_distance(pcd_a_points, pcd_b_points)
    chamfer = pcu.chamfer_distance(pcd_a_points, pcd_b_points)
    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer} [{elapsed_time}s]")


def evaluate_point_cloud_mesh(point_cloud: PointCloud, mesh: TriangleMesh):
    # Computer Chamfer and Hausdorff distances.
    pcd_points = np.asarray(point_cloud.points)
    mesh_points = np.asarray(mesh.vertices)
    start_time = time.time()
    hausdorff = pcu.hausdorff_distance(pcd_points, mesh_points)
    chamfer = pcu.chamfer_distance(pcd_points, mesh_points)

    # Compute point cloud to mesh surfaces distances.
    rcs = RaycastingScene()
    tensor_mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rcs.add_triangles(tensor_mesh)
    pts = Tensor(np.array(pcd_points).astype(dtype=np.float32))
    distances = rcs.compute_distance(query_points=pts).numpy()
    max_distance = str(round(distances.max(), 4))

    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer}, Max Dist={max_distance} [{elapsed_time}s]")

    plt.hist(distances, histtype='step', log=True, bins=100)
    plt.show()
