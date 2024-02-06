import open3d
from open3d.geometry import TriangleMesh, PointCloud
from open3d.t.geometry import RaycastingScene
from open3d.core import Tensor
import point_cloud_utils as pcu  # https://github.com/fwilliams/point-cloud-utils (Hausdorff, normals, Chamfer)
import numpy as np
import time
import matplotlib.pyplot as plt
import utils


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
    hausdorff = round(pcu.hausdorff_distance(pcd_points, mesh_points), 4)
    chamfer = round(pcu.chamfer_distance(pcd_points, mesh_points), 4)

    # Compute point cloud to mesh surfaces distances.
    rcs = RaycastingScene()
    tensor_mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rcs.add_triangles(tensor_mesh)
    pts = Tensor(np.array(pcd_points).astype(dtype=np.float32))
    distances = rcs.compute_distance(query_points=pts).numpy().astype(dtype=np.float32)
    max_distance = np.max(distances)
    distances_norm = distances / max_distance
    colors = np.asarray(point_cloud.colors)
    colors[:, 0] = distances_norm
    colors[:, 1] = np.full_like(distances_norm, 0.9)
    colors[:, 2] = np.full_like(distances_norm, 0.9)

    end_time = time.time()
    elapsed_time = str(round(end_time - start_time, 3))
    print(f"Evaluated. Hausdorff={hausdorff}, Chamfer={chamfer}, Max Dist={max_distance} [{elapsed_time}s]")

    plt.hist(distances, histtype='step', log=True, bins=100)


def evaluate_mesh(mesh: TriangleMesh, aspect_ratios=None):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Compute edge lengths and remove duplicates!
    edges = np.sort(triangles[:, [0, 1]])  # Sort the vertices by index, take only the first and second vertex
    edges = np.unique(edges, axis=0)  # Only keep the unique ones
    edges = np.vstack((edges, np.sort(triangles[:, [1, 2]])))
    edges = np.unique(edges, axis=0)
    edges = np.vstack((edges, np.sort(triangles[:, [2, 0]])))
    edges = np.unique(edges, axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    utils.get_stats(edge_lengths, name="Edge Lengths", print_only=True)

    # If we do not get the aspect ratios, calculate them ourselves.
    if aspect_ratios is None:
        aspect_ratios = utils.get_mesh_aspect_ratios(mesh)

    utils.get_stats(aspect_ratios, name="Aspect Ratios", print_only=True)
    plt.hist(aspect_ratios, histtype='step', log=True, bins=100)
    plt.show()
