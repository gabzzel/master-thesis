from typing import List, Optional, Dict

import numpy as np
import open3d
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud
import bisect


class RegionGrowingOctree:
    def __init__(self, point_cloud: PointCloud, root_margin: float = 0.1):
        self.root_node: RegionGrowingOctreeNode = None
        self.leaf_nodes = []
        self.nodes_per_depth: List[List[RegionGrowingOctreeNode]] = []
        self.origin_point_cloud = point_cloud
        self._create_root_node(root_margin)


    def _create_root_node(self, root_margin):
        points = np.asarray(self.origin_point_cloud.points)
        size = 0

        for i in range(3):
            _max = np.max(points[:][i])
            _min = np.min(points[:][i])
            size = max(size, abs(_max - _min) * (1.0 + root_margin))  # Use the maximum size

        min_position = points.min(axis=0) - np.full(shape=(3,), fill_value=size * root_margin * 0.5)
        self.root_node = RegionGrowingOctreeNode(depth=0, size=size, min_position=min_position)
        self.nodes_per_depth.append([self.root_node])

    def initial_voxelization(self, voxel_size: float):
        self._initial_voxelization_points(voxel_size)

    def _initial_voxelization_naive(self, voxel_size: float):
        points = np.asarray(self.origin_point_cloud.points)

        voxel_count_1_dim = int(np.ceil(self.root_node.size / voxel_size))
        for x in tqdm.trange(voxel_count_1_dim, desc="Initial voxelization (naive)"):
            min_x = self.root_node.position_min[0] + x * voxel_size
            max_x = min_x + voxel_size

            for y in range(voxel_count_1_dim):
                min_y = self.root_node.position_min[1] + y * voxel_size
                max_y = min_y + voxel_size

                for z in range(voxel_count_1_dim):
                    min_z = self.root_node.position_min[2] + z * voxel_size
                    max_z = min_z + voxel_size

                    indices = np.nonzero((points[:, 0] >= min_x) &
                                         (points[:, 0] < max_x) &
                                         (points[:, 1] >= min_y) &
                                         (points[:, 1] < max_y) &
                                         (points[:, 2] >= min_z) &
                                         (points[:, 2] < max_z))
                    if len(indices) > 0:
                        node = RegionGrowingOctreeNode(depth=1, size=voxel_size,
                                                       min_position=np.array([min_x, min_y, min_z]))
                        node.vertex_indices = indices
                        self.root_node.children.append(node)

        print("Voxelization done.")

    def _initial_voxelization_sorted(self, voxel_size: float):
        original_points = np.asarray(self.origin_point_cloud.points)

        print("Sorting...")
        sorted_by_coordinate: List[np.ndarray] = []
        sorted_by_coordinate_indices: List[np.ndarray] = []
        for i in range(3):
            sorted_by_coordinate_indices.append(np.argsort(original_points[:, i]))
            sorted_by_coordinate.append(original_points[sorted_by_coordinate_indices[i]][:, i])

        voxel_count_1_dim = int(np.ceil(self.root_node.size / voxel_size))

        # voxel_ranges = np.full(shape=(3, voxel_count_1_dim, 2), fill_value=-1, dtype=np.int32)
        voxel_ranges: List[Dict[int, tuple]] = [{}, {}, {}]

        for voxel_index in tqdm.trange(voxel_count_1_dim, desc="Initial voxelization (step 1)"):
            for dimension in range(3):
                min_coordinate = self.root_node.position_min[dimension] + voxel_index * voxel_size
                max_coordinate = min_coordinate + voxel_size
                min_index = bisect.bisect_left(a=sorted_by_coordinate[dimension], x=min_coordinate)
                max_index = bisect.bisect_right(a=sorted_by_coordinate[dimension], x=max_coordinate, lo=min_index)

                if max_index > min_index:
                    voxel_ranges[dimension][voxel_index] = (min_index, max_index)

        progress_bar = tqdm.tqdm(total=len(voxel_ranges[0]),
                                 desc="Initial voxelization (step 2)")

        for voxel_index_x, (min_index_x, max_index_x) in voxel_ranges[0].items():
            progress_bar.update()
            x_indices = sorted_by_coordinate_indices[0][min_index_x:max_index_x]
            for voxel_index_y, (min_index_y, max_index_y) in voxel_ranges[1].items():
                y_indices = sorted_by_coordinate_indices[1][min_index_y:max_index_y]
                relevant_indices = np.intersect1d(x_indices, y_indices)
                if len(relevant_indices) == 0:
                    continue

                for voxel_index_z, (min_index_z, max_index_z) in voxel_ranges[2].items():
                    z_indices = sorted_by_coordinate_indices[2][min_index_z:max_index_z]
                    relevant_indices = np.intersect1d(relevant_indices, z_indices)
                    if len(relevant_indices) == 0:
                        continue

                    offset = np.array([voxel_index_x, voxel_index_y, voxel_index_z]) * voxel_size
                    pos = self.root_node.position_min + offset
                    node = RegionGrowingOctreeNode(1, pos, voxel_size)
                    node.vertex_indices = relevant_indices.tolist()
                    self.root_node.children.append(node)

        dsnadas = 0

    def _initial_voxelization_points(self, voxel_size: float):
        points = np.asarray(self.origin_point_cloud.points)
        shifted_points = points - self.root_node.position_min
        voxel_grid = {}

        # Calculate the indices of the voxels that each point belongs to
        voxel_indices = np.floor(shifted_points / voxel_size).astype(int)

        voxel_count = 0
        max_vertices_in_a_voxel = 0

        self.nodes_per_depth.append([])  # Make sure we have a list to put all nodes at depth 1

        # Iterate over each point to determine its voxel index
        for i in tqdm.trange(len(voxel_indices), unit="points", desc="Initial voxelization"):
            # Convert voxel index to tuple to use it as a dictionary key
            voxel_index_tuple = tuple(voxel_indices[i])
            # Create a new Voxel object if it doesn't exist already
            if voxel_index_tuple not in voxel_grid:
                pos = self.root_node.position_min + voxel_indices[i] * voxel_size
                node = RegionGrowingOctreeNode(depth=1, min_position=pos, size=voxel_size)
                self.root_node.children.append(node)
                self.nodes_per_depth[1].append(node)

                voxel_grid[voxel_index_tuple] = node
                voxel_count += 1

            # Append the point index to the list of points in the corresponding voxel
            voxel: RegionGrowingOctreeNode = voxel_grid[voxel_index_tuple]
            voxel.vertex_indices.append(i)
            max_vertices_in_a_voxel = max(max_vertices_in_a_voxel, len(voxel.vertex_indices))

        print(f"Created {voxel_count} voxels of size {voxel_size}")
        print(f"Vertices per voxel: avg {len(points) / voxel_count} , max {max_vertices_in_a_voxel}")

    def recursive_subdivide(self,
                            full_threshold: int,
                            minimum_voxel_size: float,
                            residual_threshold: float,
                            max_depth: Optional[int] = None):

        points = np.asarray(self.origin_point_cloud.points)
        normals = np.asarray(self.origin_point_cloud.normals)

        residual_sum: float = 0.0
        residual_counts: int = 0

        for i in tqdm.trange(len(self.root_node.children), desc="Creating octree"):
            node = self.root_node.children[i]
            node.subdivide(leaf_node_list=self.leaf_nodes,
                           nodes_per_depth=self.nodes_per_depth,
                           points=points,
                           normals=normals,
                           full_threshold=full_threshold,
                           minimum_voxel_size=minimum_voxel_size,
                           residual_threshold=residual_threshold,
                           max_depth=max_depth)

            residual_sum += node.residual
            residual_counts += int(node.residual > 0)

        print(f"Octree generation complete, total leaf nodes: {len(self.leaf_nodes)} ")
        print(f"Residual average: {residual_sum / residual_counts}")

    def visualize_voxels(self, depth: int, maximum: Optional[int] = None):
        nodes = []
        for i in range(1, depth + 1):
            if maximum is None:
                nodes.extend(self.nodes_per_depth[i])
            else:
                max_index = min(maximum - len(nodes), len(self.nodes_per_depth[i]))
                nodes.extend(self.nodes_per_depth[i][:max_index])

        create_raw = False
        if create_raw:
            triangles = np.zeros(shape=(len(nodes) * 12, 3, 3))
            for i in tqdm.trange(len(nodes), desc="Creating triangles", unit="voxel"):
                triangles[i * 12:(i + 1) * 12] = nodes[i].get_triangles()
            vertices, index_mapping = np.unique(triangles, return_inverse=True)
            index_mapping = np.reshape(index_mapping, newshape=triangles.shape)
            mesh = open3d.geometry.TriangleMesh(open3d.utility.Vector3dVector(vertices),
                                                open3d.utility.Vector3iVector(index_mapping))
            open3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        else:
            meshes = []
            # nodes = np.random.choice(nodes, size=int(ratio * len(nodes)), replace=False)
            for i in tqdm.trange(len(nodes), desc="Creating boxes", unit="voxel"):
                node = nodes[i]
                mesh: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_box(width=node.size,
                                                                                             height=node.size,
                                                                                             depth=node.size)
                mesh.translate(node.position_min)
                meshes.append(mesh)

            open3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)


class RegionGrowingOctreeNode:
    def __init__(self,
                 depth: int,
                 min_position: np.ndarray,
                 size: float):

        self.depth: int = depth
        self.children: List[RegionGrowingOctreeNode] = []
        self.vertex_indices: List[int] = []

        self.size: float = size
        self.position_min: np.ndarray = min_position
        self.normal: np.ndarray = np.zeros(shape=(3,), dtype=np.float64)
        self.residual: float = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def position_max(self) -> np.ndarray:
        return self.position_min + np.full(shape=(3,), fill_value=self.size)

    @property
    def center_position(self) -> np.ndarray:
        return self.position_min + np.full(shape=(3,), fill_value=self.size * 0.5)

    def subdivide(self,
                  leaf_node_list: List,
                  nodes_per_depth: List[List],
                  points: np.ndarray,
                  normals: np.ndarray,
                  full_threshold: int,
                  minimum_voxel_size: float,
                  residual_threshold: float,
                  max_depth: Optional[int] = None):

        if len(self.vertex_indices) >= full_threshold:
            self.compute_normal_and_residual(points, normals)

        if max_depth is not None and self.depth >= max_depth:
            leaf_node_list.append(self)
            return

        if self.residual < residual_threshold:
            leaf_node_list.append(self)
            return

        if len(self.vertex_indices) < full_threshold:
            leaf_node_list.append(self)
            return

        if self.size * 0.5 < minimum_voxel_size:
            leaf_node_list.append(self)
            return

        shifted_points = points[self.vertex_indices] - self.position_min
        new_size = self.size * 0.5
        voxel_indices = np.floor(shifted_points / new_size).astype(int)
        voxel_grid = {}

        # Iterate over each point to determine its voxel index
        for i in range(len(voxel_indices)):
            # Convert voxel index to tuple to use it as a dictionary key
            voxel_index_tuple = tuple(voxel_indices[i])
            # Create a new Voxel object if it doesn't exist already
            if voxel_index_tuple not in voxel_grid:
                min_position = self.position_min + voxel_indices[i] * new_size
                node = RegionGrowingOctreeNode(depth=self.depth + 1, min_position=min_position, size=new_size)
                voxel_grid[voxel_index_tuple] = node
                self.children.append(node)

            # Append the point index to the list of points in the corresponding voxel
            voxel_grid[voxel_index_tuple].vertex_indices.append(self.vertex_indices[i])

        self.vertex_indices = None

        # Store our children in the nodes_per_depth list.
        if len(nodes_per_depth) <= self.depth:
            nodes_per_depth.append(self.children.copy())
        else:
            nodes_per_depth[self.depth].extend(self.children)

        for child in self.children:
            child.subdivide(leaf_node_list, nodes_per_depth, points, normals, full_threshold,
                            minimum_voxel_size, residual_threshold, max_depth)


    def compute_normal_and_residual(self, points, normals):
        relevant_normals = normals[self.vertex_indices]
        mean_normal = np.mean(relevant_normals, axis=0)
        centered_normals = relevant_normals - mean_normal

        # Step 3: Covariance Matrix
        cov_matrix = np.cov(centered_normals, rowvar=False)

        # Step 4: PCA
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Step 5: Extract Normal Vector
        # Find the index of the smallest eigenvalue
        min_eigenvalue_index = np.argmin(eigenvalues)
        self.normal = eigenvectors[:, min_eigenvalue_index]

        summed_squared_distances = 0
        for i in self.vertex_indices:
            vector_to_point = points[i] - self.center_position

            # Calculate dot product of vector to point and normal vector
            dot_product = np.dot(vector_to_point, self.normal)

            # Calculate squared distance
            summed_squared_distances += (dot_product ** 2)

        self.residual = np.sqrt(summed_squared_distances / len(self.vertex_indices))

    def get_corner_points(self) -> np.ndarray:
        points = np.zeros(shape=(8, 3), dtype=np.float64)

        #    5 ---- 6
        #   /|     /|
        #  / |    / |
        # 4 ---- 7  |
        # |  |   |  |
        # |  1 --|- 2
        # | /    | /
        # 0 ---- 3

        points[0] = self.position_min
        points[1] = self.position_min + np.array([0, 0, self.size])
        points[2] = self.position_min + np.array([self.size, 0, self.size])
        points[3] = self.position_min + np.array([self.size, 0, 0])
        points[4] = self.position_min + np.array([0, self.size, 0])
        points[5] = self.position_min + np.array([0, self.size, self.size])
        points[6] = self.position_max
        points[7] = self.position_min + np.array([self.size, self.size, 0])
        return points

    def get_triangles(self) -> np.ndarray:
        triangles = np.zeros(shape=(12, 3, 3), dtype=np.float64)
        corner_points = self.get_corner_points()
        triangles[0] = np.array([corner_points[0], corner_points[1], corner_points[2]])  # Bottom
        triangles[1] = np.array([corner_points[0], corner_points[2], corner_points[3]])
        triangles[2] = np.array([corner_points[0], corner_points[4], corner_points[7]])  # Front
        triangles[3] = np.array([corner_points[0], corner_points[7], corner_points[3]])
        triangles[4] = np.array([corner_points[0], corner_points[1], corner_points[4]])  # Left
        triangles[5] = np.array([corner_points[1], corner_points[4], corner_points[5]])
        triangles[6] = np.array([corner_points[1], corner_points[2], corner_points[5]])  # Back
        triangles[7] = np.array([corner_points[2], corner_points[5], corner_points[6]])
        triangles[8] = np.array([corner_points[2], corner_points[3], corner_points[6]])  # Right
        triangles[9] = np.array([corner_points[3], corner_points[6], corner_points[7]])
        triangles[10] = np.array([corner_points[4], corner_points[5], corner_points[6]])  # Top
        triangles[11] = np.array([corner_points[4], corner_points[6], corner_points[7]])
        return triangles
