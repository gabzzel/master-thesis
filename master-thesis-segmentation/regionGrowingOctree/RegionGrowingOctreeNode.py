from typing import List, Optional, Dict

import numpy as np


class RegionGrowingOctreeNode:
    def __init__(self,
                 depth: int,
                 local_index: np.ndarray,
                 global_index: np.ndarray,
                 min_position: np.ndarray,
                 size: float):

        self.depth: int = depth

        # The index of this node within its parent. For example (0, 1, 1)
        # Can contain other indices than 0 and 1 if depth is 1, because of initial voxelization
        self.local_index: np.ndarray = local_index

        # The index of this node globally, assuming all nodes exist at this depth.
        # Only valid at the depth of this node.
        self.global_index: np.ndarray = global_index
        self.global_index_tuple: tuple = tuple(global_index)

        self.children: List[RegionGrowingOctreeNode] = []
        self.vertex_indices: List[int] = []

        self.size: float = size
        self.position_min: np.ndarray = min_position
        self.normal: np.ndarray = np.zeros(shape=(3,), dtype=np.float64)
        self.residual: float = 0
        self.region = None

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
                  leaf_nodes: List[Dict],
                  nodes_per_depth: List[List],
                  points: np.ndarray,
                  normals: np.ndarray,
                  full_threshold: int,
                  minimum_voxel_size: float,
                  residual_threshold: float,
                  max_depth: Optional[int] = None):

        self.compute_normal_and_residual(points, normals)

        if (max_depth is not None and self.depth >= max_depth) or \
                self.residual < residual_threshold or \
                len(self.vertex_indices) < full_threshold or \
                self.size * 0.5 < minimum_voxel_size:

            while len(leaf_nodes) <= self.depth:
                leaf_nodes.append({})

            leaf_nodes[self.depth][tuple(self.global_index)] = self
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
                min_position: np.ndarray = self.position_min + voxel_indices[i] * new_size
                global_index: np.ndarray = self.global_index * 2 + voxel_indices[i]
                node = RegionGrowingOctreeNode(depth=self.depth + 1, min_position=min_position, size=new_size,
                                               local_index=voxel_indices[i], global_index=global_index)
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
            child.subdivide(leaf_nodes=leaf_nodes, nodes_per_depth=nodes_per_depth, points=points, normals=normals,
                            full_threshold=full_threshold, minimum_voxel_size=minimum_voxel_size,
                            residual_threshold=residual_threshold, max_depth=max_depth)

    def compute_normal_and_residual(self, points, normals):
        if len(self.vertex_indices) == 0:
            self.normal = np.zeros(shape=(3,), dtype=np.float64)
            self.residual = 0
            return

        if len(self.vertex_indices) == 1:
            self.normal = normals[self.vertex_indices[0]]
            self.residual = 0
            return

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
        self.normal /= np.linalg.norm(self.normal)

        summed_squared_distances = 0
        for i in self.vertex_indices:
            vector_to_point = points[i] - self.center_position

            # Calculate dot product of vector to point and normal vector
            dot_product = abs(np.dot(vector_to_point, self.normal))

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
