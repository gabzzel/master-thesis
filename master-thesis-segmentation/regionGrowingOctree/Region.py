from typing import Set

import numpy as np

from regionGrowingOctree.RegionGrowingOctreeNode import RegionGrowingOctreeNode


class Region:
    def __init__(self, index):
        self.index = index
        self.nodes: Set[RegionGrowingOctreeNode] = set()
        self.vertex_indices = set()
        self._vertex_indices_ndarray: np.ndarray = None
        self.area: float = 0.0

        self.centroid: np.ndarray = None
        self.normal: np.ndarray = None

    @property
    def vertex_indices_array(self) -> np.ndarray:
        if self._vertex_indices_ndarray is None:
            self._vertex_indices_ndarray = np.array(list(self.vertex_indices), dtype=np.int32)
        return self._vertex_indices_ndarray

    def union(self, other: Set[RegionGrowingOctreeNode]):
        self.nodes.union(other)
        for node in other:
            self.area += node.size ** 3
            self.vertex_indices = self.vertex_indices.union(node.vertex_indices)

    def add(self, node: RegionGrowingOctreeNode):
        self.nodes.add(node)
        self.area += node.size ** 3
        self.vertex_indices = self.vertex_indices.union(node.vertex_indices)

    @property
    def node_count(self):
        return len(self.nodes)

    def is_planar(self, points, normals, amount_threshold: float = 0.9, distance_threshold: float = 0.01):
        relevant_points = points[self.vertex_indices_array]
        self.centroid = np.mean(relevant_points, axis=0)
        relevant_normals = normals[self.vertex_indices_array]
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

        # We need to have at most this amount of points too far from the plane.
        tolerance_count = np.floor(len(relevant_points) * (1 - amount_threshold))

        for i in self.vertex_indices:
            distance = self.distance_to_point(points[i])

            # If the distance is too large, deduct 1 from the tolerance.
            if distance > distance_threshold:
                tolerance_count -= 1
                # If the tolerance is depleted, we are for sure not planar.
                if tolerance_count <= 0:
                    return False
        return True

    def distance_to_point(self, point):
        vector_to_point = point - self.centroid
        distance = abs(np.dot(vector_to_point, self.normal))
        return distance
