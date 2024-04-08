import queue
from typing import List

import open3d
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud
import numpy as np


class RegionGrowingOctree:
    def __init__(self,
                 point_cloud: PointCloud,
                 root_margin: float = 0.1,
                 initial_voxel_size: float = 0.01,
                 minimum_voxel_size: float = 0.0001,
                 residual_threshold: float = 0):

        self.root_node: RegionGrowingOctreeNode
        self.origin_point_cloud = point_cloud
        self._create_root_node(root_margin)
        self.initial_voxelization(initial_voxel_size)
        self.recursive_subdivide(minimum_voxel_size)
        x = 0

    def _create_root_node(self, root_margin):
        points = np.asarray(self.origin_point_cloud.points)
        size = 0

        for i in range(3):
            _max = np.max(points[:][i])
            _min = np.min(points[:][i])
            size = max(size, abs(_max - _min) * (1.0 + root_margin))  # Use the maximum size

        min_position = points.min(axis=0) - np.full(shape=(3,), fill_value=0.5 * root_margin)
        self.root_node = RegionGrowingOctreeNode(depth=0, size=size, min_position=min_position)

    def initial_voxelization(self, voxel_size: float):
        points = np.asarray(self.origin_point_cloud.points)
        shifted_points = points - self.root_node.position_min
        voxel_grid = {}

        # Calculate the indices of the voxels that each point belongs to
        voxel_indices = np.floor(shifted_points / voxel_size).astype(int)

        # Iterate over each point to determine its voxel index
        for i in tqdm.trange(len(voxel_indices), unit="points", desc="Initial voxelization"):
            # Convert voxel index to tuple to use it as a dictionary key
            voxel_index_tuple = tuple(voxel_indices[i])
            # Create a new Voxel object if it doesn't exist already
            if voxel_index_tuple not in voxel_grid:
                node = RegionGrowingOctreeNode(depth=1, min_position=voxel_indices[i] * voxel_size, size=voxel_size)
                self.root_node.children.append(node)
                voxel_grid[voxel_index_tuple] = node

            # Append the point index to the list of points in the corresponding voxel
            voxel_grid[voxel_index_tuple].vertex_indices.append(i)

    def recursive_subdivide(self, minimum_voxel_size):
        node_queue = self.root_node.children.copy()
        points = np.asarray(self.origin_point_cloud.points)

        while len(node_queue) > 0:
            current_node: RegionGrowingOctreeNode = node_queue.pop()

            # Node has not enough vertices to be subdivided
            if current_node.is_region_growing_empty:
                continue

            # The node is too small to be subdivided
            if current_node.size / 2 < minimum_voxel_size:
                continue

            current_node.subdivide(points)
            node_queue.extend(current_node.children)


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

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_region_growing_full(self) -> bool:
        return len(self.vertex_indices) > 3

    @property
    def is_region_growing_empty(self) -> bool:
        return not self.is_region_growing_full

    @property
    def position_max(self) -> np.ndarray:
        return self.position_min + np.full(shape=(3, ), fill_value=self.size)

    @property
    def center_position(self) -> np.ndarray:
        return self.position_min + np.full(shape=(3, ), fill_value=self.size * 0.5)

    def subdivide(self, points: np.ndarray):
        shifted_points = points[self.vertex_indices] - self.position_min
        new_size = self.size / 2.0
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