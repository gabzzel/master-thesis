from typing import List, Optional

import numpy as np
import open3d
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud
import bisect


class RegionGrowingOctree:
    def __init__(self, point_cloud: PointCloud, root_margin: float = 0.1):
        self.root_node: RegionGrowingOctreeNode
        self.origin_point_cloud = point_cloud
        self._create_root_node(root_margin)
        self.leaf_nodes = []

    def _create_root_node(self, root_margin):
        points = np.asarray(self.origin_point_cloud.points)
        size = 0

        for i in range(3):
            _max = np.max(points[:][i])
            _min = np.min(points[:][i])
            size = max(size, abs(_max - _min) * (1.0 + root_margin))  # Use the maximum size

        min_position = points.min(axis=0) - np.full(shape=(3,), fill_value=0.5 * root_margin)
        self.root_node = RegionGrowingOctreeNode(depth=0, size=size, min_position=min_position)

    def initial_voxelization(self, voxel_size: float, points: bool = True):
        if points:
            self._initial_voxelization_points(voxel_size)
        else:
            self._initial_voxelization_sorted(voxel_size)


    def _initial_voxelization_sorted(self, voxel_size: float):
        original_points = np.asarray(self.origin_point_cloud.points)

        print("Sorting...")
        sorted_by_coordinate: List[np.ndarray] = []  # The array containing a single dimension of the points data, sorted
        sorted_by_coordinate_indices: List[np.ndarray] = []
        for i in range(3):
            sorted_by_coordinate_indices.append(np.argsort(original_points[:, i]))
            sorted_by_coordinate.append(original_points[sorted_by_coordinate_indices[i]][:, i])

        voxel_count = int(np.ceil(self.root_node.size / voxel_size))
        for x in tqdm.trange(voxel_count, desc="Initial voxelization)"):
            min_x = self.root_node.position_min[0] + x * voxel_size
            max_x = min_x + voxel_size
            first_index_x = bisect.bisect_left(sorted_by_coordinate[0], min_x)
            last_index_x = bisect.bisect_right(sorted_by_coordinate[0], max_x, lo=first_index_x)

            if last_index_x <= 0 or first_index_x == last_index_x:
                continue

            search_range_y_indices = sorted_by_coordinate_indices[0][first_index_x:last_index_x]
            search_range_y = sorted_by_coordinate[1][search_range_y_indices]

            for y in range(voxel_count):
                min_y = self.root_node.position_min[1] + y * voxel_size
                max_y = min_y + voxel_size

                first_index_y = bisect.bisect_left(search_range_y, min_y)
                last_index_y = bisect.bisect_right(search_range_y, max_y, lo=first_index_y)

                if last_index_y <= 0 or first_index_y == last_index_y:
                    continue

                search_range_z_indices = sorted_by_coordinate_indices[1][first_index_y:last_index_y]
                search_range_z = sorted_by_coordinate[2][search_range_z_indices]

                for z in range(voxel_count):
                    min_z = self.root_node.position_min[2] + z * voxel_size
                    max_z = min_z + voxel_size

                    first_index_z = bisect.bisect_left(search_range_z, min_z)
                    last_index_z = bisect.bisect_right(search_range_z, max_z, lo=first_index_z)

                    if last_index_z == 0 or first_index_z == last_index_z:
                        continue

                    position = np.array([min_x, min_y, min_z])
                    node = RegionGrowingOctreeNode(depth=1, min_position=position, size=voxel_size)
                    node.vertex_indices = sorted_by_coordinate_indices[2][first_index_z:last_index_z].tolist()
                    self.root_node.children.append(node)


    def _initial_voxelization_points(self, voxel_size: float):
        points = np.asarray(self.origin_point_cloud.points)
        shifted_points = points - self.root_node.position_min
        voxel_grid = {}

        # Calculate the indices of the voxels that each point belongs to
        voxel_indices = np.floor(shifted_points / voxel_size).astype(int)

        voxel_count = 0
        max_vertices_in_a_voxel = 0

        # Iterate over each point to determine its voxel index
        for i in tqdm.trange(len(voxel_indices), unit="points", desc="Initial voxelization"):
            # Convert voxel index to tuple to use it as a dictionary key
            voxel_index_tuple = tuple(voxel_indices[i])
            # Create a new Voxel object if it doesn't exist already
            if voxel_index_tuple not in voxel_grid:
                node = RegionGrowingOctreeNode(depth=1, min_position=voxel_indices[i] * voxel_size, size=voxel_size)
                self.root_node.children.append(node)
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

        node_queue = self.root_node.children.copy()
        points = np.asarray(self.origin_point_cloud.points)
        normals = np.asarray(self.origin_point_cloud.normals)

        progress_bar = tqdm.tqdm(desc="Creating octree", total=len(node_queue), smoothing=1.0, position=0, unit="pts")

        while len(node_queue) > 0:
            current_node: RegionGrowingOctreeNode = node_queue.pop()

            # if current_node.is_region_growing_full:
            #    current_node.compute_normal_and_residual(points=points, normals=normals)
            #    residual_sum += current_node.residual
            #    residual_counts += 1

            # Treat this node as a leaf node if either...
            # 1. Node has not enough vertices to be subdivided
            # 2. The node is too small to be subdivided (i.e. we have reached minimum voxel size)
            # 3. The residual value is below the threshold
            # 4. We have reached max octree depth
            if len(current_node.vertex_indices) < full_threshold or \
                    current_node.size / 2 < minimum_voxel_size or \
                    current_node.residual < residual_threshold or \
                    (max_depth is not None and current_node.depth >= max_depth):
                self.leaf_nodes.append(current_node)

            elif len(current_node.vertex_indices) >= full_threshold:
                current_node.subdivide(points)
                node_queue.extend(current_node.children)
                progress_bar.total += len(current_node.children)

            progress_bar.update()

        print(f"Octree generation complete, total leaf nodes: {len(self.leaf_nodes)} ")

    def visualize_voxels(self):
        nodes = self.leaf_nodes if len(self.leaf_nodes) > 0 else self.root_node.children

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
            for node_index in tqdm.trange(len(nodes), desc="Creating boxes", unit="voxel"):
                node = nodes[node_index]
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
