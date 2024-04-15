import bisect
import cProfile
from typing import List, Optional, Set, Dict

import numpy as np
import open3d
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud

from regionGrowingOctree.RegionGrowingOctreeNode import RegionGrowingOctreeNode


class RegionGrowingOctree:
    def __init__(self, point_cloud: PointCloud, root_margin: float = 0.1):
        self.root_node: Optional[RegionGrowingOctreeNode] = None

        # A list whose indices are depths, values are dictionaries that contain leaf nodes by coordinate
        self.leaf_nodes: List[Dict[tuple, RegionGrowingOctreeNode]] = []

        # A list of nodes per depth level. Elements in this list are lists that contain all nodes at this level.
        self.nodes_per_depth: List[List[RegionGrowingOctreeNode]] = []

        self.origin_point_cloud = point_cloud
        self._create_root_node(root_margin)
        self.segments: List[Region] = None

        self.one_offsets = None

    def _create_root_node(self, root_margin):
        points = np.asarray(self.origin_point_cloud.points)
        size = 0

        for i in range(3):
            _max = np.max(points[:][i])
            _min = np.min(points[:][i])
            size = max(size, abs(_max - _min) * (1.0 + root_margin))  # Use the maximum size

        min_position = points.min(axis=0) - np.full(shape=(3,), fill_value=size * root_margin * 0.5)
        self.root_node = RegionGrowingOctreeNode(depth=0,
                                                 local_index=np.array([0, 0, 0]),
                                                 global_index=np.array([0, 0, 0]),
                                                 size=size, min_position=min_position)
        self.nodes_per_depth.append([self.root_node])

    def initial_voxelization(self, voxel_size: float):
        self._initial_voxelization_points(voxel_size)

    def _initial_voxelization_points(self, voxel_size: float):

        #with cProfile.Profile() as pr:
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

            voxel_index_tuple = tuple(voxel_indices[i])
            # Create a new Voxel object if it doesn't exist already
            if voxel_index_tuple not in voxel_grid:
                node = self._create_initial_voxel(local_voxel_index=voxel_indices[i], voxel_size=voxel_size)
                voxel_grid[voxel_index_tuple] = node
                voxel_count += 1

            # Append the point index to the list of points in the corresponding voxel
            voxel: RegionGrowingOctreeNode = voxel_grid[voxel_index_tuple]
            voxel.vertex_indices.append(i)
            max_vertices_in_a_voxel = max(max_vertices_in_a_voxel, len(voxel.vertex_indices))

            #pr.print_stats(sort='cumtime')
        print(f"Created {voxel_count} voxels of size {voxel_size}")
        print(f"Vertices per voxel: avg {len(points) / voxel_count} , max {max_vertices_in_a_voxel}")

    def _create_initial_voxel(self, local_voxel_index: np.ndarray, voxel_size):
        pos = self.root_node.position_min + local_voxel_index * voxel_size
        node = RegionGrowingOctreeNode(depth=1,
                                       local_index=local_voxel_index,
                                       global_index=local_voxel_index,
                                       min_position=pos,
                                       size=voxel_size)
        self.root_node.children.append(node)
        self.nodes_per_depth[1].append(node)
        return node

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
            node.subdivide(leaf_nodes=self.leaf_nodes,
                           nodes_per_depth=self.nodes_per_depth,
                           points=points,
                           normals=normals,
                           full_threshold=full_threshold,
                           minimum_voxel_size=minimum_voxel_size,
                           residual_threshold=residual_threshold,
                           max_depth=max_depth)

            residual_sum += node.residual
            residual_counts += int(node.residual > 0)

        print(f"Octree generation complete, total leaf nodes: {sum(len(i) for i in self.leaf_nodes)} ")
        print(f"Residual average: {residual_sum / residual_counts}")

    def visualize_voxels(self, segments: list, maximum: Optional[int] = None, ):
        if maximum is None:
            maximum = sum(len(i) for i in self.leaf_nodes)

        colors = {}
        rng = np.random.default_rng()
        for segment in segments:
            colors[segment] = rng.random(size=(3,))

        meshes = []

        progress_bar = tqdm.tqdm(total=len(segments), unit="segments")
        for segment in segments:
            progress_bar.update()
            for node in segment.nodes:
                if len(meshes) >= maximum:
                    break

                mesh: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_box(width=node.size,
                                                                                             height=node.size,
                                                                                             depth=node.size)
                mesh.translate(node.position_min)
                mesh.paint_uniform_color(colors[node.region])
                meshes.append(mesh)

        open3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)

    def grow_regions(self,
                     residual_threshold: float,
                     normal_deviation_threshold_radians: float,
                     minimum_valid_segment_size: int):
        self.segments = []

        A: List[RegionGrowingOctreeNode] = []
        for depth in range(0, len(self.leaf_nodes)):
            A.extend(self.leaf_nodes[depth].values())

        A.sort(reverse=False, key=lambda x: x.residual)
        last_index = bisect.bisect_left(a=A, x=residual_threshold, key=lambda node: node.residual)
        A = A[:last_index]

        original_length = len(A)
        progress_bar = tqdm.tqdm(total=original_length, unit="voxel")

        while len(A) > 0:
            v_min = A.pop()
            progress_bar.update()
            if v_min.residual > residual_threshold:
                break

            current_region: Region = Region()
            current_region.add(v_min)
            v_min.region = current_region
            current_seed = [v_min]

            i = 0
            while i < len(current_seed):
                v_i = current_seed[i]
                i += 1

                B_c = self.get_neighboring_leaf_nodes(v_i)
                for v_j in B_c:
                    theta = np.arccos(np.clip(np.dot(v_i.normal, v_j.normal), a_min=-1.0, a_max=1.0))
                    try:
                        v_j_index = A.index(v_j)
                        # We know v_j is in A, else it will throw an error
                        if theta <= normal_deviation_threshold_radians:
                            current_region.add(v_j)
                            v_j.region = current_region
                            A.pop(v_j_index)
                            progress_bar.update()
                            if v_j.residual < residual_threshold:
                                current_seed.append(v_j)

                    # v_j is not in A
                    except ValueError:
                        pass

            if current_region.node_count > minimum_valid_segment_size:
                self.segments.append(current_region)
            else:
                for n in current_region.nodes:
                    n.region = None

        self.segments.sort(key=lambda x: x.area)
        return self.segments

    def get_neighboring_leaf_nodes(self, target_node: RegionGrowingOctreeNode,
                                   buffer: int = 1,
                                   exclude_allocated_nodes:bool = False) -> List[RegionGrowingOctreeNode]:
        """
        return all voxels from that share at least a vertex, an edge or a face with the target node
        """
        offsets = self.get_offsets(buffer)

        result: List[RegionGrowingOctreeNode] = []

        for depth in range(0, len(self.leaf_nodes)):
            for offset in offsets:
                neighbour_coordinate = target_node.global_index + offset
                neighbour_coordinate = tuple(np.floor(neighbour_coordinate / (2 ** (target_node.depth - depth))))

                if neighbour_coordinate not in self.leaf_nodes[depth]:
                    continue

                candidate = self.leaf_nodes[depth][neighbour_coordinate]
                if exclude_allocated_nodes and candidate.region is not None:
                    continue

                result.append(candidate)

        return result

    def get_offsets(self, buffer):
        if buffer == 1 and self.one_offsets is not None:
            return self.one_offsets

        offsets = []
        for x in range(-buffer, buffer, 1):
            for y in range(-buffer, buffer, 1):
                for z in range(-buffer, buffer, 1):
                    if x == y == z == 0:
                        continue
                    offsets.append(np.array([x, y, z]))
        if buffer == 1:
            self.one_offsets = offsets
        return offsets

    def show_point_cloud_with_segment_color(self):
        segment_colors = {}
        rng = np.random.default_rng()
        for segment in self.segments:
            segment_colors[segment] = rng.random(size=(3,))

        original_points = np.asarray(self.origin_point_cloud.points)
        points = []
        colors = []
        progress_bar = tqdm.tqdm(total=len(self.segments), unit="segment")
        for segment in self.segments:
            progress_bar.update()
            for node in segment.nodes:
                for point_index in node.vertex_indices:
                    points.append(original_points[point_index])
                    colors.append(segment_colors[segment])

        pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
        pcd.colors = open3d.utility.Vector3dVector(colors)
        open3d.visualization.draw_geometries([pcd])

    def refine_regions(self, points, normals,
                       planar_amount_threshold: float = 0.9,
                       planar_distance_threshold: float = 0.001,
                       fast_refinement_distance_threshold: float = 0.001):
        """
        Refine the regions to get more details.

        :param fast_refinement_distance_threshold: The distance threshold for points (to a regions' best fitting \
            plane) to be merged with a segment / region during fast refinement
        :param points: The points from the point cloud
        :param normals: The normals from the point cloud
        :param planar_amount_threshold: The fraction of points in a region that need to be within distance of the \
            the best fitting plane in order to consider the region to be planar.
        :param planar_distance_threshold: The maximum distance a point can be from the best fitting plane to be \
            considered planar enough.
        :return:
        """

        allocated_points = set()

        for segment in self.segments:
            boundary_nodes = self.get_boundary_nodes(segment)

            # 2. Find points in clusters vicinity
            # TODO

            # 3. Check planarity
            is_planar = segment.is_planar(points, normals,
                                          amount_threshold=planar_amount_threshold,
                                          distance_threshold=planar_distance_threshold)


            # Step B.1a, generates a list of all boundary voxels on the boundary of R0 i, called Vb.
            # Initially all voxels in Vb are added to a set of seed voxels, S.
            # For each voxel vj in S, every unallocated neighbor v k of vj are examined.
            # If vk contains a point pl, whose distance to the fitting plane wi is smaller than a distance threshold,
            # then the point is merged into segment R0 i and vk is added into S for further iterations.
            # 4. Do fast refinement.
            if is_planar:
                self.fast_refinement(allocated_points,
                                     boundary_nodes,
                                     fast_refinement_distance_threshold,
                                     points,
                                     segment)
            # General refinement.
            else:
                # TODO
                pass

    def fast_refinement(self, allocated_points, boundary_nodes, fast_refinement_distance_threshold, points, segment):
        S = boundary_nodes.copy()
        while len(S) > 0:
            v_j = S.pop()
            B = self.get_neighboring_leaf_nodes(v_j, exclude_allocated_nodes=True)
            for v_k in B:
                for p_l in v_k.vertex_indices:
                    if segment.distance_to_point(points[p_l]) > fast_refinement_distance_threshold:
                        continue

                    point_key = tuple(points[p_l])
                    if point_key in allocated_points:
                        continue

                    allocated_points.add(point_key)
                    segment.vertex_indices.add(p_l)

    def get_boundary_nodes(self, segment):
        boundary_nodes = []
        # 1. Extract boundary voxels
        for node in segment.nodes:
            neighbour_count = 0
            for neighbour in self.get_neighboring_leaf_nodes(node):
                if neighbour.region == node.region:
                    neighbour_count += 1
            if neighbour_count < 8:
                boundary_nodes.append(node)
        return boundary_nodes


class Region:
    def __init__(self):
        self.nodes: Set[RegionGrowingOctreeNode] = set()
        self.vertex_indices = set()
        self.area: float = 0.0

        self.centroid: np.ndarray = None
        self.normal: np.ndarray = None

    def union(self, other: Set[RegionGrowingOctreeNode]):
        self.nodes.union(other)
        for node in other:
            self.area += node.size ** 3
            self.vertex_indices.union(node.vertex_indices)

    def add(self, node: RegionGrowingOctreeNode):
        self.nodes.add(node)
        self.area += node.size ** 3
        self.vertex_indices.union(node.vertex_indices)

    @property
    def node_count(self):
        return len(self.nodes)

    def is_planar(self, points, normals, amount_threshold: float = 0.9, distance_threshold:float = 0.01):
        relevant_points = points[self.vertex_indices]
        self.centroid = np.mean(relevant_points, axis=0)
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

        # We need to have at least this amount of points close to the plane.
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