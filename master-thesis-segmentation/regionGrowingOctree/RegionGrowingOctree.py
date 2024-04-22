import bisect
import cProfile
from typing import List, Optional, Dict, Set

import numpy as np
import open3d
import pymorton
import scipy.linalg
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud
from scipy.spatial.distance import cdist

from regionGrowingOctree.Region import Region
from regionGrowingOctree.RegionGrowingOctreeNode import RegionGrowingOctreeNode


class RegionGrowingOctree:
    def __init__(self, point_cloud: PointCloud, root_margin: float = 0.1):
        self.root_node: Optional[RegionGrowingOctreeNode] = None

        # A list whose indices are depths, values are dictionaries that contain leaf nodes by coordinate
        self.leaf_nodes: List[Dict[tuple, RegionGrowingOctreeNode]] = []
        self.leaf_node_count = 0

        # A list of nodes per depth level. Elements in this list are lists that contain all nodes at this level.
        self.nodes_per_depth: List[List[RegionGrowingOctreeNode]] = []

        self.origin_point_cloud = point_cloud
        self._create_root_node(root_margin)
        self.segments: List[Region] = []
        self.initial_voxel_size: float = 0

        self.one_offsets = None
        self.segment_index_per_point = np.full(shape=(len(point_cloud.points),), fill_value=-1, dtype=np.int32)

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
        self.initial_voxel_size = voxel_size
        # with cProfile.Profile() as pr:
        points = np.asarray(self.origin_point_cloud.points)
        shifted_points = points - self.root_node.position_min
        voxel_grid: Dict[tuple, RegionGrowingOctreeNode] = {}

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

            # pr.print_stats(sort='cumtime')
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
                            max_depth: Optional[int] = None,
                            profile: bool = False):

        pr = None
        if profile:
            pr = cProfile.Profile()
            pr.enable()

        points = np.asarray(self.origin_point_cloud.points)
        normals = np.asarray(self.origin_point_cloud.normals)

        residual_sum: float = 0.0
        residual_counts: int = 0

        for i in tqdm.trange(len(self.root_node.children), desc="Creating octree"):
            node = self.root_node.children[i]
            node.subdivide(octree=self,
                           points=points,
                           normals=normals,
                           full_threshold=full_threshold,
                           minimum_voxel_size=minimum_voxel_size,
                           residual_threshold=residual_threshold,
                           max_depth=max_depth)

            residual_sum += node.residual
            residual_counts += int(node.residual > 0)

        if profile:
            pr.disable()
            pr.print_stats(sort='cumulative')

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
                     normal_deviation_threshold_degrees: float,
                     minimum_valid_segment_size: int,
                     profile: bool = False,
                     residual_threshold_is_absolute: bool = True):

        if not residual_threshold_is_absolute:
            residual_threshold = max(min(residual_threshold, 1.0), 0.0)
            all_residuals = []
            for i in self.leaf_nodes:
                for node in i.values():
                    all_residuals.append(node.residual)
            residual_threshold = np.quantile(all_residuals, q=residual_threshold)

        pr: cProfile.Profile = None
        if profile:
            pr = cProfile.Profile()
            pr.enable()

        self.segments = []

        nodes_to_do: List[RegionGrowingOctreeNode] = []
        removed_nodes = set()

        for depth in range(0, len(self.leaf_nodes)):
            nodes_to_do.extend(self.leaf_nodes[depth].values())

        # Sorts ascending by default, but we want descending, since we will pop at the end.
        nodes_to_do.sort(reverse=True, key=lambda x: x.residual)

        original_length = len(nodes_to_do)
        progress_bar = tqdm.tqdm(total=original_length, unit="voxel")
        normal_deviation_threshold = np.cos(np.deg2rad(normal_deviation_threshold_degrees))

        while len(nodes_to_do) > 0:
            v_min = nodes_to_do.pop()
            progress_bar.update()

            if v_min.residual > residual_threshold:
                break

            if v_min in removed_nodes:
                continue

            removed_nodes.add(v_min)

            current_region: Region = Region(len(self.segments))
            current_region.add(v_min)
            v_min.region = current_region
            current_seed = [v_min]

            i = 0
            while i < len(current_seed):
                v_i = current_seed[i]
                i += 1

                B_c = self.get_neighboring_leaf_nodes(v_i)
                for v_j in B_c:

                    if v_j in removed_nodes:  # We will not process removed nodes.
                        continue

                    removed_nodes.add(v_j)  # Mark this node as removed (it's not actually removed, only marked!)
                    progress_bar.update()

                    # Rabbani et al. 2007: "As the direction of normal vector has a 180 degree ambiguity we have to
                    # take the absolute value of the dot product."
                    theta = abs(np.dot(v_i.normal, v_j.normal))
                    if theta >= normal_deviation_threshold:
                        current_region.add(v_j)
                        v_j.region = current_region

                        if v_j.residual < residual_threshold:
                            current_seed.append(v_j)

            if current_region.node_count > minimum_valid_segment_size:
                self.segments.append(current_region)
            else:
                for n in current_region.nodes:
                    n.region = None

        self.segments.sort(key=lambda x: x.area)

        if pr is not None:
            pr.disable()
            pr.print_stats(sort="cumulative")

        return self.segments

    def get_neighboring_leaf_nodes(self, target_node: RegionGrowingOctreeNode,
                                   buffer: int = 1,
                                   exclude_allocated_nodes: bool = False) -> List[RegionGrowingOctreeNode]:
        """
        return all voxels from that share at least a vertex, an edge or a face with the target node
        """
        offsets = self.get_offsets(buffer)

        result: List[RegionGrowingOctreeNode] = []

        # This calculation is not valid at depth 0
        for depth in range(1, len(self.leaf_nodes)):
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

    def refine_regions(self,
                       planar_amount_threshold: float = 0.9,
                       planar_distance_threshold: float = 0.001,
                       fast_refinement_distance_threshold: float = 0.001,
                       buffer_zone_size: float = 0.02,
                       angular_divergence_threshold_degrees: float = 15):
        """
        Refine the regions to get more details.

        :param buffer_zone_size: The search radius around the boundaries of the region in which points will be \
            considered during general refinement.
        :param angular_divergence_threshold_degrees: The maximum angular divergence between points' normals \
            for points to be added to the region / segment during general refinement, in degrees.
        :param fast_refinement_distance_threshold: The distance threshold for points (to a regions' best fitting \
            plane) to be merged with a segment / region during fast refinement
        :param planar_amount_threshold: The fraction of points in a region that need to be within distance of the \
            the best fitting plane in order to consider the region to be planar.
        :param planar_distance_threshold: The maximum distance a point can be from the best fitting plane of a \
             region / segment and still contribute to the "planar-ness" of the segment.
        :return:
        """

        points = np.asarray(self.origin_point_cloud.points)
        normals = np.asarray(self.origin_point_cloud.normals)

        planar_count = 0

        for segment_index in tqdm.trange(len(self.segments), desc="Refining regions/segments", unit="segment"):
            segment = self.segments[segment_index]
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
                self.fast_refinement(boundary_nodes,
                                     fast_refinement_distance_threshold,
                                     points,
                                     segment)
                planar_count += 1
            # General refinement.
            else:
                adtr = np.deg2rad(angular_divergence_threshold_degrees)
                self.general_refinement(segment=segment,
                                        boundary_nodes=boundary_nodes,
                                        buffer_zone_size=buffer_zone_size,
                                        nearest_neighbours=20,
                                        angular_divergence_threshold_radians=adtr,
                                        points=points,
                                        normals=normals)
        print(f"Completed refining. Used fast refining {planar_count}/{len(self.segments)} times")

    def get_boundary_nodes(self, segment) -> List[RegionGrowingOctreeNode]:
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

    def fast_refinement(self, boundary_nodes: List[RegionGrowingOctreeNode],
                        fast_refinement_distance_threshold: float,
                        points: np.ndarray,
                        segment,
                        copy_boundary_nodes: bool = False):

        assert points.ndim == 2
        assert points.shape[1] == 3

        S = boundary_nodes.copy() if copy_boundary_nodes else boundary_nodes
        while len(S) > 0:
            v_j = S.pop()
            B = self.get_neighboring_leaf_nodes(v_j, exclude_allocated_nodes=True)
            for v_k in B:
                for p_l in v_k.vertex_indices:
                    if segment.distance_to_point(points[p_l]) > fast_refinement_distance_threshold:
                        continue

                    if self.segment_index_per_point[p_l] != -1:
                        continue

                    self.segment_index_per_point[p_l] = segment.index
                    segment.vertex_indices.add(p_l)

    def general_refinement(self,
                           segment: Region,
                           boundary_nodes: List[RegionGrowingOctreeNode],
                           buffer_zone_size: float,
                           nearest_neighbours: int,
                           angular_divergence_threshold_radians: float,
                           points: np.ndarray,
                           normals: np.ndarray):
        seed_indices = set()
        other_indices = set()
        angular_divergence_threshold = np.cos(angular_divergence_threshold_radians)

        for node in boundary_nodes:
            seed_indices = seed_indices.union(node.vertex_indices)
            buffer = int(np.ceil(buffer_zone_size / node.size))
            for neighbor in self.get_neighboring_leaf_nodes(node, buffer, exclude_allocated_nodes=True):
                other_indices = other_indices.union(neighbor.vertex_indices)

        other_indices = other_indices.difference(seed_indices)  # We don't want to include the seeds.
        seed_indices = np.array(list(seed_indices))
        other_indices = np.array(list(other_indices))

        if len(other_indices) == 0:
            return

        all_seeds_to_all_other_distances = cdist(points[seed_indices], points[other_indices])

        for i, seed_index in enumerate(seed_indices):

            # The distance to all "other" points
            distances_seed_to_others = all_seeds_to_all_other_distances[i]

            # The indices of the other points if they would be sorted on distance to the current seed
            relevant_distances_sorted_indices = np.argsort(distances_seed_to_others)

            for j in range(min(len(relevant_distances_sorted_indices), nearest_neighbours)):

                # The index of the distance in the relevant_distances list.
                distance_index = relevant_distances_sorted_indices[j]

                # The actual distance
                distance = distances_seed_to_others[distance_index]

                if distance > buffer_zone_size:
                    continue

                other_index = other_indices[distance_index]

                normal1 = normals[seed_index]
                normal2 = normals[other_index]
                angular_divergence = np.abs(np.dot(normal1, normal2))

                # The divergence should be larger or equal to the threshold.
                if angular_divergence < angular_divergence_threshold:
                    continue

                segment.vertex_indices.add(distance_index)

