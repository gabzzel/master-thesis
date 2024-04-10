from typing import List, Optional, Set

import numpy as np
import open3d
import tqdm
from open3d.cpu.pybind.geometry import PointCloud as PointCloud

from regionGrowingOctree.RegionGrowingOctreeNode import RegionGrowingOctreeNode


class RegionGrowingOctree:
    def __init__(self, point_cloud: PointCloud, root_margin: float = 0.1):
        self.root_node: Optional[RegionGrowingOctreeNode] = None
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

    def grow_regions(self,
                     residual_threshold: float,
                     normal_deviation_threshold_radians: float,
                     minimum_valid_segment_size: int):
        segments: List[Region] = []

        A = sorted(self.leaf_nodes, reverse=True, key=lambda x: x.residual)

        while len(A) > 0:
            current_region: Region = Region()
            current_seed = set()
            v_min = A.pop()
            if v_min > residual_threshold:
                break

            current_region.add(v_min)
            current_seed.add(v_min)

            for v_i in current_seed:
                B_c = self.get_neighbors(v_i)
                for v_j in B_c:
                    theta = np.arccos(np.dot(v_i.normal, v_j.normal))
                    try:
                        v_j_index = A.index(v_j)
                        # We know v_j is in A, else it will throw an error
                        if theta <= normal_deviation_threshold_radians:
                            current_region.add(v_j)
                            A.pop(v_j_index)
                            if v_j.residual < residual_threshold:
                                current_seed.add(v_j)

                    # v_j is not in A
                    except ValueError:
                        continue

                if current_region.node_count > minimum_valid_segment_size:
                    segments.append(current_region)
                else:
                    A.extend(current_region.nodes)
                    A.sort(key=lambda x: x.residual, reverse=True)

        segments.sort(key=lambda x: x.area)
        return segments

    def get_neighbors(self, node: RegionGrowingOctreeNode) -> List[RegionGrowingOctreeNode]:
        """
        return all voxels from O that share at least a vertex, an edge or a face with v
        """
        return


class Region:
    def __init__(self):
        self.nodes: Set[RegionGrowingOctreeNode] = set()
        self.area: float = 0.0

    def union(self, other: Set[RegionGrowingOctreeNode]):
        self.nodes.union(other)
        for node in other:
            self.area += node.size ** 3

    def add(self, node: RegionGrowingOctreeNode):
        self.nodes.add(node)
        self.area += node.size ** 3

    @property
    def node_count(self):
        return len(self.nodes)
