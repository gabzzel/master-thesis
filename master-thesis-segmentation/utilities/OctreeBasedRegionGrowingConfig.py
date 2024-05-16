import json
from os import PathLike
from pathlib import Path
from typing import Union, List


class OctreeBasedRegionGrowingConfig:
    def __init__(self):
        self.data_set: str = ""

        # --------------- Subdivision --------------- #
        # Octree-based region growing first voxelizes the input before creating the octree. This parameter controls
        # the size of these initial voxels.
        self.initial_voxel_size: float = 0.1

        # During region growing, the minimum amount of octree nodes that need to be in a segment for the segment
        # to be considered valid.
        self.minimum_valid_segment_size: int = 20

        # During subdivision, i.e. octree creation, the minimum residual value to consider the octree node for
        # subdivision.
        self.subdivision_residual_threshold: float = 0.001

        # How many points must be in the octree node at a minimum to consider the node for subdivision.
        # The original authors used 4.
        self.subdivision_full_threshold: int = 4

        # The minimum size of an octree node during subdivision. The size of all octree nodes will thus be equal
        # to larger than this.
        self.subdivision_minimum_voxel_size: float = 0.01

        # --------------- Region Growing --------------- #
        # The quantile of the residuals of the octree leaf nodes that determines the actual threshold. Octree leaf
        # nodes with a residual above the resulting threshold will not be considered as seeds (for segments) for
        # region growing.
        self.region_growing_residual_threshold: float = 0.95

        # The maximum angular deviation between normals (in degrees) of neighbouring octree nodes for them to be
        # joined with a segment. Used during growing of regions over octree leaf nodes.
        self.growing_normal_deviation_threshold_degrees: float = 90.0

        # --------------- REFINEMENT --------------- #
        # The maximum angular deviation between neighbouring points for them to join the region during
        # (general) refinement. Used on points, not segments or nodes!
        self.refining_normal_deviation_threshold_degrees: float = 30.0

        # The minimum fraction of points within a region that must be within 'refining_planar_distance_threshold'
        # from the plane defined by the centroid and normal of the region for the region to be eligible for
        # fast refinement.
        self.fast_refinement_planar_amount_threshold: float = 0.8

        # The maximum distance of points to the plane (defined by the normal and centroid of a target region)
        # for them to be considered for checking eligibility of fast refinement for said target region.
        self.fast_refinement_planar_distance_threshold: float = 0.01

        # The maximum distance a point can be to the plane (defined by the normal and centroid) of a region for
        # that point to be joined with the region.
        self.fast_refinement_distance_threshold: float = 0.05

        # The buffer around the boundary nodes of a region / segment in which points will be considered
        # during general refinement.
        self.general_refinement_buffer_size: float = 0.02

        self.total_time: float = 0.0
        self.weighted_IoU: float = 0.0
        self.average_IoU_per_class: float = 0.0
        self.noise_points: int = 0

    def get_results_header(self) -> List[str]:
        return ["total_time","weighted_IoU","average_IoU_per_class","noise_points"]

    def write_results_to_file(self, file_path: Union[str, PathLike]):
        with open(file_path, mode="a") as file_path:
            file_path.write("TotalTime;WeightedIoU;AverageIoU_per_class;NoisePoints\n")

            file_path.write(";".join([str(self.total_time),
                                      str(self.weighted_IoU),
                                      str(self.average_IoU_per_class),
                                      str(self.noise_points)]))


def read_from_file(path: Union[str, PathLike]) -> OctreeBasedRegionGrowingConfig:
    path = Path(path)
    assert path.suffix == ".json"

    result = OctreeBasedRegionGrowingConfig()

    with open(file=path, mode="r+") as file:
        data = json.load(file)
        for attr in result.__dir__():
            if attr in data:
                result.__setattr__(attr, data[attr])

    return result


def write_results_to_file_multiple(file_path: Union[str, PathLike],
                                   results: List[OctreeBasedRegionGrowingConfig]):

    with open(file_path, mode="a") as file:
        dummy = OctreeBasedRegionGrowingConfig()
        for attr in dummy.__dir__():
            file.write(f"{attr};")
        file.write("\n")

        for result in results:
            for attr in result.__dir__():
                file.write(f"{result.__getattribute__(attr)};")
            file.write("\n")