from pathlib import Path
from typing import Optional, Set, Dict, Any

from utilities.enumerations import DownSampleMethod as DSM
from utilities.enumerations import MeshCleaningMethod as MCM
from utilities.enumerations import MeshEvaluationMetric as MEM
from utilities.enumerations import SurfaceReconstructionMethod as SRM
from utilities.enumerations import SurfaceReconstructionParameters as SRP
from utilities.enumerations import TriangleNormalDeviationMethod as TNDM


class RunConfiguration:
    def __init__(self):
        # Point Cloud Settings (incl. down sampling)
        self.point_cloud_path: Path = None
        self.down_sample_method: DSM = None
        self.down_sample_params: float = 0.0

        # Point cloud Normal Estimation Settings
        self.normal_estimation_neighbours: int = 0
        self.normal_estimation_radius: float = 0.0
        self.skip_normalizing_normals: bool = False
        self.orient_normals: Optional[int] = None

        # Surface Reconstruction Settings
        self.surface_reconstruction_method: SRM = SRM.DELAUNAY_TRIANGULATION
        self.surface_reconstruction_params: Optional[Dict[SRP, Any]] = None

        # Mesh Cleaning Settings
        self.mesh_cleaning_methods: Optional[Set[MCM]] = None
        self.edge_length_cleaning_portion: float = 1.0
        self.aspect_ratio_cleaning_portion: float = 1.0

        # Mesh Evaluation
        self.mesh_evaluation_metrics: Optional[Set[MEM]] = None

        # Misc settings
        self.store_mesh: bool = False
        self.store_preprocessed_pointcloud: bool = False
        self.processes: int = 1
        self.chunk_size: int = 1000

        self.triangle_normal_deviation_method: TNDM = TNDM.NAIVE

    def __copy__(self):
        c = RunConfiguration()
        for var_name, value in self.__dict__:
            setattr(c, var_name, value)
        return c

    def copy(self):
        return self.__copy__()
