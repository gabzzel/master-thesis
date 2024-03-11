from pathlib import Path
from typing import Optional, Set, Dict, Any

from utilities.enumerations import DownSampleMethod as DSM
from utilities.enumerations import MeshCleaningMethod as MCM
from utilities.enumerations import MeshEvaluationMetric as MEM
from utilities.enumerations import SurfaceReconstructionMethod as SRM
from utilities.enumerations import SurfaceReconstructionParameters as SRP


class RunConfiguration:
    def __init__(self,
                 pcd_path: Path,
                 down_sample_method: DSM = None,
                 down_sample_params: float = 0.1,
                 surface_reconstruction_method: SRM = SRM.DELAUNAY_TRIANGULATION,
                 surface_reconstruction_params: Optional[dict] = None,
                 mesh_cleaning_methods: Optional[Set[MCM]] = None,
                 edge_length_cleaning_portion: float = 0.95,
                 aspect_ratio_cleaning_portion: float = 0.95,
                 normal_estimation_neighbours: int = 30,
                 normal_estimation_radius: float = 0.5,
                 skip_normalizing_normals: bool = False,
                 orient_normals: int = 10,
                 mesh_evaluation_metrics: Optional[Set[MEM]] = None,
                 store_mesh: bool = False):
        # Point Cloud Settings (incl. down sampling)
        self.point_cloud_path: Path = pcd_path
        self.down_sample_method: DSM = down_sample_method
        self.down_sample_params: float = down_sample_params

        # Point cloud Normal Estimation Settings
        self.normal_estimation_neighbours: int = normal_estimation_neighbours
        self.normal_estimation_radius: float = normal_estimation_radius
        self.skip_normalizing_normals: bool = skip_normalizing_normals
        self.orient_normals: Optional[int] = orient_normals

        # Surface Reconstruction Settings
        self.surface_reconstruction_method: SRM = surface_reconstruction_method
        self.surface_reconstruction_params: Dict[SRP, Any] = surface_reconstruction_params

        # Mesh Cleaning Settings
        self.mesh_cleaning_methods: Optional[Set[MCM]] = mesh_cleaning_methods
        self.edge_length_cleaning_portion: float = edge_length_cleaning_portion
        self.aspect_ratio_cleaning_portion: float = aspect_ratio_cleaning_portion

        # Mesh Evaluation
        self.mesh_evaluation_metrics: Optional[Set[MEM]] = mesh_evaluation_metrics

        self.store_mesh: bool = store_mesh

