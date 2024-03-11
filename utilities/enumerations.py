import enum
from typing import Optional


class SurfaceReconstructionMethod(enum.Enum):
    BALL_PIVOTING_ALGORITHM = "bpa",
    ALPHA_SHAPES = "alpha_shapes",
    SCREENED_POISSON_SURFACE_RECONSTRUCTION = "spsr",
    DELAUNAY_TRIANGULATION = "delaunay"


class SurfaceReconstructionParameters(enum.Enum):
    ALPHA = "alpha",
    BPA_RADII = "radii",
    POISSON_OCTREE_MAX_DEPTH = "octree_max_depth",
    POISSON_DENSITY_QUANTILE_THRESHOLD = "density_quantile_threshold"


class MeshEvaluationMetric(enum.Enum):
    ALL = "all",
    DISCRETE_CURVATURE = "discrete_curvature",
    TRIANGLE_NORMAL_DEVIATIONS = "triangle_normal_deviations",
    EDGE_LENGTHS = "edge_lengths",
    TRIANGLE_ASPECT_RATIOS = "triangle_aspect_ratios",
    CONNECTIVITY = "connectivity",

    HAUSDORFF_DISTANCE = "hausdorff",
    CHAMFER_DISTANCE = "chamfer",
    MESH_TO_CLOUD_DISTANCE = "mesh_to_cloud"


class DownSampleMethod(enum.Enum):
    VOXEL = "voxel",
    RANDOM = "random"


class MeshCleaningMethod(enum.Enum):
    ALL = "all",
    EDGE_LENGTHS = "edge_lengths",
    ASPECT_RATIOS = "aspect_ratios",
    SIMPLE = "simple"


