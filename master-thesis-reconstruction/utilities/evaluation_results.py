import csv
from typing import Union
from pathlib import Path
import os
from collections.abc import Sized

import numpy as np
import open3d

import utilities.run_configuration
import utilities.utils


class EvaluationResults:
    def __init__(self, name: str):
        self.name = name

        self.loading_and_preprocessing_time: float = 0.0
        self.surface_reconstruction_time: float = 0.0
        self.cleaning_time: float = 0.0
        self.evaluation_time: float = 0.0

        self.aspect_ratios: Union[list, np.ndarray] = []
        self.edge_lengths: Union[list, np.ndarray] = []

        self.connectivity_vertices_per_component = []
        self.connectivity_triangles_per_component = []
        self.discrete_curvatures = []
        self.normal_deviations = []

        self.hausdorff_distance: float = -1.0
        self.chamfer_distance: float = -1.0
        self.point_cloud_to_mesh_distances: Union[list, np.ndarray] = []

        self.number_of_points_original: int = 0
        self.number_of_points_after_downsampling: int = 0

        self.number_of_vertices_original: int = 0
        self.number_of_triangles_original: int = 0

        self.number_of_vertices_after_cleaning: int = 0
        self.number_of_triangles_after_cleaning: int = 0

    def save_to_file(self,
                     run_config: utilities.run_configuration.RunConfiguration,
                     raw_pcd: open3d.geometry.PointCloud,
                     pcd: open3d.geometry.PointCloud,
                     folder_path: Path,
                     delimiter: str = ';'):
        if not delimiter:
            delimiter = ";"

        if not folder_path.is_dir():
            print(f"Cannot save, given path {folder_path} is not a folder.")
            return

        if not folder_path.exists():
            os.makedirs(folder_path)

        fmt = "%7.7f"

        raw_results_path = os.path.join(folder_path, "raw_results.npz")
        with open(raw_results_path, mode='wb') as raw_results_file:
            ar = self.aspect_ratios if len(self.aspect_ratios) > 0 else np.array([])
            el = self.edge_lengths if len(self.edge_lengths) > 0 else np.array([])
            dc = self.discrete_curvatures if len(self.discrete_curvatures) > 0 else np.array([])
            nd = self.normal_deviations if len(self.normal_deviations) > 0 else np.array([])
            dists = self.point_cloud_to_mesh_distances if len(self.point_cloud_to_mesh_distances) > 0 else np.array([])

            if isinstance(self.connectivity_triangles_per_component, Sized):
                tris_pc = np.asarray(self.connectivity_triangles_per_component)
            else:
                tris_pc = np.array([self.connectivity_triangles_per_component])

            if isinstance(self.connectivity_vertices_per_component, Sized):
                verts_pc = np.asarray(self.connectivity_vertices_per_component)
            else:
                verts_pc = np.array([self.connectivity_vertices_per_component])

            np.savez_compressed(file=raw_results_file,
                                aspect_ratios=ar,
                                edge_lengths=el,
                                discrete_curvatures=dc,
                                distances=dists,
                                normal_deviations=nd,
                                triangles_per_component=tris_pc,
                                vertices_per_component=verts_pc)

        results_path = folder_path.parent.joinpath("results.csv")
        if not results_path.exists():
            with open(results_path, mode="w") as file:
                file.write(f"Run Name{delimiter}"
                           f"Method{delimiter}"
                           f"Dataset{delimiter}"
                           f"Downsampling Method{delimiter}"
                           f"Downsampling Param{delimiter}"
                           f"Points Original{delimiter}"
                           f"Points after downsampling{delimiter}"
                           f"Downsampling Percentage{delimiter}"
                           f"Loading and preprocessing Time (s){delimiter}"
                           f"Alpha{delimiter}"
                           f"BPA Radii{delimiter}"
                           f"Max Octree Depth{delimiter}"
                           f"Density Threshold{delimiter}"
                           f"Vertices before cleaning{delimiter}"
                           f"Triangles before cleaning{delimiter}"
                           f"Reconstruction Time (s){delimiter}"
                           f"Edge Length Percentile{delimiter}"
                           f"Aspect Ratio Percentile{delimiter}"
                           f"Vertices after cleaning{delimiter}"
                           f"Triangles after cleaning{delimiter}"
                           f"Cleaning time (s){delimiter}"
                           f"Hausdorff distance{delimiter}"
                           f"Chamfer distance{delimiter}"
                           f"RMSE Distances{delimiter}"
                           f"Mean Triangle Normal Deviation (degrees){delimiter}"
                           f"Connectivity Largest Component Ratio{delimiter}"
                           f"Component Count{delimiter}"
                           f"Discrete Curvature Min{delimiter}"
                           f"Discrete Curvate Max{delimiter}"
                           f"Discrete Curvature Mean{delimiter}"
                           f"Discrete Curvature Std")
                file.write("\n")

        with open(results_path, mode="a") as file:
            file.write(f"{run_config.name}{delimiter}")
            file.write(f"{run_config.surface_reconstruction_method}{delimiter}")
            file.write(f"{run_config.point_cloud_path.stem}{delimiter}")
            file.write(f"{run_config.down_sample_method}{delimiter}")
            file.write(f"{run_config.down_sample_params}{delimiter}")
            file.write(f"{len(raw_pcd.points)}{delimiter}")
            file.write(f"{len(pcd.points)}{delimiter}")
            downsampling_percentage = float(len(pcd.points)) / float(len(raw_pcd.points)) * 100.0
            file.write(f"{downsampling_percentage}{delimiter}")
            file.write(f"{self.loading_and_preprocessing_time}{delimiter}")
            file.write(f"{run_config.alpha}{delimiter}")
            file.write(f"{run_config.ball_pivoting_radii}{delimiter}")

            p_max_depth = f"{run_config.poisson_octree_max_depth} (adaptive)" if run_config.poisson_adaptive else str(run_config.poisson_octree_max_depth)
            file.write(f"{p_max_depth}{delimiter}")
            file.write(f"{run_config.poisson_density_quantile}{delimiter}")
            file.write(f"{self.number_of_vertices_original}{delimiter}")
            file.write(f"{self.number_of_triangles_original}{delimiter}")
            file.write(f"{self.surface_reconstruction_time}{delimiter}")
            file.write(f"{run_config.edge_length_cleaning_portion}{delimiter}")
            file.write(f"{run_config.aspect_ratio_cleaning_portion}{delimiter}")
            file.write(f"{self.number_of_vertices_after_cleaning}{delimiter}")
            file.write(f"{self.number_of_triangles_after_cleaning}{delimiter}")
            file.write(f"{self.cleaning_time}{delimiter}")
            file.write(f"{self.hausdorff_distance}{delimiter}")
            file.write(f"{self.chamfer_distance}{delimiter}")
            rmse_distances = np.sqrt(np.mean(np.power(self.point_cloud_to_mesh_distances, 2))) if len(
                self.point_cloud_to_mesh_distances) > 0 else 0
            file.write(f"{rmse_distances}{delimiter}")
            mean_triangle_normal_deviation = np.mean(self.normal_deviations) if len(self.normal_deviations) > 0 else 0
            file.write(f"{mean_triangle_normal_deviation}{delimiter}")

            largest_component_ratio = 1
            component_count = 1
            try:
                if isinstance(self.connectivity_triangles_per_component, Sized):
                    largest_component_ratio = np.max(self.connectivity_triangles_per_component) / \
                                              np.sum(self.connectivity_triangles_per_component)
                    component_count = len(self.connectivity_triangles_per_component)
            except Exception as e:
                print("Could not write component ratio and count. It's 1 in that case.")

            file.write(f"{largest_component_ratio}{delimiter}")
            file.write(f"{component_count}{delimiter}")

            discrete_curvatures_max, discrete_curvature_min, avg, med, std = utilities.utils.get_stats(
                self.discrete_curvatures,
                print_results=False,
                return_results=True,
                round_digits=-1)
            file.write(f"{discrete_curvature_min}{delimiter}")
            file.write(f"{discrete_curvatures_max}{delimiter}")
            file.write(f"{avg}{delimiter}")
            file.write(f"{std}{delimiter}")
            if run_config.segments_path is not None:
                file.write(f"{run_config.segments_path.stem}{delimiter}")

            file.write("\n")  # Make sure to end with a newline, such that the next lines are written right.
