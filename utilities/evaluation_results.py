from typing import Union
from pathlib import Path
import os

import numpy as np


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

        self.number_of_vertices_original: int = 0
        self.number_of_vertice_after_downsampling: int = 0
        self.number_of_vertices_after_cleaning: int = 0

    def save_to_file(self, folder_path: Path, compressed: bool = False, delimiter: str = ';'):
        if not delimiter:
            delimiter = ";"

        if not folder_path.is_dir():
            print(f"Cannot save, given path {folder_path} is not a folder.")
            return

        if not folder_path.exists():
            os.makedirs(folder_path)

        suffix = ".txt" if not compressed else ".gz"

        if len(self.aspect_ratios) > 0:
            aspect_ratios_path = os.path.join(folder_path, f"aspect_ratios{suffix}")
            np.savetxt(str(aspect_ratios_path), self.aspect_ratios, delimiter=delimiter, newline='\n')

        if len(self.edge_lengths) > 0:
            edge_lengths_path = os.path.join(folder_path, f"edge_lengths{suffix}")
            np.savetxt(str(edge_lengths_path), self.edge_lengths, delimiter=delimiter, newline='\n')

        if len(self.discrete_curvatures) > 0:
            dc_path = os.path.join(folder_path, f"discrete_curvatures{suffix}")
            np.savetxt(str(dc_path), self.discrete_curvatures, delimiter=delimiter, newline="\n")

        if len(self.discrete_curvatures) > 0:
            nd_path = os.path.join(folder_path, f"normal_deviations{suffix}")
            np.savetxt(str(nd_path), self.normal_deviations, delimiter=delimiter, newline="\n")

        other_result_path = os.path.join(folder_path, "results.txt")
        with open(other_result_path, mode="w") as file:
            file.write(f"Loading_and_Preprocessing_Time{delimiter}{self.loading_and_preprocessing_time}\n")
            file.write(f"Surface_Reconstruction_Time{delimiter}{self.surface_reconstruction_time}\n")
            file.write(f"Cleaning_Time{delimiter}{self.cleaning_time}\n")
            file.write(f"Evaluation_Time{delimiter}{self.evaluation_time}\n")

            if isinstance(self.connectivity_vertices_per_component, int) or \
               isinstance(self.connectivity_vertices_per_component, np.int32):
                cvpc_string = str(self.connectivity_vertices_per_component)
                file.write(f"connectivity_vertices_per_component{delimiter}{cvpc_string}\n")
            elif not (self.connectivity_vertices_per_component is None) and len(self.connectivity_vertices_per_component) > 0:
                cvpc_string = np.array2string(self.connectivity_vertices_per_component, separator=delimiter)
                file.write(f"connectivity_vertices_per_component{delimiter}{cvpc_string}\n")

            if isinstance(self.connectivity_triangles_per_component, int) or \
               isinstance(self.connectivity_triangles_per_component, np.int32):
                ctpc_string = str(self.connectivity_triangles_per_component)
                file.write(f"connectivity_triangles_per_component{delimiter}{ctpc_string}\n")
            elif not (self.connectivity_triangles_per_component is None) and len(self.connectivity_triangles_per_component) > 0:
                ctpc_string = np.array2string(self.connectivity_triangles_per_component, separator=delimiter)
                file.write(f"connectivity_triangles_per_component{delimiter}{ctpc_string}\n")

            file.write(f"number_of_vertices_original{delimiter}{self.number_of_vertices_original}\n")
            file.write(f"number_of_vertice_after_downsampling{delimiter}{self.number_of_vertice_after_downsampling}\n")
            file.write(f"number_of_vertices_after_cleaning{delimiter}{self.number_of_vertices_after_cleaning}\n")
