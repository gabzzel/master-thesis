import copy
import itertools
import json
from os import PathLike
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import numpy as np


def get_settings_header() -> List[str]:
    result = [
        "pcd_path",
        "min_cluster_size",
        "min_samples",
        "method",
        "include_normals",
        "include_colors",
        "noise_nearest_neighbours"
    ]
    return result


class HDBSCANConfigAndResult:
    def __init__(self,
                 pcd_path: Union[str, PathLike] = "",
                 min_cluster_size: int = 100,
                 min_samples: Optional[int] = None,
                 method: str = "eom",
                 include_normals: bool = True,
                 include_colors: bool = True,
                 noise_nearest_neighbours: int = 3,
                 visualize: bool = False):
        self.pcd_path: Union[str, PathLike] = pcd_path
        self.min_cluster_size: int = min_cluster_size
        self.min_samples: int = min_samples
        self.method: str = method
        self.include_normals: bool = include_normals
        self.include_colors: bool = include_colors
        self.noise_nearest_neighbours: int = noise_nearest_neighbours
        self.visualize: int = visualize

        self.clusters: np.ndarray = None
        self.noise_indices: np.ndarray = None
        self.membership_strengths: np.ndarray = None
        self.clustering_time: float = 0
        self.total_points: int = 0

    def get_results(self) -> List:
        cluster_sizes = np.bincount(self.clusters)
        results = [
            ("NumberOfClusters",len(np.unique(self.clusters))),
            ("LargestClusterSize",np.amax(cluster_sizes)),
            ("SmallestClusterSize",np.amin(cluster_sizes)),
            ("MeanCLusterSize",np.mean(cluster_sizes)),
            ("MedianClusterSize",np.median(cluster_sizes)),
            ("StdClusterSize",np.std(cluster_sizes)),
            ("NoisePoints",self.noise_indices.shape[0]),
            ("MeanMembershipStrength",np.mean(self.membership_strengths)),
            ("MedianMembershipStrength",np.median(self.membership_strengths)),
            ("StdMembershipStrength",np.std(self.membership_strengths)),
            ("ClusteringTime",self.clustering_time),
            ("TotalPoints",self.total_points)
        ]
        return results


def read_from_file(path: Union[str, PathLike]) -> HDBSCANConfigAndResult:
    path = Path(path)
    assert path.suffix == ".json"

    result = HDBSCANConfigAndResult()

    with open(file=path, mode="r+") as file:
        data = json.load(file)
        for attr in result.__dir__():
            if attr in data:
                result.__setattr__(attr, data[attr])

    return result


def read_from_file_multiple(path: Union[str, PathLike]) -> List[HDBSCANConfigAndResult]:
    path = Path(path)
    assert path.suffix == ".json"

    dummy = HDBSCANConfigAndResult()

    config_attributes: Dict[str, Any] = {}

    with open(file=path, mode="r+") as file:
        data = json.load(file)
        for attr in dummy.__dir__():
            if attr in data:
                config_attributes[attr] = data[attr]

    result: List[HDBSCANConfigAndResult] = []
    recursive_config_creator(config_attributes, result, None)
    return result


def recursive_config_creator(config_attribute_values: Dict,
                             configs: List[HDBSCANConfigAndResult],
                             current: Optional[HDBSCANConfigAndResult] = None) -> Optional[HDBSCANConfigAndResult]:
    if len(config_attribute_values) > 0:

        config_attribute_values = copy.copy(config_attribute_values)
        key = list(config_attribute_values.keys())[0]
        values = config_attribute_values.pop(key)
        if not isinstance(values, (list, tuple)):
            values = [values]

        for value in values:
            new_current = copy.copy(current) if current is not None else HDBSCANConfigAndResult()
            new_current.__setattr__(key, value)
            # print(f"Created new hdbscan config while varying {key}.")
            to_add = recursive_config_creator(config_attribute_values, configs, new_current)
            if to_add is not None:
                configs.append(to_add)

        return None

    return current


def write_multiple(configs: List[HDBSCANConfigAndResult],
                   path: Union[str, PathLike],
                   delimiter: str):
    with open(path, "w") as file:
        for s in get_settings_header():
            file.write(f"{s}{delimiter}")
        for k in configs[0].get_results():
            file.write(f"{k[0]}{delimiter}")
        file.write("\n")

        for config in configs:
            for s in get_settings_header():
                file.write(f"{config.__getattribute__(s)}{delimiter}")
            for v in config.get_results():
                file.write(f"{v[1]}{delimiter}")
            file.write("\n")

