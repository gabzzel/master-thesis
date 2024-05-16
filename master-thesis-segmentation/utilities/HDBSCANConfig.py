import copy
import json
from os import PathLike
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import numpy as np
import tqdm


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

        # The average intersection over union taken over the different labels.
        self.mean_class_IoU: float = 0
        self.weighted_IoU: float = 0

    def get_results(self) -> List:
        cluster_sizes = np.bincount(self.clusters)
        results = [
            ("NumberOfClusters", len(np.unique(self.clusters))),
            ("LargestClusterSize", np.amax(cluster_sizes)),
            ("SmallestClusterSize", np.amin(cluster_sizes)),
            ("MeanCLusterSize", np.mean(cluster_sizes)),
            ("MedianClusterSize", np.median(cluster_sizes)),
            ("StdClusterSize", np.std(cluster_sizes)),
            ("NoisePoints", self.noise_indices.shape[0]),
            ("MeanMembershipStrength", np.mean(self.membership_strengths)),
            ("MedianMembershipStrength", np.median(self.membership_strengths)),
            ("StdMembershipStrength", np.std(self.membership_strengths)),
            ("ClusteringTime", self.clustering_time),
            ("TotalPoints", self.total_points),
            ("TotalWeightedIoU", self.weighted_IoU),
            ("MeanClassIoU", self.mean_class_IoU)
        ]
        return results

    def assign_labels_to_clusters(self, labels: np.ndarray, verbose: bool = True) -> np.ndarray:
        assert self.clusters is not None
        number_of_clusters = len(np.unique(self.clusters))


        progress_bar = None
        if verbose:
            progress_bar = tqdm.tqdm(total=4, unit="steps", desc=f"Assigning labels to {number_of_clusters} clusters.")

        classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
                   'board', 'clutter']



        point_indices_per_cluster = []
        point_indices_per_label = []
        cluster_to_label_map = np.full(shape=(number_of_clusters,), fill_value=-1, dtype=int)
        label_to_clusters_map: Dict[int, List] = {}

        for i in range(number_of_clusters):
            point_indices_per_cluster.append(np.argwhere(self.clusters == i).squeeze())

        if progress_bar:
            progress_bar.update()

        for i in range(len(classes)):
            point_indices_per_label.append(np.argwhere(labels == i).squeeze())

        if progress_bar:
            progress_bar.update()

        for i, cluster in enumerate(point_indices_per_cluster):
            max_intersection_size = 0
            max_label = -1

            for j, l in enumerate(point_indices_per_label):
                intersection_size = len(np.intersect1d(cluster, l))
                if intersection_size > max_intersection_size:
                    max_intersection_size = intersection_size
                    max_label = j

            cluster_to_label_map[i] = max_label
            if max_label in label_to_clusters_map:
                label_to_clusters_map[max_label].append(i)
            else:
                label_to_clusters_map[max_label] = [i]

            #if max_label == -1:
            #    print(f"Found no label for cluster {i}")
            #else:
            #    print(f"Found label {classes[max_label]} for cluster {i} (intersection {max_intersection_size} = {intersection_percentage}%)")

        IoU_per_class = np.zeros(shape=(len(classes),))
        class_weights = np.array([np.count_nonzero(labels == i) / float(len(labels)) for i in range(len(classes))])
        if progress_bar:
            progress_bar.update()

        for i in range(len(classes)):
            indices_with_label = np.nonzero(labels == i)[0]

            if len(indices_with_label) == 0:
                continue

            if i not in label_to_clusters_map:
                continue

            relevant_clusters: list = label_to_clusters_map[i]
            indices_in_clusters_with_label = np.nonzero(np.isin(self.clusters, relevant_clusters))[0]
            intersection_size = len(np.intersect1d(indices_in_clusters_with_label, indices_with_label))
            union = len(indices_in_clusters_with_label) + len(indices_with_label) - intersection_size
            IoU_per_class[i] = intersection_size / union if union > 0 else 0
            # print(f"Class {i} ({classes[i]}) has IoU {IoU_per_class[i]}")

        self.weighted_IoU = (IoU_per_class * class_weights).sum()
        self.mean_class_IoU = IoU_per_class.mean()
        if progress_bar:
            progress_bar.update()
        # print(f"Weighted total IoU: {to}, mean class IoU: {IoU_per_class.mean()}")
        return cluster_to_label_map


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


def read_from_file_multiple(path: Union[str, PathLike],
                            dataset_name_override: str = "") -> List[HDBSCANConfigAndResult]:
    path = Path(path)
    assert path.suffix == ".json"

    dummy = HDBSCANConfigAndResult()

    config_attributes: Dict[str, Any] = {}

    with open(file=path, mode="r+") as file:
        data = json.load(file)
        for attr in dummy.__dir__():
            if attr in data:
                config_attributes[attr] = data[attr]

    if len(dataset_name_override) > 0:
        config_attributes["pcd_path"] = dataset_name_override

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

    exists = Path(path).exists()

    with open(path, "a") as file:
        if not exists:
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
