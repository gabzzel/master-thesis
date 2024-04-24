import os
from typing import Union, List, Tuple
from pathlib import Path

import numpy as np
import open3d.cpu.pybind.geometry


def get_s3dis_area(s3dis_root_dir: Union[str, os.PathLike],
                   area_index: int,
                   one_hot: bool = True) -> Tuple[open3d.cpu.pybind.geometry.PointCloud, np.ndarray]:
    """
    Get paths to S3DIS point clouds of an area, with annotations.

    :param one_hot: Whether to return the labels in one-hot format.
    :param area_index: The index of the area to get.
    :param s3dis_root_dir: The root directory of the S3DIS dataset.
    :return: Returns a tuple, containing the point cloud of the room as it first element and the annotations as the \
        second element.
    """

    root_path = Path(s3dis_root_dir)
    assert root_path.is_dir()

    current_index = 0

    label_mapping = [
        "beam",         # 0
        "bookcase",     # 1
        "board",        # 2
        "ceiling",      # 3
        "chair",        # 4
        "clutter",      # 5
        "column",       # 6
        "door",         # 7
        "floor",        # 8
        "sofa",         # 9
        "table",        # 10
        "wall",         # 11
        "window"        # 12
    ]

    area_xyz = None
    area_colors = None
    area_labels = None

    for area_dir in os.scandir(root_path):
        if "area" not in area_dir.name.lower():
            continue

        area_dir_path = Path(area_dir.path)
        assert area_dir_path.is_dir()

        if current_index != area_index:
            current_index += 1
            continue

        for room_dir in os.scandir(area_dir.path):
            room_dir_path = Path(room_dir.path)
            if not room_dir_path.is_dir():
                continue

            # room_pcd_file_name = room_dir.name + ".txt"
            # room_pcd_file_path = Path(room_dir.path).joinpath(room_pcd_file_name)
            # assert room_pcd_file_path.is_file()
            # assert room_pcd_file_path.exists()

            annotations_dir_path = Path(room_dir.path).joinpath("Annotations")
            if not annotations_dir_path.exists() or not annotations_dir_path.is_dir():
                continue

            for annotation_file in os.scandir(annotations_dir_path):
                if not annotation_file.path.endswith(".txt"):
                    continue

                object_data = np.loadtxt(annotation_file.path, delimiter=" ")
                xyz = object_data[:, [0, 1, 2]].astype(np.float32)
                colors = object_data[:, [3, 4, 5]].astype(np.float32) / 255.0  # Normalize the colors

                label_index = label_mapping.index(annotation_file.name.split("_")[0])

                if one_hot:
                    labels = np.zeros(shape=(len(xyz), len(label_mapping)), dtype=np.int32)
                    labels[:, label_index] = 1
                else:
                    labels = np.full(shape=(len(xyz), ), fill_value=label_index, dtype=np.int32)

                area_xyz = xyz if area_xyz is None else np.vstack((area_xyz, xyz))
                area_colors = colors if area_colors is None else np.vstack((area_colors, colors))
                area_labels = labels if area_labels is None else np.vstack((area_labels, labels))

        break

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(area_xyz))
    pcd.colors = open3d.utility.Vector3dVector(area_colors)
    return pcd, area_labels
