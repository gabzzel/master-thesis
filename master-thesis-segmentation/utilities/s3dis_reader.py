import os
from typing import Union, List, Tuple
from pathlib import Path
import tqdm

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
        "beam",  # 0
        "bookcase",  # 1
        "board",  # 2
        "ceiling",  # 3
        "chair",  # 4
        "clutter",  # 5
        "column",  # 6
        "door",  # 7
        "floor",  # 8
        "sofa",  # 9
        "stairs",  # 10
        "table",  # 11
        "wall",  # 12
        "window"  # 13
    ]

    area_xyz = None
    area_colors = None
    area_labels = None

    for area_dir in os.scandir(root_path):
        if "area" not in area_dir.name.lower():
            continue

        area_dir_path = Path(area_dir.path)
        if not area_dir_path.is_dir():
            continue

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
                    labels = np.full(shape=(len(xyz),), fill_value=label_index, dtype=np.int32)

                area_xyz = xyz if area_xyz is None else np.vstack((area_xyz, xyz))
                area_colors = colors if area_colors is None else np.vstack((area_colors, colors))
                area_labels = labels if area_labels is None else np.vstack((area_labels, labels))

                print(
                    f"Processed Area {area_index} | Room {room_dir_path.stem} | Object {annotation_file.name} | ({xyz.shape[0]} points)")

        break

    pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(area_xyz))
    pcd.colors = open3d.utility.Vector3dVector(area_colors)
    return pcd, area_labels


def get_s3dis_rooms(s3dis_root_dir: Union[str, os.PathLike],
                    area_index: int,
                    one_hot: bool = True) -> Tuple[List[open3d.cpu.pybind.geometry.PointCloud], List[np.ndarray]]:
    """
    Get paths to S3DIS point clouds of all rooms within an area, with annotations.

    :param one_hot: Whether to return the labels in one-hot format.
    :param area_index: The index of the area to get the rooms of.
    :param s3dis_root_dir: The root directory of the S3DIS dataset.
    :return: Returns a tuple, containing list of point clouds of the rooms as it first element and the annotations as the \
        second element.
    """

    root_path = Path(s3dis_root_dir)
    assert root_path.is_dir()

    current_index = 0

    label_mapping = [
        "beam",  # 0
        "bookcase",  # 1
        "board",  # 2
        "ceiling",  # 3
        "chair",  # 4
        "clutter",  # 5
        "column",  # 6
        "door",  # 7
        "floor",  # 8
        "sofa",  # 9
        "stairs",  # 10
        "table",  # 11
        "wall",  # 12
        "window"  # 13
    ]

    rooms_xyz: List[np.ndarray] = []
    rooms_colors: List[np.ndarray] = []
    rooms_labels: List[np.ndarray] = []

    for area_dir in os.scandir(root_path):
        if "area" not in area_dir.name.lower():
            continue

        area_dir_path = Path(area_dir.path)
        if not area_dir_path.is_dir():
            continue

        if current_index != area_index:
            current_index += 1
            continue

        for room_dir in os.scandir(area_dir.path):
            room_dir_path = Path(room_dir.path)
            if not room_dir_path.is_dir():
                continue

            annotations_dir_path = Path(room_dir.path).joinpath("Annotations")
            if not annotations_dir_path.exists() or not annotations_dir_path.is_dir():
                continue

            room_xyz = None
            room_colors = None
            room_labels = None

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
                    labels = np.full(shape=(len(xyz),), fill_value=label_index, dtype=np.int32)

                room_xyz = xyz if room_xyz is None else np.vstack((room_xyz, xyz))
                room_colors = colors if room_colors is None else np.vstack((room_colors, colors))
                room_labels = labels if room_labels is None else np.vstack((room_labels, labels))

                print(
                    f"Processed Area {area_index} | Room {room_dir_path.stem} | Object {annotation_file.name} | ({xyz.shape[0]} points)")

            rooms_xyz.append(room_xyz)
            rooms_colors.append(room_colors)
            rooms_labels.append(room_labels)

        break

    point_clouds: List[open3d.geometry.PointCloud] = []

    for i in range(len(rooms_xyz)):
        print(f"Estimating normals for point cloud of room {i}")
        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(rooms_xyz[i]))
        pcd.colors = open3d.utility.Vector3dVector(rooms_colors[i])
        pcd.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(knn=10))
        pcd.orient_normals_consistent_tangent_plane(10)
        point_clouds.append(pcd)

    return point_clouds, rooms_labels


def save_s3dis_rooms(s3dis_root_dir: Union[str, os.PathLike],
                     save_path: Union[str, os.PathLike],
                     one_hot: bool = True):
    """
    Get paths to S3DIS point clouds of all rooms within an area, with annotations.

    :param one_hot: Whether to return the labels in one-hot format.
    :param s3dis_root_dir: The root directory of the S3DIS dataset.
    """

    root_path = Path(s3dis_root_dir)
    assert root_path.is_dir()

    label_mapping = [
        "beam",  # 0
        "bookcase",  # 1
        "board",  # 2
        "ceiling",  # 3
        "chair",  # 4
        "clutter",  # 5
        "column",  # 6
        "door",  # 7
        "floor",  # 8
        "sofa",  # 9
        "stairs",  # 10
        "table",  # 11
        "wall",  # 12
        "window"  # 13
    ]

    area_dirs = list(os.scandir(root_path))
    for area_dir in tqdm.tqdm(area_dirs, position=0, leave=True, desc="Processing areas", miniters=1):
        if "area" not in area_dir.name.lower():
            continue

        area_dir_path = Path(area_dir.path)
        if not area_dir_path.is_dir():
            continue

        rooms_dirs = list(os.scandir(area_dir.path))
        for room_dir in tqdm.tqdm(rooms_dirs, position=1, leave=False, desc="Processing rooms", miniters=1):
            room_dir_path = Path(room_dir.path)
            if not room_dir_path.is_dir():
                continue

            annotations_dir_path = Path(room_dir.path).joinpath("Annotations")
            if not annotations_dir_path.exists() or not annotations_dir_path.is_dir():
                continue

            room_xyz = None
            room_colors = None
            room_labels = None

            annotation_files = list(os.scandir(annotations_dir_path))

            for annotation_file in annotation_files:
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
                    labels = np.full(shape=(len(xyz),), fill_value=label_index, dtype=np.int32)

                room_xyz = xyz if room_xyz is None else np.vstack((room_xyz, xyz))
                room_colors = colors if room_colors is None else np.vstack((room_colors, colors))
                room_labels = labels if room_labels is None else np.vstack((room_labels, labels))

            pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(room_xyz))
            pcd.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(knn=10))
            pcd.orient_normals_consistent_tangent_plane(10)
            pcd.normalize_normals()
            room_normals = np.asarray(pcd.normals)

            room_data = np.hstack((room_xyz, room_colors, room_normals, room_labels))
            room_save_path = Path(save_path).joinpath(f"{area_dir.name}_{room_dir.name}.npy")
            np.save(str(room_save_path), room_data)
