import os
from typing import List, Union
from pathlib import Path

import open3d.cpu.pybind.geometry
import torch
from torch.utils.data import Dataset, random_split
import open3d
import numpy as np

import utilities.s3dis_reader


def load_labels(path_to_label_file):
    raise NotImplementedError()


class S3DIS_Dataset_Area(Dataset):
    def __init__(self,
                 s3dis_root_dir: Union[str, os.PathLike],
                 number_of_classes: int,
                 train: bool = True):

        self.s3dis_root_dir = s3dis_root_dir
        self.train = train

        self.point_clouds = []
        self.labels = []

        rooms_path = Path(s3dis_root_dir).joinpath("rooms")
        unique_rooms = set()

        for room_path_candidate in os.listdir(rooms_path):
            room_path = Path(room_path_candidate)
            if not room_path.is_file():
                continue
            
            # If we are training and come across area 5, continue
            # If we are not training (i.e. evaluating or inference) and we come across something NOT area 5, continue
            if ("area5" in room_path) == train:
                continue

            room_name = str(room_path).removesuffix(room_path.suffix)
            unique_rooms.add(room_name)
        
        for room_name in unique_rooms:
            pcd = open3d.io.read_point_cloud(str(rooms_path.joinpath(".ply")))
            self.point_clouds.append(pcd)
            self.labels.append(np.load(rooms_path.joinpath(".npy")))

        self.number_of_classes = number_of_classes

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        positions = torch.tensor(point_cloud.points).float()
        normals = torch.tensor(point_cloud.normals).float()
        colors = torch.tensor(point_cloud.colors).float()

        # Load ground truth labels
        labels = torch.tensor(self.labels[idx]).long()

        return positions, normals, colors, labels

    def split_train_val(self, val_split=0.2):
        # Split dataset into train and validation sets
        dataset_size = len(self)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset
    