import os
from typing import List, Union

import open3d.cpu.pybind.geometry
import torch
from torch.utils.data import Dataset, random_split
import open3d

import utilities.s3dis_reader


def load_labels(path_to_label_file):
    raise NotImplementedError()


class S3DIS_Dataset_Area(Dataset):
    def __init__(self,
                 s3dis_root_dir: Union[str, os.PathLike],
                 number_of_classes: int,
                 labels_are_one_hot: bool = True,
                 train: bool = True):

        self.s3dis_root_dir = s3dis_root_dir
        self.train = train

        self.labels_are_one_hot = labels_are_one_hot
        self.number_of_classes = number_of_classes

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        point_cloud, labels = utilities.s3dis_reader.get_s3dis_area(self.s3dis_root_dir, idx, self.labels_are_one_hot)
        positions = torch.tensor(point_cloud.points).float()
        normals = torch.tensor(point_cloud.normals).float()
        colors = torch.tensor(point_cloud.colors).float()

        # Load ground truth labels
        labels = torch.tensor(labels).long()

        return positions, normals, colors, labels

    def split_train_val(self, val_split=0.2):
        # Split dataset into train and validation sets
        dataset_size = len(self)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size])
        return train_dataset, val_dataset
    