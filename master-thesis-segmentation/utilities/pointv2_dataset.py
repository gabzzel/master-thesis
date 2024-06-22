import numpy as np
from torch.utils.data import Dataset
from utilities.pointnetv2_utilities import convert_to_batches
import torch


class PointNetV2_CustomDataset(Dataset):
    def __init__(self, points, colors, normals, point_amount, block_size, stride):
        self.blocks_data, self.indices = convert_to_batches(points, colors, normals,
                                                                point_amount=point_amount,
                                                                block_size=block_size,
                                                                stride=stride)
    def __len__(self):
        return len(self.blocks_data)

    def __getitem__(self, index):
        data = self.blocks_data[index]
        indices = self.indices[index]

        return np.hstack((data, indices[:, np.newaxis]))