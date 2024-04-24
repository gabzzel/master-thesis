import torch
from torch import nn
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv
import torch.nn.functional as F


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2Segmentation(torch.nn.Module):
    def __init__(self,
                 set_abstraction_ratio_1,
                 set_abstraction_ratio_2,
                 set_abstraction_radius_1,
                 set_abstraction_radius_2,
                 dropout,
                 number_of_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            MLP([3, 64, 64, 128])
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            MLP([128 + 3, 128, 128, 256])
        )

        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))
        # self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

        # Upsampling layer
        self.upconv1 = nn.Conv1d(1024 + 512, 512, 1)
        self.upconv2 = nn.Conv1d(512 + 256, 256, 1)

        # Output layer for segmentation
        self.segmentation_output = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, number_of_classes, 1)
        )

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Upsampling
        upsampled_sa2 = self.upconv1(torch.cat([sa2_out[0], self.interpolate(x, sa2_out[1], data.pos)], dim=1))
        upsampled_sa1 = self.upconv2(torch.cat([sa1_out[0], self.interpolate(upsampled_sa2, sa1_out[1], data.pos)], dim=1))

        # Output layer for segmentation
        logits = self.segmentation_output(upsampled_sa1)

        return logits

    def loss(self, logits, labels):
        """
        Compute the cross-entropy loss for each point and aggregate over all points.

        Args:
            logits (torch.Tensor): Predicted logits for each point. Shape (batch_size, num_classes, num_points).
            labels (torch.Tensor): Ground truth labels for each point. Shape (batch_size, num_points).

        Returns:
            torch.Tensor: Loss value.
        """
        # Per-point cross-entropy loss
        loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')

        # Aggregate loss over all points
        return loss.mean()

    def interpolate(self, features, from_pos, to_pos, k=3):
        """
        Interpolate features from downsampled set to upsampled set using the mean squared distance of the k nearest neighbors.

        Args:
            features (torch.Tensor): Features of the downsampled set. Shape (batch_size, channels, num_downsampled_points).
            from_pos (torch.Tensor): Positions of the downsampled set. Shape (batch_size, 3, num_downsampled_points).
            to_pos (torch.Tensor): Positions of the upsampled set. Shape (batch_size, 3, num_upsampled_points).
            k (int): Number of nearest neighbors to consider for interpolation.

        Returns:
            torch.Tensor: Interpolated features. Shape (batch_size, channels, num_upsampled_points).
        """
        batch_size, channels, num_downsampled_points = features.size()
        num_upsampled_points = to_pos.size(2)

        # Compute pairwise distances between downsampled and upsampled points
        # Shape: (batch_size, num_upsampled_points, num_downsampled_points)
        dist = torch.cdist(to_pos.permute(0, 2, 1).contiguous(), from_pos.permute(0, 2, 1).contiguous())

        # Get the indices of the k nearest neighbors
        _, indices = torch.topk(dist, k, dim=2, largest=False)

        # Compute weights based on inverse squared distances
        weights = 1 / torch.pow(dist.gather(2, indices), 2)  # Shape: (batch_size, num_upsampled_points, k)

        # Normalize weights
        weights = weights / torch.sum(weights, dim=2, keepdim=True)

        # Interpolate features
        # Shape: (batch_size, channels, num_upsampled_points)
        interpolated_features = torch.bmm(features, weights.transpose(1, 2))

        return interpolated_features
