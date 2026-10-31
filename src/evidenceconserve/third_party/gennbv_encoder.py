"""CPU/graph adaptation of GenNBV's released Hybrid_Encoder architecture.

Source: https://github.com/zjwzcx/GenNBV/blob/master/gennbv/network/hybrid_encoder.py
Original authors: Xiao Chen et al.; original terms: third_party/licenses/gennbv.txt.
Changes: remove Gym/SB3 constructor dependencies, accept tensors directly,
pad finite histories and resize a declared coarse grid to the original 20^3.
The released active branches are occupancy and periodic pose history.
"""

import torch
from torch import nn
from torch.nn import functional as F


class HybridEncoder(nn.Module):
    def __init__(self, width=256, history=100):
        super().__init__()
        self.history = history
        self.grid = nn.Sequential(nn.Conv3d(1, 16, 3, stride=2), nn.BatchNorm3d(16), nn.ReLU(),
                                  nn.Conv3d(16, 16, 3, stride=2), nn.BatchNorm3d(16), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(1024, width), nn.ReLU())
        self.motion = nn.Sequential(nn.Linear(24 * history, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())

    def forward(self, voxel, poses):
        voxel = F.interpolate(voxel, size=(20, 20, 20), mode="nearest")
        if poses.shape[1] > self.history:
            raise ValueError("History exceeds the registered GenNBV pose window")
        poses = F.pad(poses, (0, 0, 0, self.history - poses.shape[1]))
        frequency = poses.new_tensor([1., 2.])
        periodic = (poses[..., None] * frequency).reshape(*poses.shape[:2], 12)
        periodic = torch.cat([periodic.sin(), periodic.cos()], -1).flatten(1)
        return self.fusion(torch.cat([self.grid(voxel), self.motion(periodic)], -1))
