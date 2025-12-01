# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: Lessons from Learning to Spin "Pens"
# Copyright (c) 2024 All Authors
# Licensed under MIT License
# https://github.com/HaozhiQi/penspin/
# --------------------------------------------------------
import torch
from torch import nn


class TemporalConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(TemporalConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, T, C)
        x = x.permute((0, 2, 1))      # (N, C, T)
        x = self.temporal_aggregation(x)
        x = self.low_dim_proj(x.flatten(1))
        return x