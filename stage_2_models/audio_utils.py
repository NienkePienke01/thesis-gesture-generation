import torch
from typing import Callable, List

import torch
import torch as th
import torch.nn as nn
from einops import rearrange

def build_single_audio_conv(c: int) -> List[nn.Module]:
    return [
        torch.nn.Conv1d(c, max(256, c), kernel_size=3, dilation=1),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        #
        torch.nn.Conv1d(max(256, c), max(256, c), kernel_size=3, dilation=2),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        #
        torch.nn.Conv1d(max(128, c), max(128, c), kernel_size=3, dilation=3),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        #
        torch.nn.Conv1d(max(128, c), c, kernel_size=3, dilation=1),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        #
        torch.nn.Conv1d(c, c, kernel_size=3, dilation=2),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        #
        torch.nn.Conv1d(c, c, kernel_size=3, dilation=3),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Dropout(0.2),
        ]

def setup_audio_model(cond_feature_dim: int, d_model: int, num_audio_layers: int) -> nn.Sequential:
    pre_layers = []
    for _ in range(num_audio_layers):
        pre_layers += build_single_audio_conv(cond_feature_dim)

    pre_layers += [
        torch.nn.Conv1d(cond_feature_dim, d_model, kernel_size=1)
    ]
    return nn.Sequential(*pre_layers)