import torch as th
from torch import nn


class DebugModel(nn.Module):
    def __init__(self):
        super(DebugModel, self).__init__()

    def forward(self, inputs, feature=True):
        flat = th.flatten(inputs, start_dim=1)
        if feature:
            return flat, flat
        return flat


def debug(*_, **__):
    return DebugModel()
