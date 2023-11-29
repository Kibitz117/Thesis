import torch

import numpy as np

class SharpeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, returns):
        sharpe = np.mean(returns) / np.std(returns)
        return -sharpe