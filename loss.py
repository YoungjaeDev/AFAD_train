import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, weights):
        return (weights * (pred - target) ** 2).mean()