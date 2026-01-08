import torch
import numpy as np

def convert_to_tensor(x):
    return torch.Tensor(x)

def repeat_and_permute(x):
    # Input x: (Slices, H, W)
    # Output: (Slices, 3, H, W)
    return x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize(x, device):
    # x shape: (Batch, Slices, 3, H, W)
    # ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 1, 3, 1, 1)
    return (x - mean) / std
