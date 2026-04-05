import contextlib

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_context(device, enabled=True):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def maybe_channels_last(module, device):
    if device.type == "cuda":
        module.to(memory_format=torch.channels_last)
    return module


def prepare_volume_batch(volume, device, image_size=224, channels_last=False):
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)

    volume = volume.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    volume = volume.div_(255.0).unsqueeze(2).expand(-1, -1, 3, -1, -1)
    batch_size, slices, channels, height, width = volume.shape
    flat = volume.reshape(batch_size * slices, channels, height, width)
    flat = F.interpolate(
        flat,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    flat = flat.reshape(batch_size, slices, channels, image_size, image_size)

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    flat = (flat - mean) / std

    return flat
