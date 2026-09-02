import contextlib

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)

SIGLIP_MEAN = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)
SIGLIP_STD = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)


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


def prepare_volume_batch(volume, device, image_size=224, channels_last=False,
                         mean=IMAGENET_MEAN, std=IMAGENET_STD,
                         interp_mode="bilinear", antialias=False):
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)

    volume = volume.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    volume = volume.div_(255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1).contiguous()
    batch_size, slices, channels, height, width = volume.shape
    flat = volume.reshape(batch_size * slices, channels, height, width)

    align_corners = False if interp_mode in ("bilinear", "bicubic") else None
    flat = F.interpolate(
        flat,
        size=(image_size, image_size),
        mode=interp_mode,
        align_corners=align_corners,
        antialias=antialias,
    )
    flat = flat.reshape(batch_size, slices, channels, image_size, image_size)

    flat = (flat - mean.to(device)) / std.to(device)

    return flat
