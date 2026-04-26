import argparse
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightweight_models
import utils
from dataloader import PLANES, resolve_dataset_root


class SlicePairAugmentor:
    def __init__(self, noise_std=0.04, cutout_frac=0.15, gamma_jitter=0.15, shift_frac=0.04):
        self.noise_std = noise_std
        self.cutout_frac = cutout_frac
        self.gamma_jitter = gamma_jitter
        self.shift_frac = shift_frac

    def __call__(self, image):
        augmented = image.clone()
        if random.random() < 0.5:
            augmented = torch.flip(augmented, dims=[1])
        scale = 1.0 + random.uniform(-0.12, 0.12)
        bias = random.uniform(-0.12, 0.12)
        augmented = augmented * scale + bias

        if self.noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.noise_std

        if self.gamma_jitter > 0 and random.random() < 0.7:
            gamma = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
            lower = torch.quantile(augmented, 0.01)
            upper = torch.quantile(augmented, 0.99)
            scaled = torch.clamp((augmented - lower) / (upper - lower + 1e-6), 0.0, 1.0)
            augmented = scaled.pow(gamma) * (upper - lower) + lower

        if self.cutout_frac > 0 and random.random() < 0.5:
            height, width = augmented.shape
            cut_h = max(1, int(height * self.cutout_frac))
            cut_w = max(1, int(width * self.cutout_frac))
            top = random.randint(0, max(0, height - cut_h))
            left = random.randint(0, max(0, width - cut_w))
            augmented[top:top + cut_h, left:left + cut_w] = 0

        if self.shift_frac > 0 and random.random() < 0.5:
            max_dy = max(1, int(augmented.shape[0] * self.shift_frac))
            max_dx = max(1, int(augmented.shape[1] * self.shift_frac))
            dy = random.randint(-max_dy, max_dy)
            dx = random.randint(-max_dx, max_dx)
            augmented = torch.roll(augmented, shifts=(dy, dx), dims=(0, 1))
            if dy > 0:
                augmented[:dy, :] = 0
            elif dy < 0:
                augmented[dy:, :] = 0
            if dx > 0:
                augmented[:, :dx] = 0
            elif dx < 0:
                augmented[:, dx:] = 0

        return augmented


class MRNetSlicePairDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", mmap=True, cache_size=64, augmentor=None):
        super().__init__()
        self.root_dir = resolve_dataset_root(root_dir)
        self.split = split
        self.mmap = mmap
        self.cache_size = max(int(cache_size), 0)
        self.augmentor = augmentor or SlicePairAugmentor()
        self._cache = OrderedDict()

        self.plane_paths = []
        for plane in PLANES:
            plane_dir = os.path.join(self.root_dir, split, plane)
            for filename in sorted(os.listdir(plane_dir)):
                if filename.endswith(".npy"):
                    self.plane_paths.append(os.path.join(plane_dir, filename))

    def __len__(self):
        return len(self.plane_paths)

    def _load_volume(self, index):
        if index in self._cache:
            volume = self._cache.pop(index)
            self._cache[index] = volume
            return volume

        mmap_mode = "r" if self.mmap else None
        volume = np.load(self.plane_paths[index], mmap_mode=mmap_mode)
        volume = torch.from_numpy(np.asarray(volume, dtype=np.float32))
        if self.cache_size:
            self._cache[index] = volume
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return volume

    def __getitem__(self, index):
        volume = self._load_volume(index)
        slice_index = random.randrange(volume.shape[0])
        image = volume[slice_index]
        view_a = self.augmentor(image)
        view_b = self.augmentor(image)
        return view_a, view_b


def prepare_ssl_batch(images, device, image_size):
    if images.dim() == 3:
        images = images.unsqueeze(1)
    images = images.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    images = images.div_(255.0).repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = utils.IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    std = utils.IMAGENET_STD.view(1, 3, 1, 1).to(device)
    return (images - mean) / std


class SSLPretrainer(nn.Module):
    def __init__(self, backbone_name, pretrained, projection_dim):
        super().__init__()
        self.encoder, feature_dim = lightweight_models.build_backbone(backbone_name, pretrained=pretrained)
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, images):
        features = self.encoder(images)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)


def nt_xent_loss(z1, z2, temperature):
    batch_size = z1.shape[0]
    representations = torch.cat([z1, z2], dim=0)
    similarity = representations @ representations.T
    similarity = similarity / temperature

    mask = torch.eye(2 * batch_size, device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    targets = torch.arange(batch_size, device=similarity.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(similarity, targets)


def build_loader(dataset, args):
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    if torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def save_ssl_checkpoint(path, model, args, best_loss):
    state_dict = {
        f"encoder.{name}": tensor.detach().cpu()
        for name, tensor in model.encoder.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "ssl_args": vars(args),
            "best_ssl_loss": float(best_loss),
        },
        path,
    )


def run(args):
    device = utils.get_device()
    print(f"Using device: {device.type}")

    dataset = MRNetSlicePairDataset(
        root_dir=args.data_root,
        split="train",
        mmap=bool(args.mmap),
        cache_size=args.cache_size,
        augmentor=SlicePairAugmentor(
            noise_std=args.noise_std,
            cutout_frac=args.cutout_frac,
            gamma_jitter=args.gamma_jitter,
            shift_frac=args.shift_frac,
        ),
    )
    loader = build_loader(dataset, args)
    model = SSLPretrainer(
        backbone_name=args.model_type,
        pretrained=bool(args.pretrained),
        projection_dim=args.projection_dim,
    )
    model = utils.maybe_channels_last(model, device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_dir = os.path.abspath(args.log_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    best_epoch = -1
    start_time = time.time()
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        maybe_sync(device)
        epoch_start = time.time()

        for batch_index, (view_a, view_b) in enumerate(loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

            batch_a = prepare_ssl_batch(view_a, device, args.image_size)
            batch_b = prepare_ssl_batch(view_b, device, args.image_size)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(batch_a)
            z2 = model(batch_b)
            loss = nt_xent_loss(z1, z2, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            if batch_index > 0 and batch_index % args.log_every == 0:
                print(
                    f"[SSL Epoch: {epoch + 1} / {args.epochs} | batch: {batch_index} / {len(loader)}] "
                    f"| avg ssl loss {np.mean(losses):.4f} | lr: {utils.get_lr(optimizer):.6g}"
                )

        maybe_sync(device)
        epoch_seconds = time.time() - epoch_start
        epoch_times.append(epoch_seconds)
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"ssl loss: {epoch_loss:.4f} | elapsed time: {epoch_seconds:.2f} s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(output_dir, f"{args.prefix_name}_best_ssl.pth")
            save_ssl_checkpoint(checkpoint_path, model, args, best_loss)
            print(f"Best SSL checkpoint saved with loss: {best_loss:.4f}")

    training_seconds = time.time() - start_time
    avg_epoch_seconds = float(np.mean(epoch_times)) if epoch_times else float("nan")
    print("---")
    print(f"best_ssl_loss:      {best_loss:.6f}")
    print(f"training_seconds:   {training_seconds:.2f}")
    print(f"avg_epoch_seconds:  {avg_epoch_seconds:.2f}")
    print(f"epochs_ran:         {len(epoch_times)}")
    print(f"best_epoch:         {best_epoch}")
    print(f"device:             {device.type}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="MRNet-v1.0")
    parser.add_argument("--model_type", type=str, default="mobilenet_v3_small", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--pretrained", type=int, choices=[0, 1], default=1)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--cache_size", type=int, default=64)
    parser.add_argument("--mmap", type=int, choices=[0, 1], default=1)
    parser.add_argument("--noise_std", type=float, default=0.04)
    parser.add_argument("--cutout_frac", type=float, default=0.15)
    parser.add_argument("--gamma_jitter", type=float, default=0.15)
    parser.add_argument("--shift_frac", type=float, default=0.04)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="ssl_logs")
    parser.add_argument("--output_dir", type=str, default="ssl_pretrain_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
