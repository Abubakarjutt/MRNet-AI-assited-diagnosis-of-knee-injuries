import os
from collections import OrderedDict

import numpy as np
import random
import pandas as pd
import torch
import torch.utils.data as data


PLANES = ("sagittal", "coronal", "axial")
TASKS = ("abnormal", "acl", "meniscus")
REQUIRED_SPLIT_FILES = tuple(
    f"{split}-{task}.csv"
    for split in ("train", "valid")
    for task in TASKS
)


def _zero_pad_exam_id(exam_id):
    return f"{int(exam_id):04d}"


def _is_dataset_root(root_dir):
    required_paths = [os.path.join(root_dir, name) for name in REQUIRED_SPLIT_FILES]
    split_dirs = [
        os.path.join(root_dir, split, plane)
        for split in ("train", "valid")
        for plane in PLANES
    ]
    return all(os.path.isfile(path) for path in required_paths) and all(os.path.isdir(path) for path in split_dirs)


def resolve_dataset_root(root_dir):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    candidates = [root_dir, os.path.join(root_dir, "MRNet-v1.0")]
    for candidate in candidates:
        if _is_dataset_root(candidate):
            return candidate

    missing = []
    for candidate in candidates:
        missing.extend(
            path for path in [os.path.join(candidate, name) for name in REQUIRED_SPLIT_FILES] if not os.path.isfile(path)
        )
    preview = ", ".join(missing[:4]) if missing else "required CSV files and split directories"
    raise FileNotFoundError(
        f"MRNet dataset root is invalid: {root_dir}. Looked in {candidates}. Missing examples: {preview}"
    )


def _read_split_records(root_dir, split):
    frames = []
    for task in TASKS:
        csv_path = os.path.join(root_dir, f"{split}-{task}.csv")
        frame = pd.read_csv(csv_path, header=None, names=["id", task])
        frame["id"] = frame["id"].map(_zero_pad_exam_id)
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="id", how="inner")

    merged = merged.sort_values("id").reset_index(drop=True)
    return merged


class MRDataset(data.Dataset):
    """
    Backwards-compatible single-plane dataset.
    Prefer `MRMultiPlaneDataset` for training because it keeps plane ordering aligned.
    """

    def __init__(self, root_dir, plane, train=True, transform=None, weights=None, mmap=True):
        super().__init__()
        split = "train" if train else "valid"
        root_dir = resolve_dataset_root(root_dir)
        self.records = _read_split_records(root_dir, split)
        self.plane = plane
        self.transform = transform
        self.mmap = mmap
        self.folder_path = os.path.join(root_dir, split, plane)
        self.paths = [
            os.path.join(self.folder_path, f"{exam_id}.npy")
            for exam_id in self.records["id"].tolist()
        ]
        self.labels = self.records[list(TASKS)].to_numpy(dtype=np.float32)
        self.weights = _compute_class_weights(self.labels) if weights is None else torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        mmap_mode = "r" if self.mmap else None
        array = np.load(self.paths[index], mmap_mode=mmap_mode)
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
        label = torch.from_numpy(self.labels[index])

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label, self.weights




class MRVolumeAugmentor:
    def __init__(
        self,
        policy="none",
        noise_std=0.0,
        cutout_frac=0.0,
        slice_dropout=0.0,
        gamma_jitter=0.0,
        spatial_shift_frac=0.0,
    ):
        self.policy = policy
        self.noise_std = max(float(noise_std), 0.0)
        self.cutout_frac = min(max(float(cutout_frac), 0.0), 0.5)
        self.slice_dropout = min(max(float(slice_dropout), 0.0), 0.5)
        self.gamma_jitter = min(max(float(gamma_jitter), 0.0), 0.5)
        self.spatial_shift_frac = min(max(float(spatial_shift_frac), 0.0), 0.2)

    def __call__(self, volume):
        if self.policy == "none":
            return volume

        augmented = volume.clone()
        if self.policy in {"light", "strong", "knee_mri", "knee_mri_plus"}:
            augmented = self._random_flip(augmented)
            augmented = self._random_intensity_shift(augmented)

        if self.policy in {"strong", "knee_mri", "knee_mri_plus"}:
            augmented = self._random_noise(augmented)
            augmented = self._random_cutout(augmented)
            augmented = self._random_slice_dropout(augmented)

        if self.policy == "knee_mri_plus":
            augmented = self._random_gamma(augmented)
            augmented = self._random_spatial_shift(augmented)

        return augmented

    def _random_flip(self, volume):
        if random.random() < 0.5:
            return torch.flip(volume, dims=[2])
        return volume

    def _random_intensity_shift(self, volume):
        scale = 1.0 + random.uniform(-0.08, 0.08)
        bias = random.uniform(-0.08, 0.08)
        return volume * scale + bias

    def _random_noise(self, volume):
        if self.noise_std <= 0:
            return volume
        noise = torch.randn_like(volume) * self.noise_std
        return volume + noise

    def _random_cutout(self, volume):
        if self.cutout_frac <= 0 or random.random() >= 0.5:
            return volume
        _, height, width = volume.shape
        cut_h = max(1, int(height * self.cutout_frac))
        cut_w = max(1, int(width * self.cutout_frac))
        top = random.randint(0, max(0, height - cut_h))
        left = random.randint(0, max(0, width - cut_w))
        volume[:, top:top + cut_h, left:left + cut_w] = 0
        return volume

    def _random_slice_dropout(self, volume):
        if self.slice_dropout <= 0 or volume.shape[0] <= 4:
            return volume
        mask = torch.rand(volume.shape[0]) < self.slice_dropout
        if mask.all():
            mask[random.randrange(volume.shape[0])] = False
        volume[mask] = 0
        return volume

    def _random_gamma(self, volume):
        if self.gamma_jitter <= 0 or random.random() >= 0.5:
            return volume
        gamma = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
        lower = torch.quantile(volume, 0.01)
        upper = torch.quantile(volume, 0.99)
        scaled = torch.clamp((volume - lower) / (upper - lower + 1e-6), 0.0, 1.0)
        adjusted = scaled.pow(gamma) * (upper - lower) + lower
        return adjusted

    def _random_spatial_shift(self, volume):
        if self.spatial_shift_frac <= 0 or random.random() >= 0.5:
            return volume
        _, height, width = volume.shape
        max_dy = max(1, int(height * self.spatial_shift_frac))
        max_dx = max(1, int(width * self.spatial_shift_frac))
        dy = random.randint(-max_dy, max_dy)
        dx = random.randint(-max_dx, max_dx)
        shifted = torch.roll(volume, shifts=(dy, dx), dims=(1, 2))
        if dy > 0:
            shifted[:, :dy, :] = 0
        elif dy < 0:
            shifted[:, dy:, :] = 0
        if dx > 0:
            shifted[:, :, :dx] = 0
        elif dx < 0:
            shifted[:, :, dx:] = 0
        return shifted


def _compute_class_weights(labels):
    positive_counts = labels.sum(axis=0)
    negative_counts = labels.shape[0] - positive_counts
    safe_positive = np.clip(positive_counts, a_min=1.0, a_max=None)
    weights = negative_counts / safe_positive
    return torch.tensor(weights, dtype=torch.float32)


class MRMultiPlaneDataset(data.Dataset):
    def __init__(self, root_dir, train=True, planes=PLANES, mmap=True, cache_size=32, transform=None):
        super().__init__()
        split = "train" if train else "valid"
        root_dir = resolve_dataset_root(root_dir)
        self.root_dir = root_dir
        self.split = split
        self.planes = tuple(planes)
        self.mmap = mmap
        self.records = _read_split_records(root_dir, split)
        self.exam_ids = self.records["id"].tolist()
        self.labels = self.records[list(TASKS)].to_numpy(dtype=np.float32)
        self.weights = _compute_class_weights(self.labels)
        self.plane_paths = {
            plane: [
                os.path.join(root_dir, split, plane, f"{exam_id}.npy")
                for exam_id in self.exam_ids
            ]
            for plane in self.planes
        }
        self.cache_size = max(int(cache_size), 0)
        self.transform = transform
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.exam_ids)

    def _load_volume(self, plane, index):
        cache_key = (plane, index)
        if cache_key in self._cache:
            volume = self._cache.pop(cache_key)
            self._cache[cache_key] = volume
            return volume

        mmap_mode = "r" if self.mmap else None
        volume = np.load(self.plane_paths[plane][index], mmap_mode=mmap_mode)
        volume = torch.from_numpy(np.asarray(volume, dtype=np.float32))

        if self.cache_size:
            self._cache[cache_key] = volume
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return volume

    def __getitem__(self, index):
        volumes = tuple(self._load_volume(plane, index) for plane in self.planes)
        if self.transform is not None:
            volumes = tuple(self.transform(volume) for volume in volumes)
        label = torch.from_numpy(self.labels[index])
        return volumes, label, self.weights, self.exam_ids[index]
