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
        bias_field_std=0.0,
        blur_sigma=0.0,
        motion_prob=0.0,
    ):
        self.policy = policy
        self.noise_std = max(float(noise_std), 0.0)
        self.cutout_frac = min(max(float(cutout_frac), 0.0), 0.5)
        self.slice_dropout = min(max(float(slice_dropout), 0.0), 0.5)
        self.gamma_jitter = min(max(float(gamma_jitter), 0.0), 0.5)
        self.spatial_shift_frac = min(max(float(spatial_shift_frac), 0.0), 0.2)
        self.bias_field_std = min(max(float(bias_field_std), 0.0), 0.5)
        self.blur_sigma = min(max(float(blur_sigma), 0.0), 2.0)
        self.motion_prob = min(max(float(motion_prob), 0.0), 1.0)

    def sample_plan(self, volume):
        if self.policy == "none":
            return {"policy": "none"}

        plan = {
            "policy": self.policy,
            "flip": False,
            "intensity_scale": 1.0,
            "intensity_bias": 0.0,
            "noise_std": 0.0,
            "cutout": None,
            "slice_positions": None,
            "gamma": None,
            "shift": (0, 0),
            "bias_field": None,
            "blur_kernel": None,
            "motion_axis": None,
            "motion_shift": 0,
        }

        if self.policy in {"light", "strong", "knee_mri", "knee_mri_plus", "knee_mri_research"}:
            plan["flip"] = random.random() < 0.5
            plan["intensity_scale"] = 1.0 + random.uniform(-0.08, 0.08)
            plan["intensity_bias"] = random.uniform(-0.08, 0.08)

        if self.policy in {"strong", "knee_mri", "knee_mri_plus", "knee_mri_research"}:
            if self.noise_std > 0:
                plan["noise_std"] = self.noise_std

            if self.cutout_frac > 0 and random.random() < 0.5:
                _, height, width = volume.shape
                cut_h = max(1, int(height * self.cutout_frac))
                cut_w = max(1, int(width * self.cutout_frac))
                top = random.randint(0, max(0, height - cut_h))
                left = random.randint(0, max(0, width - cut_w))
                plan["cutout"] = (top, left, cut_h, cut_w)

            if self.slice_dropout > 0 and volume.shape[0] > 4:
                slice_positions = [
                    (idx + 0.5) / float(volume.shape[0])
                    for idx in range(volume.shape[0])
                    if random.random() < self.slice_dropout
                ]
                if len(slice_positions) >= volume.shape[0]:
                    slice_positions = slice_positions[:-1]
                if slice_positions:
                    plan["slice_positions"] = slice_positions

        if self.policy in {"knee_mri_plus", "knee_mri_research"}:
            if self.gamma_jitter > 0 and random.random() < 0.5:
                plan["gamma"] = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)

            if self.spatial_shift_frac > 0 and random.random() < 0.5:
                _, height, width = volume.shape
                max_dy = max(1, int(height * self.spatial_shift_frac))
                max_dx = max(1, int(width * self.spatial_shift_frac))
                plan["shift"] = (
                    random.randint(-max_dy, max_dy),
                    random.randint(-max_dx, max_dx),
                )

        if self.policy == "knee_mri_research":
            if self.bias_field_std > 0 and random.random() < 0.6:
                plan["bias_field"] = self._sample_bias_field(volume.shape[1], volume.shape[2])
            if self.blur_sigma > 0 and random.random() < 0.35:
                plan["blur_kernel"] = self._sample_blur_kernel()
            if self.motion_prob > 0 and random.random() < self.motion_prob:
                plan["motion_axis"] = random.choice([1, 2])
                plan["motion_shift"] = random.choice([-2, -1, 1, 2])

        return plan

    def __call__(self, volume, plan=None):
        if self.policy == "none":
            return volume

        if plan is None:
            plan = self.sample_plan(volume)

        augmented = volume.clone()
        if plan.get("flip"):
            augmented = torch.flip(augmented, dims=[2])

        augmented = augmented * plan.get("intensity_scale", 1.0) + plan.get("intensity_bias", 0.0)

        noise_std = plan.get("noise_std", 0.0)
        if noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * noise_std

        cutout = plan.get("cutout")
        if cutout is not None:
            top, left, cut_h, cut_w = cutout
            augmented[:, top:top + cut_h, left:left + cut_w] = 0

        slice_positions = plan.get("slice_positions")
        if slice_positions:
            slice_indexes = sorted(
                {
                    min(volume.shape[0] - 1, max(0, int(position * volume.shape[0])))
                    for position in slice_positions
                }
            )
            if len(slice_indexes) >= volume.shape[0]:
                slice_indexes = slice_indexes[:-1]
            if slice_indexes:
                augmented[slice_indexes] = 0

        gamma = plan.get("gamma")
        if gamma is not None:
            augmented = self._apply_gamma(augmented, gamma)

        shift = plan.get("shift", (0, 0))
        if shift != (0, 0):
            augmented = self._apply_spatial_shift(augmented, shift)

        bias_field = plan.get("bias_field")
        if bias_field is not None:
            augmented = augmented * bias_field.unsqueeze(0)

        blur_kernel = plan.get("blur_kernel")
        if blur_kernel is not None:
            augmented = self._apply_blur(augmented, blur_kernel)

        motion_axis = plan.get("motion_axis")
        if motion_axis is not None:
            augmented = self._apply_motion_artifact(augmented, motion_axis, plan.get("motion_shift", 0))

        return augmented

    def _apply_gamma(self, volume, gamma):
        lower = torch.quantile(volume, 0.01)
        upper = torch.quantile(volume, 0.99)
        scaled = torch.clamp((volume - lower) / (upper - lower + 1e-6), 0.0, 1.0)
        adjusted = scaled.pow(gamma) * (upper - lower) + lower
        return adjusted

    def _apply_spatial_shift(self, volume, shift):
        dy, dx = shift
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

    def _sample_bias_field(self, height, width):
        yy = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
        xx = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        a = random.uniform(-self.bias_field_std, self.bias_field_std)
        b = random.uniform(-self.bias_field_std, self.bias_field_std)
        c = random.uniform(-self.bias_field_std * 0.5, self.bias_field_std * 0.5)
        field = 1.0 + a * grid_x + b * grid_y + c * grid_x * grid_y
        return torch.clamp(field, min=0.7, max=1.3)

    def _sample_blur_kernel(self):
        sigma = random.uniform(max(self.blur_sigma * 0.5, 1e-3), self.blur_sigma)
        radius = max(1, int(round(sigma * 2)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_blur(self, volume, kernel):
        pad = kernel.numel() // 2
        device = volume.device
        dtype = volume.dtype
        kernel = kernel.to(device=device, dtype=dtype)
        kernel_x = kernel.view(1, 1, 1, -1)
        kernel_y = kernel.view(1, 1, -1, 1)
        image = volume.unsqueeze(1)
        image = torch.nn.functional.pad(image, (pad, pad, 0, 0), mode="reflect")
        image = torch.nn.functional.conv2d(image, kernel_x.expand(1, 1, 1, kernel.numel()), groups=1)
        image = torch.nn.functional.pad(image, (0, 0, pad, pad), mode="reflect")
        image = torch.nn.functional.conv2d(image, kernel_y.expand(1, 1, kernel.numel(), 1), groups=1)
        return image.squeeze(1)

    def _apply_motion_artifact(self, volume, axis, shift):
        shifted = torch.roll(volume, shifts=shift, dims=axis)
        return 0.7 * volume + 0.3 * shifted


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
            if hasattr(self.transform, "sample_plan"):
                plan = self.transform.sample_plan(volumes[0])
                volumes = tuple(self.transform(volume, plan=plan) for volume in volumes)
            else:
                volumes = tuple(self.transform(volume) for volume in volumes)
        label = torch.from_numpy(self.labels[index])
        return volumes, label, self.weights, self.exam_ids[index]
