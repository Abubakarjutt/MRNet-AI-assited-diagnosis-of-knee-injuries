import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


PLANES = ("sagittal", "coronal", "axial")
TASKS = ("abnormal", "acl", "meniscus")


def _zero_pad_exam_id(exam_id):
    return f"{int(exam_id):04d}"


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


def _compute_class_weights(labels):
    positive_counts = labels.sum(axis=0)
    negative_counts = labels.shape[0] - positive_counts
    safe_positive = np.clip(positive_counts, a_min=1.0, a_max=None)
    weights = negative_counts / safe_positive
    return torch.tensor(weights, dtype=torch.float32)


class MRMultiPlaneDataset(data.Dataset):
    def __init__(self, root_dir, train=True, planes=PLANES, mmap=True, cache_size=32):
        super().__init__()
        split = "train" if train else "valid"
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
        label = torch.from_numpy(self.labels[index])
        return volumes, label, self.weights, self.exam_ids[index]
