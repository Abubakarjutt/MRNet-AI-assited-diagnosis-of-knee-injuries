import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataloader import PLANES, TASKS

_SEED = 20260901
_SPLIT_SIZES = {"train": 6, "valid": 4}


def _write_split(root, split, n, rng):
    exam_ids = [f"{i:04d}" for i in range(n)]

    per_task_labels = {}
    for task in TASKS:
        column = [0, 1] + [rng.randint(0, 1) for _ in range(n - 2)]
        rng.shuffle(column)
        per_task_labels[task] = column

    for task in TASKS:
        with open(root / f"{split}-{task}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            for exam_id, label in zip(exam_ids, per_task_labels[task]):
                writer.writerow([exam_id, label])

    for plane in PLANES:
        plane_dir = root / split / plane
        plane_dir.mkdir(parents=True, exist_ok=True)
        for exam_id in exam_ids:
            slice_count = rng.randint(12, 28)
            array = np.random.default_rng(rng.randint(0, 2**31 - 1)).random(
                (slice_count, 256, 256), dtype=np.float32
            ) * 255.0
            np.save(plane_dir / f"{exam_id}.npy", array)


@pytest.fixture
def mrnet_fixture(tmp_path):
    rng = random.Random(_SEED)
    root = tmp_path / "MRNet-v1.0"
    root.mkdir(parents=True, exist_ok=True)
    for split, n in _SPLIT_SIZES.items():
        _write_split(root, split, n, rng)
    return root


class _StubTower(nn.Module):
    def __init__(self, feature_dim=1152):
        super().__init__()
        self.proj = nn.Linear(3, feature_dim)

    def forward(self, pixel_values):
        pooled = self.proj(pixel_values.mean(dim=(2, 3)))  # [N,3] -> [N,feature_dim]
        return SimpleNamespace(pooler_output=pooled)


class _StubSiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _StubTower()


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _StubSiglip()


@pytest.fixture
def stub_medsiglip(monkeypatch):
    monkeypatch.setattr("medical_encoders.AutoModel", _FakeAutoModel)
    return _FakeAutoModel
