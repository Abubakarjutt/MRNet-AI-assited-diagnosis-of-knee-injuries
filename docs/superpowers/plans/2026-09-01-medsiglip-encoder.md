# MedSigLIP Frozen Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--model_type medsiglip` to `train.py` — a frozen `google/medsiglip-448` vision tower used as the per-slice encoder inside `FastMRNet`, reusing the existing pooling / fusion / head / EMA / scheduler / metric path unchanged.

**Architecture:** A new `MedSigLIPEncoder` (frozen HF SigLIP vision tower, internal slice micro-batching, checkpoint-excluded) plugs into `lightweight_models.build_backbone` as a new backbone name. Preprocessing is generalized so `prepare_volume_batch` can apply SigLIP normalization (448 px, `[-1,1]`, bicubic+antialias) selected per `model_type` by a new `resolve_input_spec` resolver. `train.py`'s optimizer, freeze logic, and summary block get small pure-helper extractions so a partly-frozen model is handled and logged correctly. No change to `dataloader.py` or the metric computation.

**Tech Stack:** Python, PyTorch (MPS), `transformers` (HF SigLIP), `pytest`. Apple Silicon / MPS only.

**Spec:** `docs/superpowers/specs/2026-09-01-medsiglip-encoder-design.md` (read it alongside this plan; the plan implements that spec).

## Global Constraints

- **Platform:** Apple Silicon Mac, MPS only. No `bitsandbytes`, no CUDA assumptions.
- **`transformers>=4.56`** (for the `dtype=` kwarg and current SigLIP handling). Also add `huggingface_hub>=0.26`, `pillow`, `pytest` to `requirements.txt`.
- **`HF_TOKEN` + accepted `google/medsiglip-448` license** required at real-model runtime; all non-`slow` tests must run without it and without any model download.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** must be set before `import torch` — `tests/conftest.py` sets it via `os.environ.setdefault` at the top of the file.
- **Byte-for-byte behavior preservation** for existing `model_type`s (`resnet18`, `mobilenet_v3_small`, `efficientnet_b0`, `basic`, `advanced`, `multiscale`): every generalized function keeps defaults that reproduce today's output exactly.
- **`train.py`'s `best_val_auc` is one pooled micro-AUC** over flattened 3-task predictions (`train.py:242-248`). This plan does **not** add a per-task AUC line and does **not** touch `compute_auc` or the val loop.
- **The MedSigLIP tower is never trained** and never serialized into a `.pth`.
- Commit after every task. TDD: failing test first, minimal implementation, passing test, commit.

---

## File Structure

**New files:**
- `medical_encoders.py` — `MedSigLIPEncoder` only. One responsibility: wrap the frozen HF SigLIP vision tower behind the `[N,3,H,W] -> [N,1152]` contract `FastMRNet._encode_planes` needs.
- `tests/conftest.py` — shared pytest fixtures: `PYTORCH_ENABLE_MPS_FALLBACK` env, `mrnet_fixture` (synthetic dataset tree), `stub_medsiglip` (monkeypatched tiny tower).
- `tests/test_input_spec.py` — `resolve_input_spec` + `prepare_volume_batch` generalization.
- `tests/test_medsiglip_backbone.py` — `MedSigLIPEncoder` unit behavior + `FastMRNet("medsiglip")` integration.
- `tests/test_freeze_backbone.py` — `--freeze_backbone` tri-state, optimizer filter, arg parsing.
- `tests/test_train_summary.py` — `param_counts` + `model_complexity_score` pure helpers.
- `tests/test_medsiglip_real.py` — `@pytest.mark.slow`, real weights.
- `pytest.ini` — register the `slow` marker.
- `MEDICAL_MODELS.md` — usage doc (repo root).

**Modified files:**
- `utils.py` — `prepare_volume_batch` gains `mean` / `std` / `interp_mode` / `antialias`; add `SIGLIP_MEAN` / `SIGLIP_STD`.
- `lightweight_models.py` — `build_backbone` gains a `medsiglip` branch; "unsupported backbone" text.
- `train.py` — `resolve_input_spec`; `prepare_inputs` uses it; `--model_type` choices gain `medsiglip`; new `--freeze_backbone`, `--medsiglip_chunk`; `apply_backbone_freezing`, `build_optimizer`, `param_counts`, `model_complexity_score` helpers; `build_model` wiring; summary block gains `trainable_params_M`.
- `requirements.txt` — add `transformers>=4.56`, `huggingface_hub>=0.26`, `pillow`, `pytest`.
- `README.md` — one "Pretrained medical models" pointer.

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Create: `tests/test_infra.py` (temporary sanity test — kept, it's cheap and guards the fixture)
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `dataloader.PLANES`, `dataloader.TASKS`, `dataloader.MRMultiPlaneDataset`.
- Produces:
  - fixture `mrnet_fixture` → `pathlib.Path` to a directory containing
    `{train,valid}-{abnormal,acl,meniscus}.csv` and `{train,valid}/{sagittal,coronal,axial}/{id}.npy`
    with 6 train + 4 valid exams, each plane a `float32 [s,256,256]` array (`s ∈ [12,28]`),
    values in `[0,255]`, per-task labels guaranteed to contain both 0 and 1.
  - fixture `stub_medsiglip` (function-scoped, uses `monkeypatch`) → replaces
    `medical_encoders.AutoModel` with a fake whose `.from_pretrained(...)` returns a module
    exposing `.vision_model`, a tiny `nn.Module` mapping `pixel_values [N,3,H,W]` to an object
    with `.pooler_output` of shape `[N,1152]`. Yields the fake class.
  - `pytest.ini` registering marker `slow`.

- [ ] **Step 1: Write `requirements.txt` additions**

Append to `requirements.txt` (keep existing lines):

```
transformers>=4.56
huggingface_hub>=0.26
pillow
pytest
```

- [ ] **Step 2: Write `pytest.ini`**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: needs network, HF_TOKEN, or a large model download (deselect with -m "not slow")
```

- [ ] **Step 3: Write `tests/conftest.py`**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 4: Write `tests/test_infra.py`**

```python
import numpy as np

from dataloader import MRMultiPlaneDataset, TASKS


def test_mrnet_fixture_loads_with_dataset(mrnet_fixture):
    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=True)
    assert len(dataset) == 6

    volumes, label, _weights, exam_id = dataset[0]
    assert len(volumes) == 3
    assert volumes[0].dim() == 3 and volumes[0].shape[1:] == (256, 256)
    assert label.shape == (len(TASKS),)
    assert exam_id == "0000"


def test_mrnet_fixture_labels_have_both_classes(mrnet_fixture):
    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=True)
    labels = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
    for column in range(labels.shape[1]):
        assert set(labels[:, column].tolist()) == {0.0, 1.0}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest -m "not slow" -q`
Expected: 2 passed. (If `pytest` is not installed yet: `pip install -r requirements.txt` first.)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini tests/conftest.py tests/test_infra.py
git commit -m "test: add pytest infra, synthetic MRNet fixture, MedSigLIP stub"
```

---

## Task 2: Generalize `prepare_volume_batch`

**Files:**
- Modify: `utils.py`
- Test: `tests/test_input_spec.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `utils.SIGLIP_MEAN`, `utils.SIGLIP_STD` — `float32` tensors of shape `[1,1,3,1,1]`, all `0.5`.
  - `utils.prepare_volume_batch(volume, device, image_size=224, channels_last=False, mean=IMAGENET_MEAN, std=IMAGENET_STD, interp_mode="bilinear", antialias=False) -> Tensor[B, slices, 3, image_size, image_size]`. Defaults reproduce the pre-change output exactly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_spec.py`:

```python
import torch

import utils


def test_siglip_constants_shape_and_value():
    for tensor in (utils.SIGLIP_MEAN, utils.SIGLIP_STD):
        assert tensor.shape == (1, 1, 3, 1, 1)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.full_like(tensor, 0.5))


def test_prepare_volume_batch_default_matches_manual_imagenet():
    volume = torch.full((1, 4, 8, 8), 128.0)  # [B, slices, H, W]
    out = utils.prepare_volume_batch(volume, device=torch.device("cpu"), image_size=8)

    expected_r = (128.0 / 255.0 - 0.485) / 0.229
    assert out.shape == (1, 4, 3, 8, 8)
    assert torch.allclose(out[0, 0, 0], torch.full((8, 8), expected_r), atol=1e-5)


def test_prepare_volume_batch_siglip_spec_maps_to_unit_range():
    volume = torch.zeros((1, 3, 8, 8))
    volume[..., :, :] = 255.0
    out = utils.prepare_volume_batch(
        volume,
        device=torch.device("cpu"),
        image_size=16,
        mean=utils.SIGLIP_MEAN,
        std=utils.SIGLIP_STD,
        interp_mode="bicubic",
        antialias=True,
    )
    assert out.shape == (1, 3, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert out.max().item() <= 1.0 + 1e-4
    assert out.min().item() >= -1.0 - 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_input_spec.py -q`
Expected: FAIL — `AttributeError: module 'utils' has no attribute 'SIGLIP_MEAN'`.

- [ ] **Step 3: Implement in `utils.py`**

Add after the existing `IMAGENET_MEAN` / `IMAGENET_STD` definitions:

```python
SIGLIP_MEAN = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)
SIGLIP_STD = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)
```

Replace `prepare_volume_batch` with:

```python
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
```

Notes: `channels_last` stays in the signature (already unused in the body — kept for call-site stability). `antialias` is silently ignored by `F.interpolate` for `mode="nearest"`; only `bilinear`/`bicubic` paths are used here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_input_spec.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the full fast suite (guard against regressions)**

Run: `pytest -m "not slow" -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_input_spec.py
git commit -m "feat: generalize prepare_volume_batch normalization; add SigLIP constants"
```

---

## Task 3: `resolve_input_spec` + `prepare_inputs` wiring

**Files:**
- Modify: `train.py` (add `resolve_input_spec`; change `prepare_inputs`)
- Test: `tests/test_input_spec.py` (extend)

**Interfaces:**
- Consumes: `utils.SIGLIP_MEAN`, `utils.SIGLIP_STD`, `utils.IMAGENET_MEAN`, `utils.IMAGENET_STD`, `utils.prepare_volume_batch` (Task 2).
- Produces:
  - `train.resolve_input_spec(args) -> dict` with keys `image_size:int`, `mean:Tensor`, `std:Tensor`, `interp_mode:str`, `antialias:bool`.
    - `model_type == "medsiglip"` → `{448, SIGLIP_MEAN, SIGLIP_STD, "bicubic", True}`.
    - otherwise → `{args.image_size, IMAGENET_MEAN, IMAGENET_STD, "bilinear", False}`.
  - `train.prepare_inputs(volumes, device, args)` — unchanged signature, now forwards the resolved spec fields to `prepare_volume_batch`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_input_spec.py`:

```python
from types import SimpleNamespace

import train
import utils as utils_module


def test_resolve_input_spec_medsiglip():
    args = SimpleNamespace(model_type="medsiglip", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 448
    assert spec["interp_mode"] == "bicubic"
    assert spec["antialias"] is True
    assert spec["mean"] is utils_module.SIGLIP_MEAN
    assert spec["std"] is utils_module.SIGLIP_STD


def test_resolve_input_spec_default_backbone():
    args = SimpleNamespace(model_type="mobilenet_v3_small", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 224
    assert spec["interp_mode"] == "bilinear"
    assert spec["antialias"] is False
    assert spec["mean"] is utils_module.IMAGENET_MEAN
    assert spec["std"] is utils_module.IMAGENET_STD


def test_prepare_inputs_uses_medsiglip_spec(mrnet_fixture):
    from dataloader import MRMultiPlaneDataset

    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=False)
    volumes, _label, _w, _id = dataset[0]
    args = SimpleNamespace(model_type="medsiglip", image_size=224, channels_last=0)

    sagittal, coronal, axial = train.prepare_inputs(volumes, torch.device("cpu"), args)
    for plane in (sagittal, coronal, axial):
        assert plane.shape[-2:] == (448, 448)
        assert plane.min().item() >= -1.0 - 1e-3
        assert plane.max().item() <= 1.0 + 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_input_spec.py -q`
Expected: FAIL — `AttributeError: module 'train' has no attribute 'resolve_input_spec'`.

- [ ] **Step 3: Implement in `train.py`**

Add above `prepare_inputs` (near `train.py:90`):

```python
def resolve_input_spec(args):
    if args.model_type == "medsiglip":
        return {
            "image_size": 448,
            "mean": utils.SIGLIP_MEAN,
            "std": utils.SIGLIP_STD,
            "interp_mode": "bicubic",
            "antialias": True,
        }
    return {
        "image_size": args.image_size,
        "mean": utils.IMAGENET_MEAN,
        "std": utils.IMAGENET_STD,
        "interp_mode": "bilinear",
        "antialias": False,
    }
```

Replace the body of `prepare_inputs`:

```python
def prepare_inputs(volumes, device, args):
    channels_last = bool(args.channels_last)
    spec = resolve_input_spec(args)
    sagittal, coronal, axial = (
        utils.prepare_volume_batch(
            volume,
            device=device,
            image_size=spec["image_size"],
            channels_last=channels_last,
            mean=spec["mean"],
            std=spec["std"],
            interp_mode=spec["interp_mode"],
            antialias=spec["antialias"],
        )
        for volume in volumes
    )
    return sagittal, coronal, axial
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_input_spec.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_input_spec.py
git commit -m "feat: per-model input spec resolver; wire into prepare_inputs"
```

---

## Task 4: `MedSigLIPEncoder`

**Files:**
- Create: `medical_encoders.py`
- Test: `tests/test_medsiglip_backbone.py`

**Interfaces:**
- Consumes: `transformers.AutoModel` (module-level import so tests can monkeypatch `medical_encoders.AutoModel`); `stub_medsiglip` fixture (Task 1).
- Produces:
  - `medical_encoders.MedSigLIPEncoder(model_id="google/medsiglip-448", chunk_size=32)`.
    - attributes: `.feature_dim == 1152` (int), `.tower` (`nn.Module`, all params `requires_grad=False`, always `.eval()`), `.chunk_size` (int, mutable).
    - `forward(flat_inputs: FloatTensor[N,3,448,448]) -> FloatTensor[N,1152]`; runs the tower in `chunk_size`-sized micro-batches under `torch.no_grad()`; output is a plain float tensor.
    - `train(mode=True)` — sets parent mode but forces `self.tower.eval()`.
    - `state_dict(*args, **kwargs)` — contributes nothing (returns a passed-in `destination`, or an empty `OrderedDict`), so the tower is excluded from `FastMRNet.state_dict()` / `torch.save`.
    - `_load_from_state_dict(...)` — no-op.

- [ ] **Step 1: Write the failing test**

Create `tests/test_medsiglip_backbone.py`:

```python
import torch

import medical_encoders


def test_encoder_forward_shape_and_frozen(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    assert encoder.feature_dim == 1152
    assert all(not p.requires_grad for p in encoder.parameters())

    out = encoder(torch.randn(5, 3, 448, 448))
    assert out.shape == (5, 1152)
    assert out.dtype == torch.float32


def test_encoder_train_keeps_tower_in_eval(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    encoder.train()
    assert encoder.training is True
    assert encoder.tower.training is False


def test_encoder_chunk_size_is_invariant(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder(chunk_size=64)
    x = torch.randn(10, 3, 448, 448)

    encoder.chunk_size = 3
    small = encoder(x)
    encoder.chunk_size = 64
    big = encoder(x)

    assert torch.allclose(small, big, atol=1e-5)


def test_encoder_state_dict_is_empty(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    assert dict(encoder.state_dict()) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_medsiglip_backbone.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'medical_encoders'`.

- [ ] **Step 3: Implement `medical_encoders.py`**

```python
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel


class MedSigLIPEncoder(nn.Module):
    """Frozen SigLIP vision tower from google/medsiglip-448.

    Contract required by FastMRNet._encode_planes:
        forward(flat_inputs: FloatTensor[N, 3, 448, 448]) -> FloatTensor[N, 1152]
    """

    feature_dim = 1152

    def __init__(self, model_id="google/medsiglip-448", chunk_size=32):
        super().__init__()
        full = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        # Prefer the dedicated vision tower; some transformers versions expose it
        # only on the composite model.
        self.tower = getattr(full, "vision_model", full)
        self.feature_dim = 1152
        self.chunk_size = int(chunk_size)

        for parameter in self.tower.parameters():
            parameter.requires_grad_(False)
        self.tower.eval()

    def train(self, mode=True):
        super().train(mode)
        self.tower.eval()
        return self

    @torch.no_grad()
    def forward(self, flat_inputs):
        flat_inputs = flat_inputs.to(dtype=torch.float32)
        pooled_chunks = []
        for start in range(0, flat_inputs.shape[0], self.chunk_size):
            chunk = flat_inputs[start:start + self.chunk_size]
            output = self.tower(pixel_values=chunk)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output[1] if isinstance(output, (tuple, list)) else output
            pooled_chunks.append(pooled.float())
        return torch.cat(pooled_chunks, dim=0)

    def state_dict(self, *args, **kwargs):
        destination = kwargs.get("destination")
        if destination is None and args:
            destination = args[0]
        if destination is None:
            destination = OrderedDict()
        return destination

    def _load_from_state_dict(self, *args, **kwargs):
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_medsiglip_backbone.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add medical_encoders.py tests/test_medsiglip_backbone.py
git commit -m "feat: MedSigLIPEncoder — frozen SigLIP tower, chunked, checkpoint-excluded"
```

---

## Task 5: `build_backbone` medsiglip branch

**Files:**
- Modify: `lightweight_models.py`
- Test: `tests/test_medsiglip_backbone.py` (extend)

**Interfaces:**
- Consumes: `medical_encoders.MedSigLIPEncoder` (Task 4).
- Produces: `lightweight_models.build_backbone("medsiglip", pretrained) -> (MedSigLIPEncoder, 1152)`. `pretrained` is accepted and ignored (weights always come from the HF checkpoint). `FastMRNet(backbone_name="medsiglip", ...)` works end-to-end.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_medsiglip_backbone.py`:

```python
import lightweight_models
from lightweight_models import FastMRNet


def test_build_backbone_medsiglip(stub_medsiglip):
    encoder, feature_dim = lightweight_models.build_backbone("medsiglip", pretrained=1)
    assert isinstance(encoder, medical_encoders.MedSigLIPEncoder)
    assert feature_dim == 1152


def test_fastmrnet_medsiglip_forward(stub_medsiglip):
    model = FastMRNet(
        backbone_name="medsiglip",
        num_classes=3,
        pretrained=True,
        pooling="gem",
        plane_fusion="plane_attention",
        fusion_depth=3,
        hidden_dim=192,
        dropout=0.15,
    )
    model.eval()

    planes = [torch.randn(1, s, 3, 448, 448) for s in (6, 5, 7)]
    logits = model(*planes)
    assert logits.shape == (1, 3)


def test_fastmrnet_medsiglip_checkpoint_excludes_tower(stub_medsiglip, tmp_path):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    state = model.state_dict()
    assert not any(key.startswith("encoder.tower") for key in state)

    path = tmp_path / "m.pth"
    torch.save(state, path)
    reloaded = torch.load(path, weights_only=True)
    result = model.load_state_dict(reloaded, strict=False)
    assert all("encoder.tower" not in key for key in result.unexpected_keys)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_medsiglip_backbone.py -q`
Expected: FAIL — `ValueError: Unsupported backbone: medsiglip` (from `build_backbone`).

- [ ] **Step 3: Implement in `lightweight_models.py`**

In `build_backbone`, before the final `else` that raises, add:

```python
    elif name == "medsiglip":
        from medical_encoders import MedSigLIPEncoder

        encoder = MedSigLIPEncoder()  # `pretrained` intentionally ignored
        return encoder, encoder.feature_dim
```

Update the `else` message string to list `medsiglip`:

```python
        raise ValueError(
            "Unsupported backbone. Choose from: resnet18, mobilenet_v3_small, "
            "efficientnet_b0, medsiglip"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_medsiglip_backbone.py -q`
Expected: 7 passed.

- [ ] **Step 5: Run the fast suite**

Run: `pytest -m "not slow" -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add lightweight_models.py tests/test_medsiglip_backbone.py
git commit -m "feat: build_backbone medsiglip branch; FastMRNet integration"
```

---

## Task 6: `--freeze_backbone` / `--medsiglip_chunk` args + `build_model` wiring

**Files:**
- Modify: `train.py` (arg parser; `apply_backbone_freezing`; `build_model`)
- Test: `tests/test_freeze_backbone.py`

**Interfaces:**
- Consumes: `build_model` (existing), `FastMRNet` (Task 5), `stub_medsiglip` fixture.
- Produces:
  - CLI: `--model_type` choices include `"medsiglip"`; `--freeze_backbone` (`type=str`, `choices=["auto","0","1"]`, default `"auto"`); `--medsiglip_chunk` (`type=int`, default `32`).
  - `train.apply_backbone_freezing(model, args) -> None`:
    - `args.freeze_backbone == "1"` → freeze `model.encoder` (any backbone).
    - `args.freeze_backbone == "auto"` and `args.model_type == "medsiglip"` → freeze `model.encoder`.
    - `args.freeze_backbone == "0"` and `args.model_type == "medsiglip"` → leave as-is (tower already frozen in `__init__`) and print a one-line notice to stdout.
    - otherwise → no-op.
    - safe when `model` has no `.encoder`.
  - `build_model(args)` calls `apply_backbone_freezing` and, for `medsiglip`, sets `model.encoder.chunk_size = args.medsiglip_chunk`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_freeze_backbone.py`:

```python
from types import SimpleNamespace

import pytest

import train
from lightweight_models import FastMRNet


def _args(**overrides):
    base = dict(
        model_type="mobilenet_v3_small",
        freeze_backbone="auto",
        medsiglip_chunk=32,
        pretrained=1,
        dropout=0.2,
        pooling="max",
        projection_dim=0,
        hidden_dim=256,
        fusion_depth=2,
        fusion_gate="none",
        plane_fusion="concat",
        plane_transformer_heads=4,
        vit_model="vit_b_16",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_auto_leaves_cnn_backbone_trainable():
    model = FastMRNet(backbone_name="mobilenet_v3_small", num_classes=3, pretrained=False)
    train.apply_backbone_freezing(model, _args(model_type="mobilenet_v3_small"))
    assert any(p.requires_grad for p in model.encoder.parameters())


def test_auto_freezes_medsiglip_encoder(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(model, _args(model_type="medsiglip"))
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_explicit_one_freezes_cnn_backbone():
    model = FastMRNet(backbone_name="mobilenet_v3_small", num_classes=3, pretrained=False)
    train.apply_backbone_freezing(model, _args(model_type="mobilenet_v3_small", freeze_backbone="1"))
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_explicit_zero_on_medsiglip_warns_and_stays_frozen(stub_medsiglip, capsys):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(model, _args(model_type="medsiglip", freeze_backbone="0"))
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert "medsiglip" in capsys.readouterr().out.lower()


def test_build_model_sets_chunk_size(stub_medsiglip):
    model = train.build_model(_args(model_type="medsiglip", medsiglip_chunk=8))
    assert model.encoder.chunk_size == 8


def test_argparser_accepts_medsiglip(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--prefix_name", "x", "--model_type", "medsiglip"])
    parsed = train.parse_arguments()
    assert parsed.model_type == "medsiglip"
    assert parsed.freeze_backbone == "auto"
    assert parsed.medsiglip_chunk == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_freeze_backbone.py -q`
Expected: FAIL — `AttributeError: module 'train' has no attribute 'apply_backbone_freezing'`.

- [ ] **Step 3: Implement in `train.py`**

Add the helper near `build_model` (`train.py:267`):

```python
def apply_backbone_freezing(model, args):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return

    mode = getattr(args, "freeze_backbone", "auto")
    is_medsiglip = getattr(args, "model_type", "") == "medsiglip"

    if mode == "1" or (mode == "auto" and is_medsiglip):
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    elif mode == "0" and is_medsiglip:
        print(
            "[freeze_backbone=0] ignored for medsiglip: the MedSigLIP tower is "
            "frozen by construction."
        )
```

In `build_model`, after the model is constructed and before `return model`:

```python
    apply_backbone_freezing(model, args)
    if args.model_type == "medsiglip":
        model.encoder.chunk_size = args.medsiglip_chunk

    return model
```

In `parse_arguments`, add `"medsiglip"` to the `--model_type` `choices` list, and add:

```python
    parser.add_argument(
        "--freeze_backbone",
        type=str,
        choices=["auto", "0", "1"],
        default="auto",
        help="auto = freeze only the medsiglip encoder; 1/0 force freeze/trainable for any backbone.",
    )
    parser.add_argument(
        "--medsiglip_chunk",
        type=int,
        default=32,
        help="Slice micro-batch size inside the frozen MedSigLIP encoder (MPS memory bound).",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_freeze_backbone.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_freeze_backbone.py
git commit -m "feat: --freeze_backbone tri-state, --medsiglip_chunk, build_model wiring"
```

---

## Task 7: Optimizer filter + summary-block helpers

**Files:**
- Modify: `train.py` (`build_optimizer`, `param_counts`, `model_complexity_score` helpers; rewire `run`)
- Test: `tests/test_train_summary.py`

**Interfaces:**
- Consumes: `FastMRNet` (Task 5), `apply_backbone_freezing` (Task 6), `stub_medsiglip`.
- Produces:
  - `train.build_optimizer(model, args) -> torch.optim.AdamW` over `p for p in model.parameters() if p.requires_grad`, using `args.lr`, `args.weight_decay`.
  - `train.param_counts(model) -> tuple[float, float]` = `(total_millions, trainable_millions)`.
  - `train.model_complexity_score(args) -> float` — extraction of the inline block at `train.py:519-549`, with `"medsiglip": 0.6` added to the backbone term. Byte-identical results for existing `model_type`s.
  - `run` uses all three; the summary block gains a `trainable_params_M:` line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_summary.py`:

```python
from types import SimpleNamespace

import torch

import train
from lightweight_models import FastMRNet


def _complexity_args(**overrides):
    base = dict(
        model_type="mobilenet_v3_small",
        pooling="max",
        aug_policy="none",
        fusion_depth=2,
        fusion_gate="none",
        plane_fusion="concat",
        val_tta_mode="none",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_model_complexity_matches_legacy_formula():
    # legacy inline values, recomputed here as the reference
    def legacy(args):
        model_c = {"mobilenet_v3_small": 0.0, "resnet18": 0.2, "efficientnet_b0": 0.4}.get(
            args.model_type, 0.0
        )
        pooling_c = {"max": 0.0, "mean": 0.0, "lse": 0.2, "gem": 0.3, "attention": 0.5}.get(
            args.pooling, 0.3
        )
        aug_c = {
            "none": 0.0, "light": 0.05, "strong": 0.12,
            "knee_mri": 0.18, "knee_mri_plus": 0.24, "knee_mri_research": 0.30,
        }.get(args.aug_policy, 0.08)
        fusion_c = 0.2 * max(int(args.fusion_depth) - 1, 0)
        gate_c = 0.15 * (1 if args.fusion_gate == "se" else 0)
        plane_c = {"concat": 0.0, "plane_attention": 0.12, "plane_transformer": 0.22}.get(
            args.plane_fusion, 0.0
        )
        tta_c = 0.05 if args.val_tta_mode != "none" else 0.0
        return model_c + pooling_c + fusion_c + gate_c + aug_c + plane_c + tta_c

    for args in (
        _complexity_args(),
        _complexity_args(model_type="resnet18", pooling="gem", fusion_depth=3),
        _complexity_args(model_type="efficientnet_b0", aug_policy="knee_mri_plus",
                         fusion_gate="se", plane_fusion="plane_transformer",
                         val_tta_mode="flip"),
    ):
        assert train.model_complexity_score(args) == legacy(args)


def test_model_complexity_medsiglip_branch():
    assert train.model_complexity_score(_complexity_args(model_type="medsiglip")) == 0.6


def test_param_counts_reports_trainable_subset(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(
        model, SimpleNamespace(model_type="medsiglip", freeze_backbone="auto")
    )
    total_m, trainable_m = train.param_counts(model)
    assert trainable_m < total_m
    assert trainable_m > 0.0


def test_build_optimizer_only_sees_trainable_params(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(
        model, SimpleNamespace(model_type="medsiglip", freeze_backbone="auto")
    )
    args = SimpleNamespace(lr=3e-4, weight_decay=1e-4)
    optimizer = train.build_optimizer(model, args)

    in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert in_optimizer == trainable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_summary.py -q`
Expected: FAIL — `AttributeError: module 'train' has no attribute 'model_complexity_score'`.

- [ ] **Step 3: Implement the helpers in `train.py`**

Add near `build_model`:

```python
def build_optimizer(model, args):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)


def param_counts(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total / 1e6, trainable / 1e6


def model_complexity_score(args):
    model_complexity = {
        "mobilenet_v3_small": 0.0,
        "resnet18": 0.2,
        "efficientnet_b0": 0.4,
        "medsiglip": 0.6,
    }.get(args.model_type, 0.0)
    pooling_complexity = {
        "max": 0.0, "mean": 0.0, "lse": 0.2, "gem": 0.3, "attention": 0.5
    }.get(args.pooling, 0.3)
    augmentation_complexity = {
        "none": 0.0, "light": 0.05, "strong": 0.12,
        "knee_mri": 0.18, "knee_mri_plus": 0.24, "knee_mri_research": 0.30,
    }.get(args.aug_policy, 0.08)
    fusion_complexity = 0.2 * max(int(args.fusion_depth) - 1, 0)
    gate_complexity = 0.15 * (1 if args.fusion_gate == "se" else 0)
    plane_fusion_complexity = {
        "concat": 0.0, "plane_attention": 0.12, "plane_transformer": 0.22
    }.get(args.plane_fusion, 0.0)
    tta_complexity = 0.05 if args.val_tta_mode != "none" else 0.0
    return (
        model_complexity
        + pooling_complexity
        + fusion_complexity
        + gate_complexity
        + augmentation_complexity
        + plane_fusion_complexity
        + tta_complexity
    )
```

Now rewire `run`:

- Replace the optimizer construction (`train.py:403`) with:
  ```python
  optimizer = build_optimizer(model, args)
  ```
- Replace the setup-time param logging (`train.py:399-401`) with:
  ```python
  num_params_m, trainable_params_m = param_counts(model)
  print(
      f"Model: {args.model_type}, Parameters: {num_params_m:.2f}M, "
      f"Trainable: {trainable_params_m:.2f}M"
  )
  ```
- Replace the inline complexity block (`train.py:519-549`, from `model_complexity = 0.0` through the `total_complexity = (...)` assignment) with:
  ```python
  total_complexity = model_complexity_score(args)
  ```
- In the final summary block (`train.py:551-561`), replace the `num_params` recompute (`train.py:516`) and add the trainable line:
  ```python
  num_params_m, trainable_params_m = param_counts(model)
  ...
  print(f"num_params_M:       {num_params_m:.2f}")
  print(f"trainable_params_M: {trainable_params_m:.2f}")
  ```
  (Keep every other existing print line in that block unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_summary.py -q`
Expected: 4 passed.

- [ ] **Step 5: Run the full fast suite**

Run: `pytest -m "not slow" -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train_summary.py
git commit -m "refactor: extract build_optimizer/param_counts/model_complexity_score; log trainable params"
```

---

## Task 8: Real-weights slow test + documentation

**Files:**
- Create: `tests/test_medsiglip_real.py`
- Create: `MEDICAL_MODELS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above; real `google/medsiglip-448` weights (network + `HF_TOKEN`).
- Produces: a `@pytest.mark.slow` end-to-end check; user-facing docs.

- [ ] **Step 1: Write the slow test**

Create `tests/test_medsiglip_real.py`:

```python
import os

import pytest
import torch

import medical_encoders

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN + accepted license")
def test_real_medsiglip_encoder_forward():
    encoder = medical_encoders.MedSigLIPEncoder(chunk_size=2)
    out = encoder(torch.rand(3, 3, 448, 448))
    assert out.shape == (3, 1152)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run it (deselected in the fast gate)**

Run: `pytest -m "not slow" -q` → the real test is not collected.
Run (optional, if `HF_TOKEN` is set and you want to verify weights): `pytest tests/test_medsiglip_real.py -q -m slow`
Expected: 1 passed (or skipped without `HF_TOKEN`).

- [ ] **Step 3: Write `MEDICAL_MODELS.md`**

Create `MEDICAL_MODELS.md` at the repo root:

```markdown
# Medical Pretrained Models

## MedSigLIP frozen encoder (`--model_type medsiglip`)

Uses the vision tower of [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)
as a frozen per-slice encoder inside `FastMRNet`. Slice pooling, plane fusion, the
classifier head, EMA, the LR scheduler, and the validation AUC are all the existing
`train.py` machinery — only the encoder changes.

### One-time setup

1. Accept the Health AI Developer Foundations license on the model page.
2. `export HF_TOKEN=...`
3. `export PYTORCH_ENABLE_MPS_FALLBACK=1` (must be set before Python starts; some
   `F.interpolate` bicubic/antialias paths fall back to CPU on MPS).
4. `pip install -r requirements.txt`

The tower (~1.8 GB) downloads once to the HF cache. `--pretrained 0` is a no-op for this
model type. `--image_size` is forced to 448.

### Usage

```bash
python train.py \
  --prefix_name medsiglip_gem_attn \
  --model_type medsiglip \
  --pooling gem --plane_fusion plane_attention \
  --fusion_depth 3 --hidden_dim 192 --dropout 0.15 \
  --lr 3e-4 --weight_decay 5e-4 --ema_decay 0.995 \
  --data_root MRNet-v1.0
```

### Flags

| flag | default | meaning |
| --- | --- | --- |
| `--freeze_backbone` | `auto` | `auto` freezes only the medsiglip encoder; `1`/`0` force freeze/trainable for any backbone |
| `--medsiglip_chunk` | `32` | slices per micro-batch through the frozen tower (lower it if MPS memory is tight) |

### Notes

- The frozen tower is excluded from saved `.pth` checkpoints; it always reloads from the
  HF cache. `num_params_M` in the run summary includes it (~430M); `trainable_params_M`
  is the real trainable size (~1–5M).
- Encoding ~75–120 slices per exam through a 400M ViT on MPS is slow (minutes/epoch).
  Use `--max_train_batches` / `--time_budget_minutes` for short loops.
- Resize is bicubic + antialias to approximate the SigLIP processor; it is not a
  byte-exact match to `transformers`' PIL pipeline.
```

- [ ] **Step 4: Add the README pointer**

In `README.md`, under the "Requirements" or "Core Files" area, add:

```markdown
## Pretrained medical models

`--model_type medsiglip` uses a frozen MedSigLIP vision encoder. See
[`MEDICAL_MODELS.md`](MEDICAL_MODELS.md).
```

- [ ] **Step 5: Full fast suite + commit**

Run: `pytest -m "not slow" -q`
Expected: all passed.

```bash
git add tests/test_medsiglip_real.py MEDICAL_MODELS.md README.md
git commit -m "docs: MEDICAL_MODELS.md, README pointer, slow real-weights test"
```

---

## Manual verification (after all tasks, needs the real dataset + HF_TOKEN)

Not automated — run once by hand to confirm the wiring end-to-end:

```bash
export HF_TOKEN=...
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train.py --prefix_name medsiglip_smoke --model_type medsiglip \
  --pooling gem --plane_fusion plane_attention \
  --data_root MRNet-v1.0 --epochs 1 --max_train_batches 4 --max_val_batches 4
```

Confirm: it prints `Model: medsiglip, Parameters: ~430.xxM, Trainable: <5M`, runs a train
and val pass without OOM, writes one `models/model_medsiglip_smoke_*.pth`, and the summary
block shows `trainable_params_M:` and `model_complexity: 0.60+`.

---

## Self-Review

**Spec coverage:**
- Frozen MedSigLIP tower as `FastMRNet` encoder, `feature_dim 1152` → Tasks 4, 5.
- `--pooling` / `--plane_fusion` / etc. still work → Task 5 (`test_fastmrnet_medsiglip_forward` uses `gem` + `plane_attention` + `fusion_depth 3`).
- MPS-safe preprocessing (448, `[-1,1]`, bicubic+antialias) via `prepare_volume_batch` + `resolve_input_spec` → Tasks 2, 3.
- Memory-bounded encoding (micro-batching) → Task 4 (`test_encoder_chunk_size_is_invariant`), `--medsiglip_chunk` → Task 6.
- Tower excluded from checkpoints without breaking `torch.save` / warm-start → Task 4 (`test_encoder_state_dict_is_empty`), Task 5 (`test_fastmrnet_medsiglip_checkpoint_excludes_tower`).
- Existing loop / metric / checkpoint path unchanged → no task touches `iterate_epoch`, `compute_auc`, the val loop, or the checkpoint-on-best block.
- `--freeze_backbone` tri-state, default `auto` preserves CNN behavior → Task 6.
- Optimizer `requires_grad` filter → Task 7 (`build_optimizer`).
- `trainable_params_M` in summary; `model_complexity` `medsiglip` branch → Task 7.
- Tests run without dataset (synthetic fixture) and without model (stub); real behind `slow` → Tasks 1, 8.
- `MEDICAL_MODELS.md` + README pointer → Task 8.
- `requirements.txt`: `transformers>=4.56`, `huggingface_hub>=0.26`, `pillow`, `pytest` → Task 1.
- Deferred (no task, intentional per spec §8): embedding cache; adding `medsiglip` to the autoresearch search space.
- `--image_size` forced-448 stderr note (spec §3.3): folded into `resolve_input_spec` behavior + documented in `MEDICAL_MODELS.md`; an explicit `print` to stderr can be added in Task 3 Step 3 if desired — low value, not separately tested.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step has literal code. The one spec item intentionally left as prose (the `--image_size` note) is called out above.

**Type consistency:**
- `resolve_input_spec` returns a dict with `image_size/mean/std/interp_mode/antialias` — consumed with those exact keys in `prepare_inputs` (Task 3).
- `MedSigLIPEncoder.forward: [N,3,448,448] -> [N,1152]`; `feature_dim` is both a class attr and set on the instance (int `1152`) — `build_backbone` returns `encoder.feature_dim` (Task 5).
- `apply_backbone_freezing(model, args) -> None`; `build_optimizer(model, args) -> AdamW`; `param_counts(model) -> (float, float)`; `model_complexity_score(args) -> float` — names match between Tasks 6, 7 and their tests.
- `stub_medsiglip` patches `medical_encoders.AutoModel`; `medical_encoders.py` imports `AutoModel` at module level (Task 4 Step 3) — consistent.
