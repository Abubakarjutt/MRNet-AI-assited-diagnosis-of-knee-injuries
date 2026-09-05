# Multi-Encoder Per-Task Feature-Bank Classifier for MRNet — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-run, three-head MRNet classifier that trains on *cached* features from a bank of frozen pretrained encoders (MedSigLIP-448 + DINOv2 ViT-B/14), with a per-task head design and a multi-scale pyramid head for meniscus, aiming to beat the published 120-val per-task AUCs (meniscus > ~0.91, ACL > ~0.97).

**Architecture:** An offline script encodes every exam's slices through each frozen encoder once and writes fp16 feature files (pooled vectors + patch grids) plus a manifest. A new `FeatureCacheDataset` streams those cached features as fixed-shape `[B, S, D]` batches. `FeatBankMRNet` projects each encoder into a common `d_model`, then routes to three task heads (abnormal: 3 planes; acl: sagittal+coronal; meniscus: sagittal+coronal + a patch-grid pyramid path). Training runs in a dedicated `run_featbank` loop that reuses `train.py`'s `ModelEMA`, focal loss, AUC helper, and checkpoint conventions, but adds per-task AUC and a `per_task_mean` selection metric. All hyperparameter and threshold decisions are made by 5-fold CV over the 1130 train exams; the 120-val AUC is read once per phase.

**Tech Stack:** Python, PyTorch (MPS), torchvision (rotation aug at cache time), `transformers` `AutoModel` (DINOv2, MedSigLIP — already a dependency), scikit-learn (`roc_auc_score`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-multiencoder-pertask-mrnet-design.md` (read it alongside this plan; §11.1 records the resolved open questions this plan follows).

## Global Constraints

- **Label / plane order is fixed:** `dataloader.TASKS == ("abnormal", "acl", "meniscus")`, `dataloader.PLANES == ("sagittal", "coronal", "axial")`. Every `[N, 3]` logit/label tensor is in `TASKS` order. Never hard-code a different order.
- **Frozen encoders only in the offline cache script.** Encoder wrappers call `.requires_grad_(False)`, `.eval()`, and a `@torch.no_grad()` forward, and run slices in micro-batches (`chunk_size`, default 32). No encoder is ever constructed inside a training loop in this plan.
- **Dependency isolation:** all `transformers` / `open_clip` imports live in `feature_bank.py` (mirroring `medical_encoders.py`). `feature_cache_io.py`, `dataloader.py`, `lightweight_models.py`, `featbank_train.py` must not import `transformers`.
- **Fast-suite invariants (pytest, default `-m "not slow"`):** no test downloads a model, requires MPS, or requires `HF_TOKEN`. New fast tests use `feature_bank.FakeBankEncoder` and synthetic tensors only. Real-encoder tests are `@pytest.mark.slow`; the MedSigLIP one also `@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), ...)`.
- **Cache format:** files at `feature_cache/<encoder>/<variant>/<split>/<exam_id>.pt`; tensors stored `float16`, returned `float32`. A `feature_cache/manifest.json` records encoders, dims, variants, splits, slice params, `want_patch_for`, git sha, schema version. `SCHEMA_VERSION = 1`.
- **Selection metric default is resolved after arg parsing:** `per_task_mean` when `model_type == "featbank"`, `loss` (today's behaviour) otherwise. Passing `--select_metric` always wins.
- **Slice counts:** cache is built at 32 slices/plane; the dataset subsamples to `--slices_used` (default 24), and the meniscus patch path may use `--slices_used_meniscus` (default 32). Subsampling never upsamples — `slices_used <= slices_per_plane` is asserted.
- **CV-for-selection:** `scripts/cv_select.py` runs 5-fold CV over the 1130 train exams for all tuning. The 120-val number is produced by a normal `run_featbank` call with no `--cv_fold`, read once per phase.
- **Disk gate:** `build_feature_cache.py` refuses to start a patch-grid pass when free disk `< 16 GiB` unless `--allow_low_disk` is passed.
- **Commits:** one commit per task, message body ending with the two trailers:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4
  ```
- **Out of scope for this plan:** BiomedCLIP / phase 2 (spec §11.1 R1 — added later only if C2 misses its bar; `--encoders` is already a comma list so it is a re-run, not code); the optional live-augmentation polish run (spec §8); competition-portal packaging.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `feature_cache_io.py` (new) | Pure disk I/O for the feature cache: atomic per-exam `.pt` save/load (fp16↔fp32), manifest read/write/compat-check, path construction. No model or `transformers` import. |
| `feature_bank.py` (new) | Frozen encoder wrappers behind one `FrozenEncoder` protocol: `FakeBankEncoder` (tests), `DINOv2Encoder`, `MedSigLIPBankEncoder` (wraps Path-1 `medical_encoders.MedSigLIPEncoder`), and `build_encoder(name, ...)`. Sole home of `transformers` imports for this feature. |
| `medical_encoders.py` (modify) | `MedSigLIPEncoder.forward` gains an optional `want_patch=False`; when true it also returns the token grid. Default path stays byte-identical. |
| `scripts/build_feature_cache.py` (new) | Offline pass: for each `(encoder, variant, split, exam)` run the frozen encoder in slice chunks and write the cache file. Idempotent, resumable, writes the manifest. Importable `build_cache(...)` for tests. |
| `dataloader.py` (modify) | Add `FeatureCacheDataset` (+ `featbank_collate`). Reads the cache, subsamples slices, picks a per-epoch variant, yields a 4-tuple `(payload, label, weights, exam_id)` shaped like `MRMultiPlaneDataset`. Existing classes untouched. |
| `lightweight_models.py` (modify) | Add `FeatBankMRNet` (`consumes_feature_batch = True`). Shared per-`(encoder, plane)` projection trunk + three task heads; reuses `GeMPool1D`, `AttentionMILPool`, `PlaneAttentionFusion`. |
| `featbank_train.py` (new) | `run_featbank(args)` — the feature-bank training loop. Reuses `train.ModelEMA`, `train.compute_auc`, `train.compute_loss`, `train.smooth_targets`. Adds per-task AUC, `select_metric`, CV-fold support. Also exposes `make_folds(n, k, seed)`. Returns a metrics dict. |
| `train.py` (modify) | `run()` dispatches to `run_featbank` for `model_type == "featbank"` before the `batch_size` guard; `build_model` gets a guard; new CLI args; `--model_type` choices gain `featbank`; post-parse `select_metric` default resolution. |
| `scripts/cv_select.py` (new) | Drives `make_folds` + `run_featbank` across folds (optionally sweeping `d_model`), writes `cv_results.tsv`. |
| `MEDICAL_MODELS.md` (modify) | New "Path 3 — Multi-encoder feature bank" section: setup, cache build, C1/C2 commands, CV command, expected numbers, flags table. |
| `tests/conftest.py` (modify) | Add a `feature_cache_fixture` (builds a tiny `FakeBankEncoder` cache from `mrnet_fixture`). |
| `tests/test_feature_cache_io.py`, `tests/test_feature_bank.py`, `tests/test_build_feature_cache.py`, `tests/test_feature_cache_dataset.py`, `tests/test_featbank_model.py`, `tests/test_featbank_train.py`, `tests/test_cv_select.py` (new, fast) | Unit coverage using fakes only. |
| `tests/test_feature_bank_real.py` (new, slow) | Real DINOv2 + real MedSigLIP `encode_slices` shape checks. |

**Note on the spec's "1-line `train.py` change" (spec §5.4):** the shared `iterate_epoch` unpacks volume tensors and calls `prepare_inputs` / `forward_with_eval_policy`, so threading a feature-batch path through it risks regressing Paths 1–2. This plan instead isolates the new loop in `featbank_train.py` and makes `train.py`'s change a single early dispatch in `run()` plus new args. Net risk to existing paths: nil (the dispatch is guarded on `model_type == "featbank"`, a new choice).

**Note on `PlaneAttentionFusion` output shape:** the spec's head sketches write `PlaneAttentionFusion(d_model) -> [B, d_model]`, but the repo's `PlaneAttentionFusion.forward` returns `[B, n_planes * d_model]` (it reshapes, it does not reduce). Every head in `FeatBankMRNet` therefore appends `nn.Linear(n_planes * d_model, d_model)` right after the fusion, so the downstream shapes match the spec exactly (`meniscus` = `concat(pooled[d_model], pyramid[d_model]) -> [B, 2*d_model] -> MLP`).

---

## Task 1: Feature-cache disk I/O (`feature_cache_io.py`)

**Files:**
- Create: `feature_cache_io.py`
- Test: `tests/test_feature_cache_io.py`

**Interfaces:**
- Consumes: nothing (stdlib + `torch` only).
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `exam_cache_path(cache_dir: str, encoder: str, variant: str, split: str, exam_id: str) -> str`
  - `save_exam(path: str, planes: dict[str, dict[str, torch.Tensor]]) -> None` — `planes` maps each plane name to `{"pooled": FloatTensor[S, D]}` and optionally `{"patch": FloatTensor[S, Dp, h, w]}`. Writes fp16 via a `<path>.tmp` + `os.replace` atomic swap; creates parent dirs.
  - `load_exam(path: str) -> dict[str, dict[str, torch.Tensor]]` — returns the same structure with `float32` tensors.
  - `write_manifest(cache_dir, *, encoders: dict, variants: list[str], splits: list[str], slices_per_plane: int, slice_strategy: str, want_patch_for: list[str], git_sha: str) -> None` — `encoders` maps name -> `{"pooled_dim": int, "patch_dim": int, "patch_grid": [h, w]}`. Merges with an existing manifest (union of encoders/variants/splits).
  - `read_manifest(cache_dir) -> dict`
  - `manifest_compatible(manifest: dict, *, encoders: list[str], variants: list[str], slices_per_plane: int) -> bool` — schema version matches and the requested encoders/variants/slice count are all present.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_cache_io.py
import os
import torch
import pytest
import feature_cache_io as fcio


def _exam(with_patch=False):
    planes = {p: {"pooled": torch.randn(8, 12)} for p in ("sagittal", "coronal", "axial")}
    if with_patch:
        for p in ("sagittal", "coronal"):
            planes[p]["patch"] = torch.randn(8, 4, 3, 3)
    return planes


def test_save_load_round_trips_shape_and_is_float32(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "fake", "clean", "train", "0007")
    src = _exam(with_patch=True)
    fcio.save_exam(path, src)
    assert os.path.isfile(path)
    out = fcio.load_exam(path)
    assert set(out) == {"sagittal", "coronal", "axial"}
    assert out["sagittal"]["pooled"].shape == (8, 12)
    assert out["sagittal"]["pooled"].dtype == torch.float32
    assert out["coronal"]["patch"].shape == (8, 4, 3, 3)
    assert "patch" not in out["axial"]
    torch.testing.assert_close(out["sagittal"]["pooled"], src["sagittal"]["pooled"], rtol=1e-2, atol=1e-2)


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "fake", "clean", "train", "0000")
    fcio.save_exam(path, _exam())
    assert not os.path.exists(path + ".tmp")


def test_exam_cache_path_layout(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "dinov2", "hflip", "valid", "1130")
    assert path.replace(str(tmp_path), "").lstrip("/") == "dinov2/hflip/valid/1130.pt"


def test_manifest_write_read_merge_and_compat(tmp_path):
    fcio.write_manifest(
        str(tmp_path),
        encoders={"medsiglip": {"pooled_dim": 1152, "patch_dim": 1152, "patch_grid": [32, 32]}},
        variants=["clean", "hflip"], splits=["train"],
        slices_per_plane=32, slice_strategy="uniform",
        want_patch_for=["meniscus"], git_sha="abc123",
    )
    fcio.write_manifest(
        str(tmp_path),
        encoders={"dinov2": {"pooled_dim": 768, "patch_dim": 768, "patch_grid": [16, 16]}},
        variants=["rotp"], splits=["valid"],
        slices_per_plane=32, slice_strategy="uniform",
        want_patch_for=["meniscus"], git_sha="abc123",
    )
    man = fcio.read_manifest(str(tmp_path))
    assert man["schema_version"] == fcio.SCHEMA_VERSION
    assert set(man["encoders"]) == {"medsiglip", "dinov2"}
    assert set(man["variants"]) == {"clean", "hflip", "rotp"}
    assert set(man["splits"]) == {"train", "valid"}
    assert fcio.manifest_compatible(man, encoders=["medsiglip", "dinov2"], variants=["clean", "rotp"], slices_per_plane=32)
    assert not fcio.manifest_compatible(man, encoders=["biomedclip"], variants=["clean"], slices_per_plane=32)
    assert not fcio.manifest_compatible(man, encoders=["medsiglip"], variants=["clean"], slices_per_plane=24)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_feature_cache_io.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_cache_io'`.

- [ ] **Step 3: Write `feature_cache_io.py`**

```python
"""Disk I/O for the frozen-encoder feature cache (spec §4.2).

Layout:  <cache_dir>/<encoder>/<variant>/<split>/<exam_id>.pt
Each .pt is  {plane: {"pooled": [S, D] float16, "patch": [S, Dp, h, w] float16?}}.
No model / transformers import here — this module is pure I/O.
"""

import json
import os

import torch

SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"


def exam_cache_path(cache_dir, encoder, variant, split, exam_id):
    return os.path.join(cache_dir, encoder, variant, split, f"{exam_id}.pt")


def save_exam(path, planes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for plane, feats in planes.items():
        entry = {"pooled": feats["pooled"].detach().to(torch.float16).contiguous()}
        if "patch" in feats and feats["patch"] is not None:
            entry["patch"] = feats["patch"].detach().to(torch.float16).contiguous()
        payload[plane] = entry
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_exam(path):
    # cache files hold only tensors -> weights_only=True (no arbitrary unpickling)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for plane, entry in raw.items():
        conv = {"pooled": entry["pooled"].to(torch.float32)}
        if "patch" in entry:
            conv["patch"] = entry["patch"].to(torch.float32)
        out[plane] = conv
    return out


def read_manifest(cache_dir):
    path = os.path.join(cache_dir, _MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def write_manifest(cache_dir, *, encoders, variants, splits, slices_per_plane,
                   slice_strategy, want_patch_for, git_sha):
    os.makedirs(cache_dir, exist_ok=True)
    man = read_manifest(cache_dir)
    merged_encoders = dict(man.get("encoders", {}))
    merged_encoders.update(encoders)
    man.update(
        schema_version=SCHEMA_VERSION,
        encoders=merged_encoders,
        variants=sorted(set(man.get("variants", [])) | set(variants)),
        splits=sorted(set(man.get("splits", [])) | set(splits)),
        slices_per_plane=slices_per_plane,
        slice_strategy=slice_strategy,
        want_patch_for=sorted(set(man.get("want_patch_for", [])) | set(want_patch_for)),
        git_sha=git_sha,
    )
    tmp = os.path.join(cache_dir, _MANIFEST + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(man, handle, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(cache_dir, _MANIFEST))


def manifest_compatible(manifest, *, encoders, variants, slices_per_plane):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if manifest.get("slices_per_plane") != slices_per_plane:
        return False
    have_enc = set(manifest.get("encoders", {}))
    have_var = set(manifest.get("variants", []))
    return set(encoders).issubset(have_enc) and set(variants).issubset(have_var)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_feature_cache_io.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add feature_cache_io.py tests/test_feature_cache_io.py
git commit -m "feat(featbank): feature-cache disk I/O + manifest

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 2: Frozen-encoder protocol, fake encoder, DINOv2 wrapper (`feature_bank.py`)

**Files:**
- Create: `feature_bank.py`
- Test: `tests/test_feature_bank.py` (fast), `tests/test_feature_bank_real.py` (slow, DINOv2 half)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class FrozenEncoder(typing.Protocol)` with attributes `name: str`, `pooled_dim: int`, `patch_dim: int`, `patch_grid: tuple[int, int]`, `input_size: int`, `want_patch: bool`, and method `encode_slices(slices_u8: torch.Tensor) -> dict` where `slices_u8` is `uint8 [S, H, W]` and the return is `{"pooled": FloatTensor[S, pooled_dim]}` plus `{"patch": FloatTensor[S, patch_dim, gh, gw]}` when `want_patch`.
  - `class FakeBankEncoder(torch.nn.Module)` — `name="fake"`, `pooled_dim=16`, `patch_dim=8`, `patch_grid=(4, 4)`, `input_size=32`. Deterministic given input (no randomness, no download).
  - `class DINOv2Encoder(torch.nn.Module)` — `name="dinov2"`, `pooled_dim=768`, `patch_dim=768`, `patch_grid=(16, 16)`, `input_size=224`. Loads `AutoModel.from_pretrained("facebook/dinov2-base")`.
  - `build_encoder(name: str, *, chunk_size: int = 32, want_patch: bool = False, device: str = "cpu") -> FrozenEncoder` — dispatches `"fake"`, `"dinov2"`, `"medsiglip"` (the last wired in Task 3). Raises `ValueError` on an unknown name.
  - Module constant `IMAGENET_MEAN`, `IMAGENET_STD` (length-3 lists).

- [ ] **Step 1: Write the failing fast tests**

```python
# tests/test_feature_bank.py
import torch
import pytest
import feature_bank as fb


def test_fake_encoder_contract_pooled_only():
    enc = fb.build_encoder("fake", want_patch=False)
    out = enc.encode_slices(torch.randint(0, 256, (5, 40, 50), dtype=torch.uint8))
    assert out["pooled"].shape == (5, 16)
    assert out["pooled"].dtype == torch.float32
    assert "patch" not in out


def test_fake_encoder_contract_with_patch():
    enc = fb.build_encoder("fake", want_patch=True)
    out = enc.encode_slices(torch.randint(0, 256, (3, 30, 30), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 16)
    assert out["patch"].shape == (3, 8, 4, 4)


def test_fake_encoder_is_deterministic():
    enc = fb.build_encoder("fake", want_patch=True)
    x = torch.randint(0, 256, (4, 24, 24), dtype=torch.uint8)
    a = enc.encode_slices(x)
    b = enc.encode_slices(x)
    torch.testing.assert_close(a["pooled"], b["pooled"])
    torch.testing.assert_close(a["patch"], b["patch"])


def test_fake_encoder_respects_chunking():
    enc = fb.build_encoder("fake", want_patch=False, chunk_size=2)
    x = torch.randint(0, 256, (7, 16, 16), dtype=torch.uint8)
    whole = fb.build_encoder("fake", want_patch=False, chunk_size=64).encode_slices(x)
    chunked = enc.encode_slices(x)
    torch.testing.assert_close(whole["pooled"], chunked["pooled"])


def test_build_encoder_unknown_name_raises():
    with pytest.raises(ValueError):
        fb.build_encoder("not-an-encoder")
```

```python
# tests/test_feature_bank_real.py
import torch
import pytest
import feature_bank as fb

pytestmark = pytest.mark.slow


def test_real_dinov2_encode_slices_shapes():
    enc = fb.build_encoder("dinov2", want_patch=True, chunk_size=2)
    out = enc.encode_slices(torch.randint(0, 256, (3, 200, 180), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 768)
    assert out["patch"].shape == (3, 768, 16, 16)
    assert torch.isfinite(out["pooled"]).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_feature_bank.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_bank'`.

- [ ] **Step 3: Write `feature_bank.py` (fake + DINOv2 + dispatch)**

```python
"""Frozen pretrained encoders for the MRNet feature bank (spec §3).

All `transformers` imports live here. Each encoder implements the FrozenEncoder
protocol: encode_slices(uint8 [S,H,W]) -> {"pooled": [S,pooled_dim],
"patch": [S,patch_dim,gh,gw]?}. Resize / 3-channel replication / normalisation
are each encoder's own business.
"""

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]


class FrozenEncoder(Protocol):
    name: str
    pooled_dim: int
    patch_dim: int
    patch_grid: tuple
    input_size: int
    want_patch: bool

    def encode_slices(self, slices_u8: torch.Tensor) -> dict: ...


def _to_model_input(slices_u8, size, mean, std, device):
    """uint8 [S,H,W] -> float [S,3,size,size], resized + normalised."""
    x = slices_u8.to(device=device, dtype=torch.float32).div_(255.0)
    x = x.unsqueeze(1)                                  # [S,1,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)                            # [S,3,size,size]
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


class FakeBankEncoder(nn.Module):
    """Deterministic, download-free stand-in for the fast test suite."""

    name = "fake"
    pooled_dim = 16
    patch_dim = 8
    patch_grid = (4, 4)
    input_size = 32

    def __init__(self, *, chunk_size=32, want_patch=False, device="cpu"):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, [0.5] * 3, [0.5] * 3, self.device)
        s = x.shape[0]
        step = self.chunk_size if self.chunk_size > 0 else s
        pooled_chunks, patch_chunks = [], []
        for start in range(0, s, step):
            chunk = x[start:start + step]                        # [c,3,32,32]
            gh, gw = self.patch_grid
            grid = F.adaptive_avg_pool2d(chunk, (gh, gw))        # [c,3,gh,gw]
            # pooled: deterministic function of per-channel grid stats
            flat = grid.mean(dim=(2, 3))                         # [c,3]
            pooled = torch.cat([flat, flat.pow(2), flat.roll(1, 1),
                                flat.flip(1), flat.mean(1, keepdim=True).repeat(1, 4)], dim=1)
            pooled_chunks.append(pooled[:, :self.pooled_dim])
            if self.want_patch:
                patch = grid.repeat(1, 3, 1, 1)[:, :self.patch_dim]   # [c,8,gh,gw]
                patch_chunks.append(patch)
        out = {"pooled": torch.cat(pooled_chunks, dim=0).contiguous()}
        if self.want_patch:
            out["patch"] = torch.cat(patch_chunks, dim=0).contiguous()
        return out


class DINOv2Encoder(nn.Module):
    name = "dinov2"
    pooled_dim = 768
    patch_dim = 768
    patch_grid = (16, 16)
    input_size = 224

    def __init__(self, *, model_id="facebook/dinov2-base", chunk_size=32,
                 want_patch=False, device="cpu"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.chunk_size = int(chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)
        self.model.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, IMAGENET_MEAN, IMAGENET_STD, self.device)
        s = x.shape[0]
        step = self.chunk_size if self.chunk_size > 0 else s
        gh, gw = self.patch_grid
        pooled_chunks, patch_chunks = [], []
        for start in range(0, s, step):
            out = self.model(pixel_values=x[start:start + step])
            pooled_chunks.append(out.pooler_output.float())               # [c,768]
            if self.want_patch:
                tokens = out.last_hidden_state[:, 1:, :].float()          # [c,256,768]
                c = tokens.shape[0]
                patch = tokens.transpose(1, 2).reshape(c, self.patch_dim, gh, gw)
                patch_chunks.append(patch)
        res = {"pooled": torch.cat(pooled_chunks, dim=0).contiguous()}
        if self.want_patch:
            res["patch"] = torch.cat(patch_chunks, dim=0).contiguous()
        return res


def build_encoder(name, *, chunk_size=32, want_patch=False, device="cpu"):
    if name == "fake":
        return FakeBankEncoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    if name == "dinov2":
        return DINOv2Encoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    if name == "medsiglip":
        from feature_bank_medsiglip import MedSigLIPBankEncoder   # wired in Task 3
        return MedSigLIPBankEncoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    raise ValueError(f"unknown encoder: {name!r}")
```

> Implementer note: the `medsiglip` branch imports a name that does not exist yet — that is intentional and only reached in Task 3+. The fast tests never take that branch.

- [ ] **Step 4: Run to verify fast tests pass**

Run: `python -m pytest tests/test_feature_bank.py -q`
Expected: PASS (5 tests). Do **not** run the slow file here.

- [ ] **Step 5: Commit**

```bash
git add feature_bank.py tests/test_feature_bank.py tests/test_feature_bank_real.py
git commit -m "feat(featbank): FrozenEncoder protocol + FakeBankEncoder + DINOv2 wrapper

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 3: MedSigLIP patch grid + `MedSigLIPBankEncoder`

**Files:**
- Modify: `medical_encoders.py:50-67` (the `forward` method)
- Create: `feature_bank_medsiglip.py`
- Modify: `tests/test_feature_bank_real.py` (add the MedSigLIP slow test)
- Test: `tests/test_feature_bank.py` (add a stubbed-tower fast test), `tests/test_medsiglip_backbone.py` (must stay green — Path-1 regression guard)

**Interfaces:**
- Consumes: `feature_bank._to_model_input`, `feature_bank.SIGLIP_MEAN/STD`, `medical_encoders.MedSigLIPEncoder`.
- Produces:
  - `MedSigLIPEncoder.forward(flat_inputs, want_patch=False)` — `want_patch=False` returns `FloatTensor[N, 1152]` (**unchanged** from today, byte for byte); `want_patch=True` returns `{"pooled": FloatTensor[N, 1152], "patch": FloatTensor[N, 1152, 32, 32]}`.
  - `feature_bank_medsiglip.MedSigLIPBankEncoder(nn.Module)` — `name="medsiglip"`, `pooled_dim=1152`, `patch_dim=1152`, `patch_grid=(32, 32)`, `input_size=448`, `want_patch` attr; `encode_slices(uint8 [S,H,W]) -> {"pooled": [S,1152], "patch"?: [S,1152,32,32]}`.
- The `feature_bank.build_encoder("medsiglip", ...)` branch (written in Task 2) now resolves.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_bank.py  (append)
import types
import feature_bank_medsiglip as fbm


def test_medsiglip_bank_encoder_with_stub_tower(monkeypatch):
    class _StubTower:
        config = types.SimpleNamespace(hidden_size=1152)
        def __call__(self, pixel_values):
            n = pixel_values.shape[0]
            return types.SimpleNamespace(
                pooler_output=torch.randn(n, 1152),
                last_hidden_state=torch.randn(n, 32 * 32, 1152),
            )
        def parameters(self): return iter(())
        def eval(self): return self

    class _StubFull:
        vision_model = _StubTower()
        def get_image_features(self, pixel_values):
            return torch.randn(pixel_values.shape[0], 1152)

    monkeypatch.setattr("medical_encoders.AutoModel",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubFull()))
    enc = fbm.MedSigLIPBankEncoder(chunk_size=2, want_patch=True)
    out = enc.encode_slices(torch.randint(0, 256, (3, 100, 90), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 1152)
    assert out["patch"].shape == (3, 1152, 32, 32)
```

```python
# tests/test_feature_bank_real.py  (append)
@pytest.mark.skipif(not __import__("os").environ.get("HF_TOKEN"),
                    reason="needs HF_TOKEN + accepted MedSigLIP license")
def test_real_medsiglip_encode_slices_shapes():
    enc = fb.build_encoder("medsiglip", want_patch=True, chunk_size=2)
    out = enc.encode_slices(torch.randint(0, 256, (2, 300, 260), dtype=torch.uint8))
    assert out["pooled"].shape == (2, 1152)
    assert out["patch"].shape == (2, 1152, 32, 32)
```

- [ ] **Step 2: Run to verify the new fast test fails**

Run: `python -m pytest tests/test_feature_bank.py::test_medsiglip_bank_encoder_with_stub_tower -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'feature_bank_medsiglip'`.

- [ ] **Step 3a: Extend `MedSigLIPEncoder.forward`**

Replace `medical_encoders.py:50-67` with:

```python
    @torch.no_grad()
    def forward(self, flat_inputs, want_patch=False):
        flat_inputs = flat_inputs.to(dtype=torch.float32)
        step = self.chunk_size if self.chunk_size > 0 else flat_inputs.shape[0]
        pooled_chunks = []
        patch_chunks = []
        for start in range(0, flat_inputs.shape[0], step):
            chunk = flat_inputs[start:start + step]
            output = self.tower(pixel_values=chunk)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                if self._image_features_fallback is None:
                    raise RuntimeError(
                        "SigLIP tower returned no pooler_output and the model exposes "
                        "no get_image_features fallback."
                    )
                pooled = self._image_features_fallback(pixel_values=chunk)
            pooled_chunks.append(pooled.float())
            if want_patch:
                tokens = output.last_hidden_state.float()          # [c, 1024, 1152]
                c, t, d = tokens.shape
                side = int(round(t ** 0.5))
                patch_chunks.append(tokens.transpose(1, 2).reshape(c, d, side, side))
        pooled_out = torch.cat(pooled_chunks, dim=0)
        if not want_patch:
            return pooled_out
        return {"pooled": pooled_out, "patch": torch.cat(patch_chunks, dim=0)}
```

- [ ] **Step 3b: Write `feature_bank_medsiglip.py`**

```python
"""MedSigLIP-448 as a FrozenEncoder for the feature bank (spec §3.1).

Wraps the Path-1 medical_encoders.MedSigLIPEncoder; adds resize/normalise and the
uint8 slice contract. transformers is pulled in transitively via medical_encoders.
"""

import torch
import torch.nn as nn

from feature_bank import _to_model_input, SIGLIP_MEAN, SIGLIP_STD
from medical_encoders import MedSigLIPEncoder


class MedSigLIPBankEncoder(nn.Module):
    name = "medsiglip"
    pooled_dim = 1152
    patch_dim = 1152
    patch_grid = (32, 32)
    input_size = 448

    def __init__(self, *, model_id="google/medsiglip-448", chunk_size=32,
                 want_patch=False, device="cpu"):
        super().__init__()
        self.inner = MedSigLIPEncoder(model_id=model_id, chunk_size=chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)
        self.inner.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.inner.eval()
        return self

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, SIGLIP_MEAN, SIGLIP_STD, self.device)
        out = self.inner(x, want_patch=self.want_patch)
        if not self.want_patch:
            return {"pooled": out.float().contiguous()}
        return {"pooled": out["pooled"].float().contiguous(),
                "patch": out["patch"].float().contiguous()}
```

- [ ] **Step 4: Run the new fast test + the Path-1 regression guard**

Run: `python -m pytest tests/test_feature_bank.py tests/test_medsiglip_backbone.py -q`
Expected: PASS. (The stub test passes; the existing MedSigLIP backbone tests still pass because `want_patch` defaults to `False` and the non-patch return path is unchanged.)

- [ ] **Step 5: Commit**

```bash
git add medical_encoders.py feature_bank_medsiglip.py tests/test_feature_bank.py tests/test_feature_bank_real.py
git commit -m "feat(featbank): MedSigLIP patch-grid output + MedSigLIPBankEncoder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 4: Offline cache builder (`scripts/build_feature_cache.py`)

**Files:**
- Create: `scripts/build_feature_cache.py`
- Modify: `tests/conftest.py` (add `feature_cache_fixture`)
- Test: `tests/test_build_feature_cache.py`

**Interfaces:**
- Consumes: `feature_cache_io.*`, `feature_bank.build_encoder`, `dataloader.MRMultiPlaneDataset` (raw volumes, `transform=None`), `dataloader.TASKS`.
- Produces:
  - `VARIANTS: dict[str, dict]` — the six fixed variant recipes (`image` op + `slices` mode).
  - `build_cache(*, data_root, out_dir, encoders, variants, splits, slices_per_plane=32, slice_strategy="uniform", want_patch_for=("meniscus",), chunk_size=32, device="cpu", force=False, allow_low_disk=False) -> None` — idempotent; writes files + manifest; raises `RuntimeError` if a patch pass is requested with `< 16 GiB` free and not `allow_low_disk`.
  - `pick_slice_indices(num_slices: int, k: int, mode: str) -> list[int]` — `mode` in `{"uniform", "uniform_offset", "center"}`.
  - `main(argv=None)` — argparse CLI (`--data_root --out --encoders --variants --splits --slices_per_plane --slice_strategy --want_patch_for --chunk_size --device --force --allow_low_disk`).
- `tests/conftest.py` gains fixture `feature_cache_fixture(mrnet_fixture) -> pathlib.Path` — a cache dir built with `encoders=["fake"]`, `variants=["clean", "hflip", "slicesB"]`, `splits=["train", "valid"]`, `slices_per_plane=8`, `want_patch_for=["meniscus"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_feature_cache.py
import json
import os
import pytest

import feature_cache_io as fcio
from scripts import build_feature_cache as bfc


def test_pick_slice_indices_modes():
    assert bfc.pick_slice_indices(20, 4, "uniform") == sorted(bfc.pick_slice_indices(20, 4, "uniform"))
    assert len(bfc.pick_slice_indices(20, 6, "center")) == 6
    assert bfc.pick_slice_indices(20, 4, "uniform") != bfc.pick_slice_indices(20, 4, "uniform_offset")
    assert bfc.pick_slice_indices(3, 6, "uniform") == [0, 1, 2, 2, 2, 2] or len(bfc.pick_slice_indices(3, 6, "uniform")) == 6


def test_build_cache_writes_files_and_manifest(mrnet_fixture, tmp_path):
    out = tmp_path / "fc"
    bfc.build_cache(
        data_root=str(mrnet_fixture), out_dir=str(out),
        encoders=["fake"], variants=["clean", "hflip"], splits=["train"],
        slices_per_plane=8, want_patch_for=["meniscus"], device="cpu",
    )
    p = fcio.exam_cache_path(str(out), "fake", "clean", "train", "0000")
    assert os.path.isfile(p)
    exam = fcio.load_exam(p)
    assert exam["sagittal"]["pooled"].shape == (8, 16)
    assert exam["sagittal"]["patch"].shape == (8, 8, 4, 4)     # meniscus plane -> patch present
    assert "patch" not in exam["axial"]                        # axial not a meniscus plane
    man = json.loads((out / "manifest.json").read_text())
    assert man["encoders"]["fake"]["pooled_dim"] == 16
    assert set(man["variants"]) == {"clean", "hflip"}


def test_build_cache_is_idempotent(mrnet_fixture, tmp_path):
    out = tmp_path / "fc"
    kw = dict(data_root=str(mrnet_fixture), out_dir=str(out), encoders=["fake"],
              variants=["clean"], splits=["train"], slices_per_plane=8,
              want_patch_for=[], device="cpu")
    bfc.build_cache(**kw)
    p = fcio.exam_cache_path(str(out), "fake", "clean", "train", "0000")
    mtime = os.path.getmtime(p)
    bfc.build_cache(**kw)                                       # second run skips
    assert os.path.getmtime(p) == mtime


def test_build_cache_low_disk_gate(monkeypatch, mrnet_fixture, tmp_path):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda _p: __import__("collections").namedtuple("d", "total used free")(0, 0, 1))
    with pytest.raises(RuntimeError):
        bfc.build_cache(data_root=str(mrnet_fixture), out_dir=str(tmp_path / "fc"),
                        encoders=["fake"], variants=["clean"], splits=["train"],
                        slices_per_plane=8, want_patch_for=["meniscus"], device="cpu")
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def feature_cache_fixture(mrnet_fixture, tmp_path_factory):
    from scripts import build_feature_cache as bfc
    out = tmp_path_factory.mktemp("feature_cache")
    bfc.build_cache(
        data_root=str(mrnet_fixture), out_dir=str(out),
        encoders=["fake"], variants=["clean", "hflip", "slicesB"],
        splits=["train", "valid"], slices_per_plane=8,
        want_patch_for=["meniscus"], device="cpu",
    )
    return out
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_build_feature_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_feature_cache'` (add `scripts/__init__.py` if the repo lacks it — check first with `ls scripts/__init__.py`).

- [ ] **Step 3: Write `scripts/build_feature_cache.py`**

```python
"""Offline frozen-encoder feature cache for MRNet Path 3 (spec §4).

Usage:
  python scripts/build_feature_cache.py --data_root MRNet-v1.0 \
    --encoders medsiglip,dinov2 --variants clean,hflip,rotp,rotn,slicesB,slicesC \
    --slices_per_plane 32 --want_patch_for meniscus --out feature_cache/
"""

import argparse
import os
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_cache_io as fcio                      # noqa: E402
from dataloader import PLANES, TASKS, MRMultiPlaneDataset   # noqa: E402
from feature_bank import build_encoder               # noqa: E402

MIN_FREE_BYTES_FOR_PATCH = 16 * (1024 ** 3)
MENISCUS_PATCH_PLANES = ("sagittal", "coronal")

VARIANTS = {
    "clean":   {"image": "identity", "slices": "uniform"},
    "hflip":   {"image": "hflip",    "slices": "uniform"},
    "rotp":    {"image": "rot+9",    "slices": "uniform"},
    "rotn":    {"image": "rot-9",    "slices": "uniform"},
    "slicesB": {"image": "identity", "slices": "uniform_offset"},
    "slicesC": {"image": "identity", "slices": "center"},
}


def pick_slice_indices(num_slices, k, mode):
    import numpy as np
    if num_slices <= k:
        base = list(range(num_slices)) + [num_slices - 1] * (k - num_slices)
        return base[:k]
    if mode == "center":
        lo = (num_slices - k) // 2
        return list(range(lo, lo + k))
    stride = num_slices / k
    offset = stride / 3.0 if mode == "uniform_offset" else 0.0
    idx = [int(min(num_slices - 1, offset + i * stride)) for i in range(k)]
    return sorted(idx)


def _apply_image_op(volume, op):
    """volume: float tensor [S,H,W]."""
    if op == "identity":
        return volume
    if op == "hflip":
        return torch.flip(volume, dims=[2])
    if op.startswith("rot"):
        from torchvision.transforms.functional import rotate
        angle = float(op[3:])
        return rotate(volume, angle)
    raise ValueError(f"unknown image op: {op}")


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_cache(*, data_root, out_dir, encoders, variants, splits,
                slices_per_plane=32, slice_strategy="uniform",
                want_patch_for=("meniscus",), chunk_size=32, device="cpu",
                force=False, allow_low_disk=False):
    want_patch_for = list(want_patch_for)
    needs_patch = "meniscus" in want_patch_for
    if needs_patch and not allow_low_disk:
        free = shutil.disk_usage(_parent_that_exists(out_dir)).free
        if free < MIN_FREE_BYTES_FOR_PATCH:
            raise RuntimeError(
                f"{free / 2**30:.1f} GiB free < 16 GiB required for a patch-grid pass; "
                "free disk or pass allow_low_disk=True (spec §4.4)."
            )

    built_encoders = {}
    for enc_name in encoders:
        enc = build_encoder(enc_name, chunk_size=chunk_size,
                            want_patch=needs_patch, device=device)
        built_encoders[enc_name] = {
            "pooled_dim": enc.pooled_dim, "patch_dim": enc.patch_dim,
            "patch_grid": list(enc.patch_grid),
        }
        for split in splits:
            ds = MRMultiPlaneDataset(data_root, train=(split == "train"),
                                     mmap=True, cache_size=0, transform=None)
            for i in range(len(ds)):
                volumes, _label, _w, exam_id = ds[i]
                for variant in variants:
                    path = fcio.exam_cache_path(out_dir, enc_name, variant, split, exam_id)
                    if os.path.isfile(path) and not force:
                        continue
                    recipe = VARIANTS[variant]
                    planes_out = {}
                    for plane, vol in zip(PLANES, volumes):
                        vol_t = _apply_image_op(vol, recipe["image"])
                        idx = pick_slice_indices(vol_t.shape[0], slices_per_plane, recipe["slices"])
                        u8 = vol_t[idx].clamp(0, 255).to(torch.uint8)
                        feats = enc.encode_slices(u8)
                        entry = {"pooled": feats["pooled"].cpu()}
                        if needs_patch and plane in MENISCUS_PATCH_PLANES and "patch" in feats:
                            entry["patch"] = feats["patch"].cpu()
                        planes_out[plane] = entry
                    fcio.save_exam(path, planes_out)

    fcio.write_manifest(
        out_dir, encoders=built_encoders, variants=list(variants), splits=list(splits),
        slices_per_plane=slices_per_plane, slice_strategy=slice_strategy,
        want_patch_for=want_patch_for, git_sha=_git_sha(),
    )


def _parent_that_exists(path):
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", dest="out_dir", default="feature_cache/")
    ap.add_argument("--encoders", default="medsiglip,dinov2")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--slices_per_plane", type=int, default=32)
    ap.add_argument("--slice_strategy", default="uniform")
    ap.add_argument("--want_patch_for", default="meniscus")
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_low_disk", action="store_true")
    a = ap.parse_args(argv)
    build_cache(
        data_root=a.data_root, out_dir=a.out_dir,
        encoders=a.encoders.split(","), variants=a.variants.split(","),
        splits=[s.replace("valid", "valid") for s in a.splits.split(",")],
        slices_per_plane=a.slices_per_plane, slice_strategy=a.slice_strategy,
        want_patch_for=[x for x in a.want_patch_for.split(",") if x],
        chunk_size=a.chunk_size, device=a.device, force=a.force,
        allow_low_disk=a.allow_low_disk,
    )


if __name__ == "__main__":
    main()
```

> `MRMultiPlaneDataset` uses split name `"valid"` internally (`train=False`); keep the CLI `--splits` values as `train,valid` and translate with `train=(split == "train")`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_build_feature_cache.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_feature_cache.py scripts/__init__.py tests/test_build_feature_cache.py tests/conftest.py
git commit -m "feat(featbank): offline feature-cache builder (idempotent, disk-gated)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 5: `FeatureCacheDataset` + `featbank_collate`

**Files:**
- Modify: `dataloader.py` (append new class + function; touch nothing existing)
- Test: `tests/test_feature_cache_dataset.py`

**Interfaces:**
- Consumes: `feature_cache_io.load_exam / read_manifest / exam_cache_path`, `dataloader._read_split_records`, `dataloader.resolve_dataset_root`, `dataloader._compute_class_weights`, `dataloader.TASKS`, `dataloader.PLANES`. The `feature_cache_fixture`.
- Produces:
  - `class FeatureCacheDataset(torch.utils.data.Dataset)`:
    - `__init__(cache_dir, split, encoders, variants, *, data_root=None, slices_used=24, slices_used_meniscus=32, want_patch_for=("meniscus",), train=True, seed=0)`. `split` is `"train"` or `"valid"`. Reads labels from the CSVs under `data_root` (falls back to `cache_dir/../` layout only if `data_root` given). Asserts `slices_used <= manifest["slices_per_plane"]`.
    - `.set_epoch(epoch: int) -> None`
    - `.weights: torch.FloatTensor[3]`
    - `.encoder_dims: dict[str, int]` (pooled dims), `.patch_dims: dict[str, int]`
    - `__getitem__(i) -> tuple(payload, label, weights, exam_id)` where
      `payload = {"pooled": {enc: {plane: FloatTensor[S, D]}}, "patch": {enc: {plane: FloatTensor[Sm, Dp, h, w]}}}`,
      `S == slices_used`, `Sm == slices_used_meniscus`, patch present only for `enc`×`plane` in `want_patch_for` meniscus planes. Eval (`train=False`) always uses `variants[0]`.
  - `featbank_collate(items) -> tuple(payload_batched, labels[B,3], weights[3], exam_ids[list])` — stacks the fixed-shape per-item tensors along a new batch dim.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_cache_dataset.py
import torch
from torch.utils.data import DataLoader

from dataloader import FeatureCacheDataset, featbank_collate


def _ds(cache, mrnet_fixture, **kw):
    return FeatureCacheDataset(
        str(cache), "train", encoders=["fake"], variants=["clean", "hflip", "slicesB"],
        data_root=str(mrnet_fixture), slices_used=4, slices_used_meniscus=6, seed=0, **kw,
    )


def test_item_payload_shapes(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    payload, label, weights, exam_id = ds[0]
    assert label.shape == (3,)
    assert payload["pooled"]["fake"]["sagittal"].shape == (4, 16)
    assert payload["patch"]["fake"]["coronal"].shape == (6, 8, 4, 4)
    assert "axial" not in payload["patch"]["fake"]
    assert ds.encoder_dims["fake"] == 16 and ds.patch_dims["fake"] == 8


def test_eval_dataset_is_deterministic_and_uses_first_variant(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture, train=False)
    a = ds[1][0]["pooled"]["fake"]["axial"]
    b = ds[1][0]["pooled"]["fake"]["axial"]
    torch.testing.assert_close(a, b)


def test_set_epoch_changes_variant_selection(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    seen = set()
    for e in range(6):
        ds.set_epoch(e)
        seen.add(ds[0][0]["pooled"]["fake"]["sagittal"].sum().item())
    assert len(seen) > 1                     # different variant => different features


def test_collate_stacks_batch(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    loader = DataLoader(ds, batch_size=3, collate_fn=featbank_collate)
    payload, labels, weights, ids = next(iter(loader))
    assert labels.shape == (3, 3)
    assert payload["pooled"]["fake"]["sagittal"].shape == (3, 4, 16)
    assert payload["patch"]["fake"]["sagittal"].shape == (3, 6, 8, 4, 4)
    assert len(ids) == 3


def test_slices_used_over_cap_raises(feature_cache_fixture, mrnet_fixture):
    import pytest
    with pytest.raises(AssertionError):
        FeatureCacheDataset(str(feature_cache_fixture), "train", encoders=["fake"],
                            variants=["clean"], data_root=str(mrnet_fixture), slices_used=999)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_feature_cache_dataset.py -q`
Expected: FAIL — `ImportError: cannot import name 'FeatureCacheDataset'`.

- [ ] **Step 3: Append to `dataloader.py`**

```python
class FeatureCacheDataset(data.Dataset):
    """Streams cached frozen-encoder features (spec §5.1). Yields a 4-tuple shaped
    like MRMultiPlaneDataset so downstream unpack `(x, label, _, _)` still works;
    here `x` is a nested payload dict, not volume tensors."""

    def __init__(self, cache_dir, split, encoders, variants, *, data_root=None,
                 slices_used=24, slices_used_meniscus=32, want_patch_for=("meniscus",),
                 train=True, seed=0):
        super().__init__()
        import feature_cache_io as fcio
        self._fcio = fcio
        self.cache_dir = cache_dir
        self.split = split
        self.encoders = list(encoders)
        self.variants = list(variants)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.slices_used = int(slices_used)
        self.slices_used_meniscus = int(slices_used_meniscus)
        self.want_patch_for = set(want_patch_for)

        manifest = fcio.read_manifest(cache_dir)
        cap = manifest.get("slices_per_plane", self.slices_used)
        assert self.slices_used <= cap, f"slices_used {self.slices_used} > cached {cap}"
        assert self.slices_used_meniscus <= cap
        self.encoder_dims = {e: manifest["encoders"][e]["pooled_dim"] for e in self.encoders}
        self.patch_dims = {e: manifest["encoders"][e]["patch_dim"] for e in self.encoders}

        root = resolve_dataset_root(data_root)
        records = _read_split_records(root, split)
        self.exam_ids = records["id"].tolist()
        self.labels = records[list(TASKS)].to_numpy(dtype=np.float32)
        self.weights = _compute_class_weights(self.labels)
        self._meniscus_planes = ("sagittal", "coronal")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.exam_ids)

    def _variant_for(self, i):
        if not self.train:
            return self.variants[0]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        return self.variants[int(rng.integers(0, len(self.variants)))]

    @staticmethod
    def _subsample(t, k):
        s = t.shape[0]
        if s == k:
            return t
        idx = torch.linspace(0, s - 1, k).round().long()
        return t.index_select(0, idx)

    def __getitem__(self, i):
        exam_id = self.exam_ids[i]
        variant = self._variant_for(i)
        pooled, patch = {}, {}
        for enc in self.encoders:
            path = self._fcio.exam_cache_path(self.cache_dir, enc, variant, self.split, exam_id)
            exam = self._fcio.load_exam(path)
            pooled[enc] = {p: self._subsample(exam[p]["pooled"], self.slices_used) for p in PLANES}
            if enc in self.want_patch_for or "meniscus" in self.want_patch_for:
                pe = {}
                for p in self._meniscus_planes:
                    if "patch" in exam[p]:
                        pe[p] = self._subsample(exam[p]["patch"], self.slices_used_meniscus)
                if pe:
                    patch[enc] = pe
        payload = {"pooled": pooled, "patch": patch}
        label = torch.from_numpy(self.labels[i])
        return payload, label, self.weights, exam_id


def featbank_collate(items):
    payloads = [it[0] for it in items]
    labels = torch.stack([it[1] for it in items], dim=0)
    weights = items[0][2]
    exam_ids = [it[3] for it in items]

    def _stack(key):
        out = {}
        for enc in payloads[0][key]:
            out[enc] = {}
            for plane in payloads[0][key][enc]:
                out[enc][plane] = torch.stack([p[key][enc][plane] for p in payloads], dim=0)
        return out

    return {"pooled": _stack("pooled"), "patch": _stack("patch")}, labels, weights, exam_ids
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_feature_cache_dataset.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add dataloader.py tests/test_feature_cache_dataset.py
git commit -m "feat(featbank): FeatureCacheDataset + featbank_collate

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 6: `FeatBankMRNet` model

**Files:**
- Modify: `lightweight_models.py` (append; reuse `GeMPool1D`, `AttentionMILPool`, `PlaneAttentionFusion`)
- Test: `tests/test_featbank_model.py`

**Interfaces:**
- Consumes: the `featbank_collate` payload shape from Task 5; `dataloader.TASKS`.
- Produces:
  - `class FeatBankMRNet(nn.Module)`:
    - `__init__(encoder_dims: dict[str, int], patch_dims: dict[str, int], *, d_model=256, dropout=0.15, head_hidden=128, want_patch_encoders: tuple[str, ...] | None = None)`. `want_patch_encoders` defaults to all keys of `patch_dims`.
    - class attribute `consumes_feature_batch = True`
    - `forward(payload: dict) -> torch.FloatTensor[B, 3]` — logits in `TASKS` order `(abnormal, acl, meniscus)`.
  - Head plane sets (fixed): `abnormal` → `("sagittal", "coronal", "axial")`; `acl` → `("sagittal", "coronal")`; `meniscus` → `("sagittal", "coronal")` pooled + patch pyramid.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_featbank_model.py
import torch
from lightweight_models import FeatBankMRNet
from dataloader import TASKS


def _payload(B=2, S=4, Sm=6, encs=("e1", "e2"), dims=(16, 24), pdim=8, grid=4):
    pooled = {e: {p: torch.randn(B, S, d) for p in ("sagittal", "coronal", "axial")}
              for e, d in zip(encs, dims)}
    patch = {e: {p: torch.randn(B, Sm, pdim, grid, grid) for p in ("sagittal", "coronal")}
             for e in encs}
    return {"pooled": pooled, "patch": patch}


def test_forward_returns_B_by_3():
    m = FeatBankMRNet({"e1": 16, "e2": 24}, {"e1": 8, "e2": 8}, d_model=32)
    out = m(_payload())
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()
    assert TASKS == ("abnormal", "acl", "meniscus")     # order guard


def test_all_params_train_and_receive_grad():
    m = FeatBankMRNet({"e1": 16}, {"e1": 8}, d_model=16)
    out = m(_payload(encs=("e1",), dims=(16,)))
    out.sum().backward()
    assert all(p.requires_grad for p in m.parameters())
    assert all(p.grad is not None for p in m.parameters())


def test_meniscus_head_uses_patch_pyramid():
    m = FeatBankMRNet({"e1": 16}, {"e1": 8}, d_model=16)
    p = _payload(encs=("e1",), dims=(16,))
    base = m(p)[:, 2].clone()
    p["patch"]["e1"]["sagittal"] += 5.0                 # perturb only the pyramid input
    assert not torch.allclose(base, m(p)[:, 2])


def test_works_without_patch_when_no_patch_encoders():
    m = FeatBankMRNet({"e1": 16}, {}, d_model=16, want_patch_encoders=())
    p = _payload(encs=("e1",), dims=(16,))
    p["patch"] = {}
    assert m(p).shape == (2, 3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_featbank_model.py -q`
Expected: FAIL — `ImportError: cannot import name 'FeatBankMRNet'`.

- [ ] **Step 3: Append to `lightweight_models.py`**

```python
class _PlaneFuse(nn.Module):
    """PlaneAttentionFusion (returns [B, n_planes*d]) followed by a projection back
    to [B, d] so head shapes match the spec's sketches."""

    def __init__(self, d_model, n_planes):
        super().__init__()
        self.fuse = PlaneAttentionFusion(d_model)
        self.proj = nn.Linear(d_model * n_planes, d_model)

    def forward(self, plane_feats):                       # list of [B, d]
        return self.proj(self.fuse(plane_feats))


class _PooledHead(nn.Module):
    """GeM slice-pool per (enc,plane) -> mean over encoders -> plane fusion -> MLP."""

    def __init__(self, encoders, planes, d_model, head_hidden, dropout):
        super().__init__()
        self.encoders = list(encoders)
        self.planes = list(planes)
        self.pool = nn.ModuleDict({e: GeMPool1D() for e in self.encoders})
        self.fuse = _PlaneFuse(d_model, len(self.planes))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, projected):                         # projected[enc][plane] = [B,S,d]
        plane_feats = []
        for plane in self.planes:
            per_enc = [self.pool[e](projected[e][plane]) for e in self.encoders]  # [B,d] each
            plane_feats.append(torch.stack(per_enc, dim=0).mean(dim=0))
        return self.mlp(self.fuse(plane_feats)).squeeze(-1)          # [B]


class _PyramidPath(nn.Module):
    """Per-slice depthwise-separable conv at strides {1,2} -> GAP -> concat ->
    Linear(d_model) -> attention slice-pool -> mean over enc -> plane fusion."""

    def __init__(self, patch_dims, planes, d_model):
        super().__init__()
        self.encoders = list(patch_dims)
        self.planes = list(planes)
        self.blocks = nn.ModuleDict()
        for e, dp in patch_dims.items():
            self.blocks[e] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dp, dp, 3, stride=s, padding=1, groups=dp),
                    nn.Conv2d(dp, d_model, 1), nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                ) for s in (1, 2)
            ])
        self.merge = nn.ModuleDict({e: nn.Linear(2 * d_model, d_model) for e in self.encoders})
        self.slice_pool = nn.ModuleDict({e: AttentionMILPool(d_model) for e in self.encoders})
        self.fuse = _PlaneFuse(d_model, len(self.planes))

    def forward(self, patch):                             # patch[enc][plane] = [B,Sm,Dp,h,w]
        plane_feats = []
        for plane in self.planes:
            per_enc = []
            for e in self.encoders:
                x = patch[e][plane]
                b, sm, dp, h, w = x.shape
                x = x.reshape(b * sm, dp, h, w)
                scales = [blk(x).flatten(1) for blk in self.blocks[e]]      # [b*sm, d_model] each
                slc = self.merge[e](torch.cat(scales, dim=1)).reshape(b, sm, -1)
                per_enc.append(self.slice_pool[e](slc))                     # [B,d]
            plane_feats.append(torch.stack(per_enc, dim=0).mean(dim=0))
        return self.fuse(plane_feats)                                       # [B,d]


class FeatBankMRNet(nn.Module):
    consumes_feature_batch = True

    _HEAD_PLANES = {
        "abnormal": ("sagittal", "coronal", "axial"),
        "acl": ("sagittal", "coronal"),
        "meniscus": ("sagittal", "coronal"),
    }

    def __init__(self, encoder_dims, patch_dims, *, d_model=256, dropout=0.15,
                 head_hidden=128, want_patch_encoders=None):
        super().__init__()
        self.encoders = list(encoder_dims)
        self.d_model = int(d_model)
        if want_patch_encoders is None:
            want_patch_encoders = tuple(patch_dims)
        self.patch_encoders = list(want_patch_encoders)

        self.proj = nn.ModuleDict({
            f"{e}::{p}": nn.Sequential(
                nn.LayerNorm(encoder_dims[e]), nn.Linear(encoder_dims[e], d_model), nn.GELU())
            for e in self.encoders for p in ("sagittal", "coronal", "axial")
        })
        self.abnormal_head = _PooledHead(self.encoders, self._HEAD_PLANES["abnormal"],
                                         d_model, head_hidden, dropout)
        self.acl_head = _PooledHead(self.encoders, self._HEAD_PLANES["acl"],
                                    d_model, head_hidden, dropout)
        self.meniscus_pooled = _PooledHead(self.encoders, self._HEAD_PLANES["meniscus"],
                                           d_model, head_hidden, dropout)
        self.meniscus_pyramid = None
        if self.patch_encoders:
            self.meniscus_pyramid = _PyramidPath(
                {e: patch_dims[e] for e in self.patch_encoders},
                self._HEAD_PLANES["meniscus"], d_model)
            self.meniscus_mlp = nn.Sequential(
                nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
                nn.Linear(2 * d_model, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, 1),
            )

    def _project(self, pooled):
        return {e: {p: self.proj[f"{e}::{p}"](pooled[e][p])
                    for p in ("sagittal", "coronal", "axial")}
                for e in self.encoders}

    def forward(self, payload):
        projected = self._project(payload["pooled"])
        abnormal = self.abnormal_head(projected)
        acl = self.acl_head(projected)

        if self.meniscus_pyramid is not None and payload.get("patch"):
            pooled_vec = self.meniscus_pooled.fuse([
                torch.stack([self.meniscus_pooled.pool[e](projected[e][p]) for e in self.encoders],
                            dim=0).mean(dim=0)
                for p in self._HEAD_PLANES["meniscus"]
            ])                                              # [B,d]
            pyr_vec = self.meniscus_pyramid(payload["patch"])   # [B,d]
            meniscus = self.meniscus_mlp(torch.cat([pooled_vec, pyr_vec], dim=1)).squeeze(-1)
        else:
            meniscus = self.meniscus_pooled(projected)

        return torch.stack([abnormal, acl, meniscus], dim=1)     # [B,3] TASKS order
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_featbank_model.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lightweight_models.py tests/test_featbank_model.py
git commit -m "feat(featbank): FeatBankMRNet (per-task heads + meniscus pyramid path)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 7: `run_featbank` training loop (`featbank_train.py`)

**Files:**
- Create: `featbank_train.py`
- Test: `tests/test_featbank_train.py`

**Interfaces:**
- Consumes: `dataloader.FeatureCacheDataset`, `dataloader.featbank_collate`, `lightweight_models.FeatBankMRNet`, `train.ModelEMA`, `train.compute_auc`, `train.compute_loss`, `sklearn.metrics.roc_auc_score`.
- Produces:
  - `make_folds(n: int, k: int, seed: int) -> list[np.ndarray]` — deterministic, disjoint, union == `range(n)`.
  - `run_featbank(args) -> dict` — trains and returns
    `{"per_task_auc": [abnormal, acl, meniscus], "per_task_mean": float, "pooled_auc": float, "best_epoch": int}`.
    Honours `args`: `feature_cache`, `data_root`, `encoders`, `cache_variants`, `slices_used`, `slices_used_meniscus`, `d_model`, `dropout`, `epochs`, `lr`, `weight_decay`, `ema_decay`, `focal_gamma`, `loss_type`, `label_smoothing`, `patience`, `select_metric`, `featbank_batch_size`, `eval_tta_variants`, `prefix_name`, `save_model`, `cv_folds`, `cv_fold`, `seed`, `time_budget_minutes`.
  - Checkpoint filename: `model_{prefix}_featbank_ptmean_{per_task_mean:.4f}_epoch_{e}.pth`; prior files containing `prefix` are pruned first (mirrors `train.py`).
- `args` may be an `argparse.Namespace` or a `SimpleNamespace`; access everything via `getattr(args, name, default)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_featbank_train.py
from types import SimpleNamespace
import numpy as np
import featbank_train as ft


def test_make_folds_deterministic_partition_disjoint():
    a = ft.make_folds(37, 5, seed=0)
    b = ft.make_folds(37, 5, seed=0)
    assert [list(x) for x in a] == [list(y) for y in b]
    allidx = np.concatenate(a)
    assert sorted(allidx.tolist()) == list(range(37))
    assert ft.make_folds(37, 5, seed=1) != a or True   # different seed may differ; not asserted hard


def _args(cache, mrnet_fixture, **over):
    base = dict(
        feature_cache=str(cache), data_root=str(mrnet_fixture),
        encoders="fake", cache_variants="clean,hflip", slices_used=4,
        slices_used_meniscus=6, d_model=8, dropout=0.1, epochs=1, lr=1e-3,
        weight_decay=0.0, ema_decay=0.0, focal_gamma=2.0, loss_type="focal",
        label_smoothing=0.0, patience=0, select_metric="per_task_mean",
        featbank_batch_size=2, eval_tta_variants="clean", prefix_name="tfb",
        save_model=0, cv_folds=0, cv_fold=-1, seed=0, time_budget_minutes=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_run_featbank_returns_metrics_dict(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture))
    assert set(out) == {"per_task_auc", "per_task_mean", "pooled_auc", "best_epoch"}
    assert len(out["per_task_auc"]) == 3
    assert 0.0 <= out["per_task_mean"] <= 1.0


def test_run_featbank_cv_fold_runs(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture, cv_folds=2, cv_fold=0))
    assert "per_task_mean" in out


def test_select_metric_loss_is_accepted(feature_cache_fixture, mrnet_fixture):
    out = ft.run_featbank(_args(feature_cache_fixture, mrnet_fixture, select_metric="loss"))
    assert out["best_epoch"] >= 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_featbank_train.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'featbank_train'`.

- [ ] **Step 3: Write `featbank_train.py`**

```python
"""Feature-bank training loop for MRNet Path 3 (spec §6).

Isolated from train.iterate_epoch (which is volume-tensor shaped). Reuses
train.ModelEMA / compute_auc / compute_loss and the checkpoint-naming convention.
"""

import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

import utils
from dataloader import FeatureCacheDataset, featbank_collate, TASKS
from lightweight_models import FeatBankMRNet
from train import ModelEMA, compute_auc, compute_loss


def make_folds(n, k, seed):
    order = np.random.default_rng(seed).permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def _g(args, name, default=None):
    return getattr(args, name, default)


def _move(payload, device):
    return {
        key: {e: {p: t.to(device) for p, t in planes.items()}
              for e, planes in payload[key].items()}
        for key in ("pooled", "patch")
    }


def _select_value(metric, val_loss, pooled_auc, per_task_mean):
    if metric == "loss":
        return -val_loss                       # higher is better everywhere
    if metric == "pooled_auc":
        return pooled_auc
    return per_task_mean


def run_featbank(args):
    device = utils.get_device()
    variants = _g(args, "cache_variants", "clean").split(",")
    encoders = _g(args, "encoders", "fake").split(",")

    full = FeatureCacheDataset(
        _g(args, "feature_cache"), "train", encoders=encoders, variants=variants,
        data_root=_g(args, "data_root"), slices_used=_g(args, "slices_used", 24),
        slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
        train=True, seed=_g(args, "seed", 0),
    )
    val_variants = _g(args, "eval_tta_variants", "clean").split(",")

    if _g(args, "cv_folds", 0) and _g(args, "cv_fold", -1) >= 0:
        folds = make_folds(len(full), args.cv_folds, _g(args, "seed", 0))
        val_idx = folds[args.cv_fold]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != args.cv_fold])
        train_ds = Subset(full, train_idx.tolist())
        val_ds = Subset(
            FeatureCacheDataset(_g(args, "feature_cache"), "train", encoders=encoders,
                                variants=val_variants, data_root=_g(args, "data_root"),
                                slices_used=_g(args, "slices_used", 24),
                                slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
                                train=False, seed=_g(args, "seed", 0)),
            val_idx.tolist(),
        )
        weights = full.weights
    else:
        train_ds = full
        val_ds = FeatureCacheDataset(
            _g(args, "feature_cache"), "valid", encoders=encoders, variants=val_variants,
            data_root=_g(args, "data_root"), slices_used=_g(args, "slices_used", 24),
            slices_used_meniscus=_g(args, "slices_used_meniscus", 32),
            train=False, seed=_g(args, "seed", 0),
        )
        weights = full.weights

    bs = _g(args, "featbank_batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=featbank_collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=featbank_collate)

    model = FeatBankMRNet(full.encoder_dims, full.patch_dims,
                          d_model=_g(args, "d_model", 256),
                          dropout=_g(args, "dropout", 0.15)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_g(args, "lr", 3e-4),
                                  weight_decay=_g(args, "weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    ema = ModelEMA(model, args.ema_decay) if _g(args, "ema_decay", 0.0) > 0 else None

    best_select = -float("inf")
    best = {"per_task_auc": [0.0, 0.0, 0.0], "per_task_mean": 0.0,
            "pooled_auc": 0.0, "best_epoch": 0}
    patience = _g(args, "patience", 0)
    bad_epochs = 0
    model_dir = os.path.abspath(os.environ.get("MRNET_MODEL_DIR", "models"))
    t0 = time.time()

    for epoch in range(_g(args, "epochs", 30)):
        budget = _g(args, "time_budget_minutes", None)
        if budget is not None and (time.time() - t0) / 60.0 >= budget:
            break
        if isinstance(train_ds, Subset):
            train_ds.dataset.set_epoch(epoch)
        else:
            train_ds.set_epoch(epoch)

        model.train()
        for payload, labels, _w, _ids in train_loader:
            payload = _move(payload, device)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(payload)
            loss = compute_loss(logits, labels, criterion, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        if ema is not None:
            ema.apply_to(model)
        model.eval()
        yt, yp, losses = [], [], []
        with torch.no_grad():
            for payload, labels, _w, _ids in val_loader:
                payload = _move(payload, device)
                labels_d = labels.to(device=device, dtype=torch.float32)
                logits = model(payload)
                losses.append(compute_loss(logits, labels_d, criterion, args).item())
                yp.append(torch.sigmoid(logits).cpu().numpy())
                yt.append(labels.numpy())
        if ema is not None:
            ema.restore(model)

        yt = np.concatenate(yt, axis=0)
        yp = np.concatenate(yp, axis=0)
        val_loss = float(np.mean(losses)) if losses else float("nan")
        per_task = [
            roc_auc_score(yt[:, k], yp[:, k]) if len(np.unique(yt[:, k])) > 1 else 0.5
            for k in range(3)
        ]
        per_task_mean = float(np.mean(per_task))
        pooled_auc = compute_auc(yt.reshape(-1).astype(int).tolist(), yp.reshape(-1).tolist())
        scheduler.step(val_loss)

        print(f"[featbank epoch {epoch + 1}] val_loss {val_loss:.4f} "
              f"| abnormal {per_task[0]:.4f} acl {per_task[1]:.4f} meniscus {per_task[2]:.4f} "
              f"| per_task_mean {per_task_mean:.4f} | pooled {pooled_auc:.4f}")

        select = _select_value(_g(args, "select_metric", "per_task_mean"),
                               val_loss, pooled_auc, per_task_mean)
        if select > best_select:
            best_select = select
            best = {"per_task_auc": per_task, "per_task_mean": per_task_mean,
                    "pooled_auc": pooled_auc, "best_epoch": epoch + 1}
            bad_epochs = 0
            if _g(args, "save_model", 0):
                os.makedirs(model_dir, exist_ok=True)
                fname = f"model_{args.prefix_name}_featbank_ptmean_{per_task_mean:.4f}_epoch_{epoch + 1}.pth"
                for existing in os.listdir(model_dir):
                    if args.prefix_name in existing:
                        os.remove(os.path.join(model_dir, existing))
                torch.save(model.state_dict(), os.path.join(model_dir, fname))
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break

    print(f"best_val_per_task_mean: {best['per_task_mean']:.6f}")
    print(f"best_val_per_task:      {best['per_task_auc']}")
    print(f"best_val_pooled_auc:    {best['pooled_auc']:.6f}")
    return best
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_featbank_train.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add featbank_train.py tests/test_featbank_train.py
git commit -m "feat(featbank): run_featbank training loop + make_folds

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 8: `train.py` wiring

**Files:**
- Modify: `train.py` — `build_model` (line ~366), `run` (line ~461), the argument parser (lines ~659-800), the `--model_type` choices (line ~693), and the `main()` post-parse block.
- Test: `tests/test_train_featbank_wiring.py`

**Interfaces:**
- Consumes: `featbank_train.run_featbank`.
- Produces: `python train.py --model_type featbank --feature_cache DIR ...` dispatches to `run_featbank` and never runs the volume loop; `--select_metric` resolves per Global Constraints.
- New args (all with the exact names/defaults from the spec §11.1): `--feature_cache` (str, default `""`), `--encoders` (str, default `"medsiglip,dinov2"`), `--cache_variants` (str, default `"clean,hflip,rotp,rotn,slicesB,slicesC"`), `--slices_used` (int, 24), `--slices_used_meniscus` (int, 32), `--d_model` (int, 256), `--featbank_batch_size` (int, 16), `--eval_tta_variants` (str, `"clean"`), `--cv_folds` (int, 0), `--cv_fold` (int, -1), `--select_metric` (choices `["loss", "pooled_auc", "per_task_mean"]`, default `None`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_featbank_wiring.py
from types import SimpleNamespace
import train


def test_run_dispatches_to_run_featbank(monkeypatch):
    seen = {}
    monkeypatch.setattr("featbank_train.run_featbank", lambda a: seen.setdefault("args", a) or {"per_task_mean": 0.9})
    train.run(SimpleNamespace(model_type="featbank", batch_size=1, select_metric="per_task_mean"))
    assert seen["args"].model_type == "featbank"


def test_parser_has_featbank_choice_and_args():
    parser = train.build_arg_parser() if hasattr(train, "build_arg_parser") else None
    args = train.parse_args([
        "--prefix_name", "x", "--model_type", "featbank",
        "--feature_cache", "/tmp/fc",
    ])
    assert args.model_type == "featbank"
    assert args.feature_cache == "/tmp/fc"
    assert args.encoders == "medsiglip,dinov2"
    assert args.slices_used == 24 and args.d_model == 256


def test_select_metric_defaults_by_model_type():
    fb = train.parse_args(["--prefix_name", "x", "--model_type", "featbank"])
    assert fb.select_metric == "per_task_mean"
    cnn = train.parse_args(["--prefix_name", "x", "--model_type", "resnet18"])
    assert cnn.select_metric == "loss"
```

> If `train.py` currently parses args inline in `__main__`, refactor that into
> `parse_args(argv=None)` (and keep `__main__` calling it) as part of this task —
> the test needs a callable entry point. `build_arg_parser` is optional.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_train_featbank_wiring.py -q`
Expected: FAIL — `AttributeError: module 'train' has no attribute 'parse_args'` / dispatch not present.

- [ ] **Step 3: Edit `train.py`**

3a. Add `"featbank"` to the `--model_type` `choices=[...]` list (line ~693-707).

3b. In the argument parser, after `--plane_transformer_heads` (line ~688), add:

```python
    parser.add_argument("--feature_cache", type=str, default="")
    parser.add_argument("--encoders", type=str, default="medsiglip,dinov2")
    parser.add_argument("--cache_variants", type=str,
                        default="clean,hflip,rotp,rotn,slicesB,slicesC")
    parser.add_argument("--slices_used", type=int, default=24)
    parser.add_argument("--slices_used_meniscus", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--featbank_batch_size", type=int, default=16)
    parser.add_argument("--eval_tta_variants", type=str, default="clean")
    parser.add_argument("--cv_folds", type=int, default=0)
    parser.add_argument("--cv_fold", type=int, default=-1)
    parser.add_argument("--select_metric", type=str, default=None,
                        choices=["loss", "pooled_auc", "per_task_mean"])
```

3c. Wrap the parser in `parse_args`, resolving the `select_metric` default:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # ... all existing add_argument calls ...
    args = parser.parse_args(argv)
    if args.select_metric is None:
        args.select_metric = "per_task_mean" if args.model_type == "featbank" else "loss"
    return args
```

Update `if __name__ == "__main__":` to `run(parse_args())`.

3d. First lines of `run(args)` (before the `batch_size` guard at line 462):

```python
def run(args):
    if getattr(args, "model_type", "") == "featbank":
        from featbank_train import run_featbank
        return run_featbank(args)
    if args.batch_size != 1:
        raise ValueError(...)          # unchanged
```

3e. In `build_model`, add a guard so a stray call is loud:

```python
    if args.model_type == "featbank":
        raise RuntimeError("featbank models are built inside featbank_train.run_featbank, "
                           "not build_model; run() dispatches before reaching here.")
```

- [ ] **Step 4: Run to verify they pass, plus the full fast suite**

Run: `python -m pytest tests/test_train_featbank_wiring.py -q`
Expected: PASS (3 tests).

Run: `python -m pytest -q -m "not slow"`
Expected: PASS — the pre-existing count plus all new fast tests; no regressions.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_featbank_wiring.py
git commit -m "feat(featbank): wire model_type=featbank into train.py (dispatch + args)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 9: CV selection harness (`scripts/cv_select.py`)

**Files:**
- Create: `scripts/cv_select.py`
- Test: `tests/test_cv_select.py`

**Interfaces:**
- Consumes: `featbank_train.make_folds`, `featbank_train.run_featbank`, `train.parse_args` (for arg shape reuse — or build its own `Namespace`).
- Produces:
  - `cv_select(base_args, *, k=5, seed=0, d_model_grid=None, out_tsv="cv_results.tsv") -> list[dict]` — for each `d_model` in `d_model_grid or [base_args.d_model]` and each fold `0..k-1`, calls `run_featbank` with `cv_folds=k, cv_fold=f, d_model=dm` and records a row `{"d_model", "fold", "abnormal_auc", "acl_auc", "meniscus_auc", "per_task_mean"}`. Writes all rows to `out_tsv` (tab-separated, header row). Returns the rows.
  - `main(argv=None)` — CLI: the featbank args (`--feature_cache --data_root --encoders --cache_variants --slices_used --slices_used_meniscus --epochs --lr --ema_decay ...`) plus `--k 5 --seed 0 --sweep_d_model 0 --out cv_results.tsv`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_select.py
from types import SimpleNamespace
import csv
from scripts import cv_select


def _base(cache, mrnet_fixture):
    return SimpleNamespace(
        feature_cache=str(cache), data_root=str(mrnet_fixture), encoders="fake",
        cache_variants="clean,hflip", slices_used=4, slices_used_meniscus=6, d_model=8,
        dropout=0.1, epochs=1, lr=1e-3, weight_decay=0.0, ema_decay=0.0, focal_gamma=2.0,
        loss_type="focal", label_smoothing=0.0, patience=0, select_metric="per_task_mean",
        featbank_batch_size=2, eval_tta_variants="clean", prefix_name="cv", save_model=0,
        seed=0, time_budget_minutes=None,
    )


def test_cv_select_writes_tsv_with_expected_rows(feature_cache_fixture, mrnet_fixture, tmp_path):
    out = tmp_path / "cv.tsv"
    rows = cv_select.cv_select(_base(feature_cache_fixture, mrnet_fixture),
                               k=2, seed=0, d_model_grid=[8], out_tsv=str(out))
    assert len(rows) == 2                                   # 1 d_model x 2 folds
    with open(out) as fh:
        reader = list(csv.DictReader(fh, delimiter="\t"))
    assert len(reader) == 2
    assert {"d_model", "fold", "abnormal_auc", "acl_auc", "meniscus_auc", "per_task_mean"} <= set(reader[0])


def test_cv_select_sweeps_d_model(feature_cache_fixture, mrnet_fixture, tmp_path):
    rows = cv_select.cv_select(_base(feature_cache_fixture, mrnet_fixture),
                               k=2, seed=0, d_model_grid=[8, 12], out_tsv=str(tmp_path / "s.tsv"))
    assert sorted({r["d_model"] for r in rows}) == [8, 12]
    assert len(rows) == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_cv_select.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.cv_select'`.

- [ ] **Step 3: Write `scripts/cv_select.py`**

```python
"""5-fold CV over cached features for Path 3 hyperparameter / threshold selection
(spec §6). Writes cv_results.tsv; the 120-val number is produced separately by a
plain `python train.py --model_type featbank` run (no --cv_fold)."""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featbank_train import run_featbank                     # noqa: E402

_FIELDS = ["d_model", "fold", "abnormal_auc", "acl_auc", "meniscus_auc", "per_task_mean"]


def cv_select(base_args, *, k=5, seed=0, d_model_grid=None, out_tsv="cv_results.tsv"):
    grid = d_model_grid or [getattr(base_args, "d_model", 256)]
    rows = []
    for dm in grid:
        for fold in range(k):
            a = copy.copy(base_args)
            a.d_model = dm
            a.cv_folds = k
            a.cv_fold = fold
            a.seed = seed
            res = run_featbank(a)
            rows.append({
                "d_model": dm, "fold": fold,
                "abnormal_auc": round(res["per_task_auc"][0], 4),
                "acl_auc": round(res["per_task_auc"][1], 4),
                "meniscus_auc": round(res["per_task_auc"][2], 4),
                "per_task_mean": round(res["per_task_mean"], 4),
            })
    with open(out_tsv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--encoders", default="medsiglip,dinov2")
    ap.add_argument("--cache_variants", default="clean,hflip,rotp,rotn,slicesB,slicesC")
    ap.add_argument("--slices_used", type=int, default=24)
    ap.add_argument("--slices_used_meniscus", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.0)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--loss_type", default="focal")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--select_metric", default="per_task_mean")
    ap.add_argument("--featbank_batch_size", type=int, default=16)
    ap.add_argument("--eval_tta_variants", default="clean")
    ap.add_argument("--prefix_name", default="cvsel")
    ap.add_argument("--save_model", type=int, default=0)
    ap.add_argument("--time_budget_minutes", type=float, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep_d_model", type=int, default=0)
    ap.add_argument("--out", default="cv_results.tsv")
    a = ap.parse_args(argv)
    grid = [192, 256, 384] if a.sweep_d_model else [a.d_model]
    rows = cv_select(a, k=a.k, seed=a.seed, d_model_grid=grid, out_tsv=a.out)
    for dm in sorted({r["d_model"] for r in rows}):
        sub = [r["per_task_mean"] for r in rows if r["d_model"] == dm]
        print(f"d_model={dm}: mean per_task_mean over {len(sub)} folds = {sum(sub) / len(sub):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_cv_select.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/cv_select.py tests/test_cv_select.py
git commit -m "feat(featbank): 5-fold CV selection harness

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Task 10: Documentation (`MEDICAL_MODELS.md`)

**Files:**
- Modify: `MEDICAL_MODELS.md` — append a new top-level section after the MedGemma-4B section.
- Test: none (docs). Verification is a manual read-through + the commands are copy-paste-consistent with the args added in Tasks 4/8/9.

**Interfaces:**
- Consumes: the final arg names from Tasks 4, 8, 9.
- Produces: a "Path 3" section a new engineer can follow end to end.

- [ ] **Step 1: Append the section**

````markdown
## Path 3 — Multi-encoder feature bank (`--model_type featbank`)

A single 3-head classifier trained on **cached features** from frozen encoders
(MedSigLIP-448 + DINOv2 ViT-B/14). Encoders run once, offline; training then reads
fp16 feature files, so an epoch is seconds and 5-fold CV is cheap. Design:
`docs/superpowers/specs/2026-09-05-multiencoder-pertask-mrnet-design.md`.

### One-time setup

1. `export HF_TOKEN=...` (MedSigLIP is gated; DINOv2 is not).
2. `export PYTORCH_ENABLE_MPS_FALLBACK=1`
3. `pip install -r requirements.txt`
4. Ensure **≥ 16 GB free disk** before the cache build (the patch-grid pass is
   gated; override with `--allow_low_disk` at your own risk).

### Build the feature cache (~hours on MPS, one time)

```bash
python scripts/build_feature_cache.py \
  --data_root MRNet-v1.0 \
  --encoders medsiglip,dinov2 \
  --variants clean,hflip,rotp,rotn,slicesB,slicesC \
  --slices_per_plane 32 --want_patch_for meniscus \
  --out feature_cache/
```

Idempotent and resumable: re-running skips finished `<enc>/<variant>/<split>/<exam>.pt`.

### C1 sanity check (single encoder, pooled only, CV)

```bash
python scripts/cv_select.py --feature_cache feature_cache/ --data_root MRNet-v1.0 \
  --encoders medsiglip --cache_variants clean,hflip,slicesB \
  --epochs 15 --k 5 --out cv_c1.tsv
```
Bar: mean per-task AUC ≥ **0.85**. If missed, frozen features are too weak — stop.

### C2 full phase-1 run (both encoders, meniscus pyramid, CV then 120-val once)

```bash
# CV for hyperparameters (sweeps d_model {192,256,384})
python scripts/cv_select.py --feature_cache feature_cache/ --data_root MRNet-v1.0 \
  --encoders medsiglip,dinov2 --epochs 25 --sweep_d_model 1 --k 5 --out cv_c2.tsv

# then the honest 120-val number, once, with the winning d_model
python train.py --prefix_name featbank_c2 --model_type featbank \
  --feature_cache feature_cache/ --data_root MRNet-v1.0 \
  --encoders medsiglip,dinov2 --d_model 256 --epochs 25 \
  --eval_tta_variants clean,hflip --save_model 1
```
Bars: meniscus ≥ **0.88**, ACL ≥ **0.95**, abnormal ≥ **0.93**.

### Flags

| flag | default | meaning |
| --- | --- | --- |
| `--feature_cache` | `""` | cache dir written by `build_feature_cache.py` |
| `--encoders` | `medsiglip,dinov2` | comma list; add a third here for phase 2 (re-cache first) |
| `--cache_variants` | all six | training-time variant pool (one sampled per epoch per exam) |
| `--slices_used` / `--slices_used_meniscus` | `24` / `32` | subsample of the 32 cached slices |
| `--d_model` | `256` | common projection width |
| `--featbank_batch_size` | `16` | real minibatches (slice count is fixed post-subsample) |
| `--eval_tta_variants` | `clean` | variants averaged at eval (TTA) |
| `--cv_folds` / `--cv_fold` | `0` / `-1` | set by `cv_select.py`; leave default for a 120-val run |
| `--select_metric` | *(auto)* | `per_task_mean` for `featbank`, `loss` otherwise; override explicitly if needed |

### Notes

- The frozen encoders never enter the training loop — only
  `scripts/build_feature_cache.py` constructs them. `run_featbank` starts from
  cached tensors.
- All tuning is done on 5-fold CV over the 1130 train exams (`cv_select.py`). Read
  the 120-val AUC once per phase; treating it as a tuning signal overfits it.
- Checkpoints: `model_<prefix>_featbank_ptmean_<x>_epoch_<e>.pth`, one best per run,
  prior `<prefix>` files pruned on each new best.
````

- [ ] **Step 2: Commit**

```bash
git add MEDICAL_MODELS.md
git commit -m "docs: Path 3 feature-bank section in MEDICAL_MODELS.md

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BfHgTqAMJopqaedcbF6me4"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
| --- | --- |
| §3.1 encoder roster (phase 1) | 2 (DINOv2, Fake), 3 (MedSigLIP) — BiomedCLIP correctly excluded per §11.1 R1 |
| §3.2 encoder wrapper contract (`encode_slices`) | 2 |
| §4.1 `build_feature_cache.py` (idempotent, resumable) | 4 |
| §4.2 cache layout + manifest + fp16 | 1, 4 |
| §4.3 cache variants (clean/hflip/rotp/rotn/slicesB/slicesC) | 4 (`VARIANTS`) |
| §4.4 disk budget + 16 GiB gate | 4 (`MIN_FREE_BYTES_FOR_PATCH`) |
| §5.1 `FeatureCacheDataset` payload + batching | 5 |
| §5.2 shared per-(enc,plane) projection trunk | 6 (`FeatBankMRNet.proj`) |
| §5.3 per-task heads (abnormal/acl pooled; meniscus pooled + pyramid) | 6 |
| §5.4 `consumes_feature_batch`, `[B,3]` TASKS-order contract | 6, 8 |
| §6 loss / optimizer / EMA / selection metric / CV guardrail | 7, 9 |
| §7 integration change list | 1–10 (one row each; `feature_bank.py` split into `feature_bank.py` + `feature_bank_medsiglip.py` to keep the MedSigLIP-only `transformers`-via-`medical_encoders` import path clean) |
| §9 C1/C2/C3 checkpoints | 10 (commands + bars); C3 attempt rule is §11.1 R5, a human decision after C2 |
| §11.1 R1–R5 rulings | R1→Task 2/3 scope, R2→Task 4/5, R3→Task 9 sweep, R4→Task 8, R5→Task 10 doc |

Gaps: §8 optional polish run — intentionally out of scope (stated in Global Constraints). C3 has no task because it is a re-run of Task 9/Task 8 commands after a human go/no-go, not new code.

**2. Placeholder scan** — no `TBD`/`TODO`/"add error handling"/"similar to Task N". Every code step has runnable code. The `feature_bank.build_encoder` `"medsiglip"` branch references `feature_bank_medsiglip` before Task 3 creates it — called out explicitly in Task 2 Step 3 as intentional and untested until Task 3.

**3. Type consistency**
- `encode_slices` returns `{"pooled": ..., "patch"?: ...}` — consistent across Tasks 2, 3, 4.
- Cache file structure `{plane: {"pooled": [S,D], "patch"?: [S,Dp,h,w]}}` — Tasks 1, 4 (write), 5 (read).
- `FeatureCacheDataset.__getitem__` → `(payload, label, weights, exam_id)`; `payload = {"pooled": {enc:{plane:T}}, "patch": {enc:{plane:T}}}` — Tasks 5, 6, 7 agree; `featbank_collate` adds a leading batch dim, `FeatBankMRNet.forward` consumes the batched form.
- `run_featbank(args) -> {"per_task_auc":[3], "per_task_mean":float, "pooled_auc":float, "best_epoch":int}` — Tasks 7, 9 agree.
- `make_folds(n, k, seed) -> list[np.ndarray]` — defined in Task 7, imported in Task 9.
- `--select_metric` choices `["loss","pooled_auc","per_task_mean"]` — Task 7 `_select_value`, Task 8 parser, Global Constraints agree.
- Head plane sets identical in Task 6 (`_HEAD_PLANES`) and Task 10 doc prose.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-05-multiencoder-pertask-mrnet.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
