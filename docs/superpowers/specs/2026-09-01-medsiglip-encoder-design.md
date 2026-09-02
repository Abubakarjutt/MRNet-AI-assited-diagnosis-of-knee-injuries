# MedSigLIP Frozen Encoder for MRNet — Design Spec

**Date:** 2026-09-01
**Status:** Approved for planning
**Scope:** Add `--model_type medsiglip` to `train.py`: a frozen `google/medsiglip-448`
vision tower used as the per-slice encoder inside `FastMRNet`, reusing the existing slice
pooling, plane fusion, classifier head, EMA, scheduler, and metric path unchanged.

**Sibling spec:** `2026-09-01-medgemma-lora-design.md` (path 2, MedGemma-4B LoRA). The two
share only the dataset, the `TASKS` tuple, and a documentation file; they are independent
implementations and independent PRs. This spec (path 1) is the lower-risk one and should
land first.

**Supersedes:** `2026-09-01-medical-pretrained-models-design.md` (combined spec, split after
independent review found the embedding cache non-viable as designed and the AUC metric
mis-described).

---

## 1. Context

### 1.1 The task and the metric (corrected)

MRNet is exam-level, multi-plane, 3D knee-MRI classification. Each exam has `sagittal`,
`coronal`, `axial` series (stacks of ~12–40 grayscale `256×256` slices) and three
independent binary labels: `abnormal` (~80% positive), `acl`, `meniscus`.

**How `train.py` actually scores (verified against source):**

- `iterate_epoch` flattens each `[1, 3]` prediction with `.reshape(-1)` and `.extend()`s the
  three per-task values into flat `y_preds` / `y_trues` lists **pooled across all tasks and
  all exams** (`train.py:242-246`).
- `compute_auc(y_trues, y_preds)` (`train.py:122-128`) is `sklearn.metrics.roc_auc_score`
  over those flat lists — **one pooled micro-AUC over `3 × N` points**. It returns **`0.5`**
  (not NaN) when only one class is present.
- The final metrics block (`train.py:551-561`) prints, at the best epoch:
  `best_val_auc` (that pooled micro-AUC), `best_val_loss`, `training_seconds`,
  `avg_epoch_seconds`, `epochs_ran`, `best_epoch`, `num_params_M`, `model_type`,
  `model_complexity`, `device`. **There is no per-task AUC line.**
- Checkpoint-on-best is on **val AUC** (`train.py:486-499`) and deletes prior files whose
  name contains `run_name`.
- Early stopping is on **val loss** with `patience_counter` vs `args.patience` (default 8)
  (`train.py:501-510`).
- `ReduceLROnPlateau` is on **val loss** with a **hardcoded `patience=3`**, `factor=0.3`,
  `threshold=1e-4` (`train.py:404-409, 471`) — `args.patience` does not affect it.

Path 1 routes entirely through this existing loop, so it inherits this metric and these
behaviors verbatim. No new metric code.

### 1.2 Existing integration points (verified)

- `build_model(args)` (`train.py:267-295`): for any `model_type` not in
  `{advanced, multiscale}`, sets `backbone_name = "resnet18" if args.model_type == "basic"
  else args.model_type` and constructs
  `FastMRNet(backbone_name=..., num_classes=3, pretrained=bool(args.pretrained), dropout=,
  pooling=, projection_dim=, hidden_dim=, fusion_depth=, fusion_gate=, plane_fusion=,
  plane_transformer_heads=)`.
- `--model_type` has an explicit `choices` list (`train.py:614-627`):
  `["basic","advanced","multiscale","resnet18","mobilenet_v3_small","efficientnet_b0"]`.
  **argparse rejects anything not in it** — `"medsiglip"` must be added.
- `lightweight_models.build_backbone(name, pretrained) -> (module, feature_dim)`
  (`lightweight_models.py:29-47`).
- `FastMRNet.__init__` sets `self.encoder, self.feature_dim = build_backbone(...)`
  (`lightweight_models.py:176`). `self.feature_dim` flows through projection / fusion /
  classifier sizing automatically.
- `FastMRNet._encode_planes` (`lightweight_models.py:237-257`) — **important shape
  contract**:
  - concatenates all three planes into **one** batch:
    `flat_inputs = torch.cat([plane.reshape(-1, *plane.shape[-3:]) for plane in
    plane_tensors], dim=0)` — expects each `plane` to be `[B, slices, 3, H, W]` so
    `plane.shape[-3:] == (3, H, W)`.
  - calls `self.encoder(flat_inputs)` **once** → expects a bare 2-D tensor
    `[B*(s_sag+s_cor+s_ax), feature_dim]`.
  - splits by `offset : offset + batch_size * slice_count` and
    `.reshape(batch_size, slice_count, self.feature_dim)` (`lightweight_models.py:251-252`).
  - So the encoder MUST accept `[N, 3, H, W]` and return `[N, feature_dim]` as a plain
    float tensor (not a HF `BaseModelOutputWithPooling` / tuple).
- `utils.prepare_volume_batch(volume, device, image_size=224, channels_last=False)`
  (`utils.py:36-56`): `div_(255)` → `unsqueeze(2).repeat(1,1,3,1,1)` gray→RGB →
  `F.interpolate(mode="bilinear", align_corners=False)` to `image_size` → hardcoded
  `IMAGENET_MEAN/STD` normalization. The `channels_last` parameter is already dead in the
  body (kept for signature stability).
- `prepare_volume_batch` is called only from `prepare_inputs` (`train.py:90-101`), which has
  two call sites: `forward_with_eval_policy` (`train.py:117`) and `iterate_epoch`
  (`train.py:222`). `pretrain_ssl.py` has its own preprocessing and is unaffected.
- Optimizer: `optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)`
  (`train.py:403`) — passes **all** params; needs a `requires_grad` filter for a partly
  frozen model.
- `ModelEMA` (`train.py:131-162`) already filters on `requires_grad` at construction and
  skips unknown names in `update`/`apply_to`/`restore` — a frozen encoder is naturally
  excluded, no change needed.
- `clip_grad_norm_(model.parameters(), 1.0)` (`train.py:237`) tolerates frozen params
  (None grads skipped).
- `maybe_load_init_checkpoint` (`train.py:312-350`) filters the incoming checkpoint to keys
  present in `model.state_dict()` and loads with `strict=False`.

### 1.3 Why MedSigLIP

`google/medsiglip-448` is the SigLIP SoViT-400m/14 vision+text model with a vision tower
pretrained on de-identified medical images (chest X-ray, dermatology, ophthalmology,
histopathology). Vision hidden size **1152**; native input **448×448**; SigLIP
normalization **`(x/255 − 0.5) / 0.5` → `[−1, 1]`** with **bicubic, antialiased** resize.
It was not trained on knee MRI, so a small trainable head is fit on top; the tower stays
frozen. Weights (~1.8 GB) download once to the HF cache; `HF_TOKEN` + accepted license
required.

### 1.4 Constraints

Apple Silicon Mac, MPS only, ≥32 GB unified memory, `MRNet-v1.0/` present locally, HF gated
access available.

---

## 2. Goals / Non-goals

### Goals

1. `--model_type medsiglip`: a frozen MedSigLIP vision tower as the `FastMRNet` per-slice
   encoder (`feature_dim = 1152`), with `--pooling`, `--plane_fusion`, `--fusion_depth`,
   `--hidden_dim`, `--projection_dim`, `--dropout`, `--ema_decay`, `--loss_type` all working
   as they do for the CNN backbones.
2. Correct, MPS-safe preprocessing for this encoder (448, `[−1,1]`, bicubic+antialias) via a
   contained generalization of `prepare_volume_batch` + a `resolve_input_spec` resolver — no
   change to `dataloader.py`.
3. Memory-bounded encoding on MPS: the frozen tower runs in slice-sized micro-batches, not
   one ~100-image batch.
4. Frozen tower excluded from `.pth` checkpoints (no 1.8 GB bloat) without breaking
   `torch.save(model.state_dict())` or warm-start loading.
5. `train.py`'s existing loop/metric/checkpoint path used verbatim.
6. Tests that run without the 6 GB dataset (synthetic fixture) and without the 1.8 GB model
   (stub tower); real-weight checks marked `slow`.
7. A `MEDICAL_MODELS.md` section + README pointer.

### Non-goals

- **No embedding cache.** (Cut after review: it cannot be done without invasive
  `_encode_planes` surgery, it disables augmentation, and it saves only ~0.2–0.4 GB. Listed
  as future work in §8.)
- No fine-tuning / unfreezing of the MedSigLIP tower.
- No changes to `dataloader.py`, `autoresearch_loop.py`, `research_*.py`, `pretrain_ssl.py`.
- No new metric; no per-task AUC line in `train.py`'s block.
- `medsiglip` is **not** added to the autoresearch search space.

---

## 3. Design

### 3.1 New module: `medical_encoders.py`

```python
import torch
import torch.nn as nn

class MedSigLIPEncoder(nn.Module):
    """Frozen SigLIP vision tower from google/medsiglip-448.

    Contract required by FastMRNet._encode_planes:
        forward(flat_inputs: FloatTensor[N, 3, 448, 448]) -> FloatTensor[N, 1152]
    """

    def __init__(self, model_id: str = "google/medsiglip-448", chunk_size: int = 32):
        super().__init__()
        from transformers import AutoModel
        full = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.tower = full.vision_model          # keep vision tower only
        self.feature_dim = 1152                 # instance attr, like the CNN backbones
        self.chunk_size = int(chunk_size)
        for p in self.tower.parameters():
            p.requires_grad_(False)
        self.tower.eval()

    def train(self, mode: bool = True):
        # keep the tower in eval() regardless of parent mode (no dropout / drop-path
        # during feature extraction)
        super().train(mode)
        self.tower.eval()
        return self

    @torch.no_grad()
    def forward(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        flat_inputs = flat_inputs.to(dtype=torch.float32)
        outs = []
        for start in range(0, flat_inputs.shape[0], self.chunk_size):
            chunk = flat_inputs[start:start + self.chunk_size]
            pooled = self.tower(pixel_values=chunk).pooler_output   # [c, 1152]
            outs.append(pooled.float())
        return torch.cat(outs, dim=0)                                # [N, 1152]

    # --- keep the 1.8 GB frozen tower out of checkpoints, safely ---
    def state_dict(self, *args, **kwargs):
        # Honor the recursive-call protocol: nn.Module.state_dict passes
        # destination=/prefix=/keep_vars= when a parent recurses into this child.
        destination = kwargs.get("destination")
        if destination is None and args:
            destination = args[0]
        if destination is None:
            from collections import OrderedDict
            destination = OrderedDict()
        return destination                       # contribute nothing

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        return                                    # nothing to load; tower comes from HF
```

Notes / decisions:
- **`.pooler_output`** is the `[N, 1152]` pooled image embedding. If a `transformers`
  version returns it under a different attribute, fall back to
  `full.get_image_features(pixel_values=...)` on the full model (implementation detail,
  resolved at build time — see §8).
- **`dtype=torch.float32`** pinned so it matches `prepare_volume_batch`'s float32 output on
  MPS (a bf16/fp16 tower would raise a dtype-mismatch). `dtype=` (not deprecated
  `torch_dtype=`).
- **`chunk_size=32`** (arg `--medsiglip_chunk`, §3.6) bounds MPS activation memory: instead
  of one forward over ~75–120 `448²` images, the tower sees ≤32 at a time. `_encode_planes`
  still calls the encoder once; the chunking is internal.
- **`state_dict` override** returns the caller's `destination` unchanged, so
  `FastMRNet.state_dict()` recursion (and `torch.save`) works and simply omits the tower.
  `_load_from_state_dict` is a no-op so `load_state_dict` never complains about the tower.
  `maybe_load_init_checkpoint` already tolerates the resulting missing keys (`strict=False`).

### 3.2 `lightweight_models.py` changes

`build_backbone(name, pretrained)` gains:

```python
elif name == "medsiglip":
    from medical_encoders import MedSigLIPEncoder
    encoder = MedSigLIPEncoder()        # `pretrained` is intentionally ignored:
                                        # weights always come from the HF checkpoint
    return encoder, encoder.feature_dim
```

- The "Unsupported backbone" message (`lightweight_models.py:24, 43-45`) gains `medsiglip`.
- `_strip_classifier` is not involved (encoder already returns pooled features).
- `FastMRNet` is **otherwise unchanged**. `_encode_planes`' single fused `self.encoder(...)`
  call now hits `MedSigLIPEncoder.forward`, which returns the required `[N, 1152]` tensor;
  the internal micro-batching is invisible to it.
- `--pretrained 0` is a **no-op** for `medsiglip` (documented in help text + `MEDICAL_MODELS.md`).

### 3.3 Preprocessing generalization

**`utils.py`:**
```python
SIGLIP_MEAN = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)
SIGLIP_STD  = torch.full((1, 1, 3, 1, 1), 0.5, dtype=torch.float32)

def prepare_volume_batch(volume, device, image_size=224, channels_last=False,
                         mean=IMAGENET_MEAN, std=IMAGENET_STD,
                         interp_mode="bilinear", antialias=False):
    ...
    flat = F.interpolate(flat, size=(image_size, image_size), mode=interp_mode,
                         align_corners=False if interp_mode in ("bilinear", "bicubic") else None,
                         antialias=antialias)
    ...
    flat = (flat - mean.to(device)) / std.to(device)
```
Defaults reproduce today's behavior byte-for-byte for every existing model type. `antialias`
is only honored by `bilinear`/`bicubic` in `F.interpolate` (fine).

**`train.py`:** new resolver, consulted once per batch inside `prepare_inputs`:
```python
def resolve_input_spec(args):
    if args.model_type == "medsiglip":
        return dict(image_size=448, mean=utils.SIGLIP_MEAN, std=utils.SIGLIP_STD,
                    interp_mode="bicubic", antialias=True)
    return dict(image_size=args.image_size, mean=utils.IMAGENET_MEAN,
                std=utils.IMAGENET_STD, interp_mode="bilinear", antialias=False)
```
`prepare_inputs(volumes, device, args)` keeps its signature; it reads the spec and forwards
the fields to `prepare_volume_batch`. Subprocess callers (`autoresearch_loop.py`,
`experiment_runner.py`) build CLI arg lists and are unaffected.

If `args.image_size != 224` while `model_type == "medsiglip"`, print a one-line stderr note
that `--image_size` is forced to 448.

### 3.4 Freezing & optimizer

- **`--freeze_backbone`** — tri-state, default `auto`:
  - `auto` → freeze the encoder **only** for `model_type == "medsiglip"`; every other
    backbone stays fully trainable (byte-for-byte current behavior for
    `mobilenet_v3_small` / `resnet18` / `efficientnet_b0`).
  - `1` → `model.encoder.requires_grad_(False)` for any backbone.
  - `0` → force trainable. For `medsiglip` this is a **no-op** (the tower froze itself in
    `__init__`); `build_model` emits a warning saying so.
- **Optimizer** (`train.py:403`) becomes:
  ```python
  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
  ```
  Behavior-preserving for all existing model types.
- `ModelEMA` unchanged (already `requires_grad`-filtered).

### 3.5 Metrics-block additions (small, all model types)

Two low-risk additions to the final block (`train.py:551-561`) so `medsiglip` runs are not
mis-summarized in autoresearch `results.tsv`:
- Add `trainable_params_M:` alongside the existing `num_params_M:` (the latter will read
  ~430M for `medsiglip` because the frozen tower counts — documented, not hidden).
- `model_complexity` (`train.py:520-539`) gains a `medsiglip` branch (value `0.6`,
  above `efficientnet_b0`'s `0.4`).

No other block keys change; `best_val_auc` stays the pooled micro-AUC.

### 3.6 New / changed `train.py` args

| arg | type | default | purpose |
|---|---|---|---|
| `--model_type` | str (choices) | `resnet18` | **add `"medsiglip"` to `choices`** |
| `--freeze_backbone` | `auto`/`0`/`1` | `auto` | freeze encoder; `auto` = medsiglip only |
| `--medsiglip_chunk` | int | `32` | slice micro-batch size inside `MedSigLIPEncoder` |

`--image_size` ignored (forced 448) for `medsiglip`, with a stderr note.

### 3.7 Usage

```bash
export HF_TOKEN=...
export PYTORCH_ENABLE_MPS_FALLBACK=1

python train.py \
  --prefix_name medsiglip_gem_attn \
  --model_type medsiglip \
  --pooling gem --plane_fusion plane_attention \
  --fusion_depth 3 --hidden_dim 192 --dropout 0.15 \
  --lr 3e-4 --weight_decay 5e-4 --ema_decay 0.995 \
  --data_root MRNet-v1.0
```

---

## 4. Dependencies

`requirements.txt` additions (MPS-compatible, conservative):

```
transformers>=4.56    # `dtype=` kwarg; current Gemma-3 / SigLIP handling
huggingface_hub>=0.26
pillow
pytest                # test-only; first test infra in this repo
```

No `peft`/`accelerate` (those are path 2). No `bitsandbytes`, `open_clip`, `mlx`. First run
downloads ~1.8 GB to the HF cache; `HF_TOKEN` + accepted `google/medsiglip-448` license
required. `MEDICAL_MODELS.md` documents `PYTORCH_ENABLE_MPS_FALLBACK=1` (must be exported
before `python` starts — it is read at `import torch`).

---

## 5. Testing

### 5.1 `tests/conftest.py`

- **Before importing torch**, set `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`.
- `mrnet_fixture` — writes a throwaway MRNet tree to `tmp_path`: 6 train + 4 valid exams;
  each plane a `float32 [s, 256, 256]` npy, `s` drawn from a **seeded** RNG in `[12, 28]`;
  the 6 `{split}-{task}.csv` files with labels forced to contain **both classes** per task
  (so `compute_auc` never sees a single-class column and AUC assertions are stable).
- `register` a `slow` marker (`pytest.ini` or `pyproject.toml`); `pytest -m "not slow"` is
  the fast gate.
- `stub_medsiglip` — monkeypatches `medical_encoders.AutoModel.from_pretrained` to return a
  tiny module whose `.vision_model(pixel_values=x)` yields an object with
  `.pooler_output = Linear(3*448*448 → 1152)(x.flatten(1))`-shaped output, so
  `MedSigLIPEncoder` can be exercised with no download.

### 5.2 Tests

| file | assertions |
|---|---|
| `tests/test_input_spec.py` | `resolve_input_spec` → `448 / SIGLIP_* / bicubic / antialias=True` for `medsiglip`; `args.image_size / IMAGENET_* / bilinear / False` otherwise. `prepare_volume_batch` with `mean=SIGLIP_MEAN, std=SIGLIP_STD, interp_mode="bicubic", antialias=True` → output shape `[B, s, 3, 448, 448]`, finite, range ≈ `[−1, 1]` for a `[0,255]` input. Default call byte-identical to pre-change output on a fixed tensor. |
| `tests/test_medsiglip_backbone.py` | With `stub_medsiglip`: `build_backbone("medsiglip", pretrained=1)` → `(module, 1152)`. `FastMRNet(backbone_name="medsiglip", pooling="gem", plane_fusion="plane_attention")` forward on 3×`[1, s, 3, 448, 448]` → `[1, 3]`. All encoder params `requires_grad is False`. After `model.train()`, `model.encoder.tower.training is False`. `MedSigLIPEncoder` forward with `chunk_size=4` over `s=10` gives the same result as `chunk_size=64` (micro-batch invariance). `torch.save(model.state_dict(), tmp)` succeeds and the file contains **no** `encoder.tower.*` keys; `load_state_dict(torch.load(tmp, weights_only=True), strict=False)` reports the tower keys as missing without error. |
| `tests/test_freeze_backbone.py` | `--freeze_backbone auto` leaves a `mobilenet_v3_small` backbone fully trainable; freezes a `medsiglip` encoder. `--freeze_backbone 1` freezes a `mobilenet` backbone. `--freeze_backbone 0` on `medsiglip` keeps it frozen and warns. Optimizer built from the model has `param_groups` param count == count of `requires_grad` params. |
| `tests/test_medsiglip_real.py` | `@pytest.mark.slow`. Needs `HF_TOKEN`. Real `MedSigLIPEncoder` on 2 random `448²` slices → `[2, 1152]`, dtype float32, finite. |

---

## 6. File inventory

**New**
- `medical_encoders.py` — `MedSigLIPEncoder`.
- `MEDICAL_MODELS.md` (repo root, beside `program.md` / `RESEARCH_DRIVEN_IMPROVEMENTS.md`) —
  path-1 section (path 2 appends its own section later).
- `tests/conftest.py`, `tests/test_input_spec.py`, `tests/test_medsiglip_backbone.py`,
  `tests/test_freeze_backbone.py`, `tests/test_medsiglip_real.py`.
- `pytest.ini` (or `[tool.pytest.ini_options]`) — `slow` marker registration.

**Modified**
- `lightweight_models.py` — `build_backbone` `medsiglip` branch; error-message text.
- `utils.py` — `prepare_volume_batch` gains `mean`/`std`/`interp_mode`/`antialias`;
  `SIGLIP_MEAN`/`SIGLIP_STD` constants.
- `train.py` — `resolve_input_spec`; `prepare_inputs` uses it; `--model_type` choices gain
  `medsiglip`; new `--freeze_backbone`, `--medsiglip_chunk`; `build_model` freeze logic +
  warning; optimizer `requires_grad` filter; final block gains `trainable_params_M`;
  `model_complexity` gains a `medsiglip` branch; `--image_size` stderr note.
- `requirements.txt` — 4 additions.
- `README.md` — "Pretrained medical models" pointer to `MEDICAL_MODELS.md`.

**Untouched:** `dataloader.py`, `autoresearch_loop.py`, `research_controller.py`,
`research_priors.py`, `pretrain_ssl.py`, `experiment_*.py`, `advanced_vit.py`, `resnet.py`,
`alexnet.py`, `vit.py`.

---

## 7. Risks & mitigations

1. **Un-chunked encoding OOM** — resolved by `MedSigLIPEncoder`'s internal `chunk_size`
   loop; `--medsiglip_chunk` lets the user tune it down on tighter machines.
2. **MPS throughput** — a frozen SoViT-400m over ~75–120 `448²` slices/exam is slow
   (minutes/epoch). Mitigations: `--max_train_batches`, `--time_budget_minutes` (both
   already in `train.py`). No cache in v1 (see §8).
3. **`pooler_output` attribute name drift across `transformers`** — resolved at build time;
   fallback is `full.get_image_features(...)`. `test_medsiglip_real.py` (slow) is the guard.
4. **`state_dict` override correctness** — the recursive-call protocol is handled explicitly
   (`*args/**kwargs`, honor `destination`); `test_medsiglip_backbone.py` asserts
   `torch.save` works and the tower is absent.
5. **`num_params_M` reads ~430M** for `medsiglip` — accepted and documented;
   `trainable_params_M` added so the real trainable size (~1–5M) is visible in logs and
   `results.tsv`.
6. **Resize still isn't the SigLIP processor's exact pipeline** — bicubic + antialias closes
   most of the gap; residual difference (PIL vs tensor resampling) is acceptable for a
   frozen encoder and noted in `MEDICAL_MODELS.md`.
7. **`program.md` "fixed harness"** — this spec edits `train.py` (the optimizer line,
   `prepare_inputs`, arg list, final block). Behavior is byte-identical for existing model
   types, but the spec acknowledges it is touching a file `program.md` calls fixed rather
   than implying otherwise. `dataloader.py` and the metric computation itself are untouched.

---

## 8. Deferred / future work

- **Embedding cache.** A standalone `precompute_embeddings.py` that writes per-`(exam,
  plane)` `[num_slices, 1152]` arrays, plus a first-class `CachedEmbeddingEncoder` **with an
  explicit `_encode_planes` branch** (on input `ndim` or encoder type) — not a `FastMRNet`
  identity hack. Would need `lightweight_models.py` surgery and forces augmentation off.
  Deferred until path 1 lands and head/fusion sweeps prove to be the bottleneck.
- Adding `medsiglip` to the autoresearch search space (out of scope; `program.md` change).

---

## 9. Open items for implementation

- Confirm the pooled-feature access path against the installed `transformers` version
  (`vision_model(...).pooler_output` vs `get_image_features(...)`); wire the fallback.
- Confirm `F.interpolate` accepts `antialias=True` with `mode="bicubic"` on the target torch
  version for MPS tensors (CPU fallback acceptable if not).
- Decide `model_complexity` value for `medsiglip` (spec proposes `0.6`).
