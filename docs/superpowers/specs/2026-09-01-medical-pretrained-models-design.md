# Medical Pretrained Models for MRNet — Design Spec

**Date:** 2026-09-01
**Status:** Approved for planning
**Scope:** Add two ways to use open-source medical pretrained models for MRNet knee-injury
classification: (1) a frozen **MedSigLIP** vision encoder wired into the existing training
pipeline, and (2) a **MedGemma-4B** vision-language model fine-tuned with LoRA as an
AUC-comparable classifier.

---

## 1. Context

### 1.1 The task

MRNet is an exam-level, multi-plane, 3D knee-MRI classification problem. Each exam has three
series — `sagittal`, `coronal`, `axial` — each a stack of `~12–40` grayscale slices at
`256×256`. The model predicts three independent binary labels:

- `abnormal` — any abnormality present (~80% positive, class-imbalanced)
- `acl` — ACL tear
- `meniscus` — meniscal tear

Primary metric: **per-task ROC-AUC** (`abnormal`, `acl`, `meniscus`) plus their **mean**,
computed with `sklearn.metrics.roc_auc_score` exactly as in `train.py:compute_auc`.

### 1.2 Existing pipeline (the parts we integrate with)

- `train.py` drives training through one contract: `model(sagittal, coronal, axial)` where
  each argument is `[B, slices, 3, H, W]` and `B == 1` (enforced; slice counts vary per exam).
  It owns the train/val loop, EMA, `ReduceLROnPlateau`, early stopping, per-batch AUC,
  checkpoint-on-best-val-AUC, and a final metrics block (`best_val_auc:` etc.).
- `lightweight_models.py` exposes `build_backbone(name, pretrained) -> (module, feature_dim)`
  and `FastMRNet`, which handles slice pooling (`max/mean/lse/attention/gem`), per-plane
  projection, plane fusion (`concat/plane_attention/plane_transformer`), an optional SE
  fusion gate, and the classifier head.
- `advanced_vit.py` shows the pattern for a heavier ViT backbone family behind
  `--model_type advanced|multiscale`.
- `dataloader.py` — `MRMultiPlaneDataset` yields `(volumes, label, weights, exam_id)` where
  `volumes` is a 3-tuple of `[slices, 256, 256]` float32 tensors; `MRVolumeAugmentor`
  applies study-consistent augmentation. Label column order is `("abnormal","acl","meniscus")`.
- `utils.py` — `prepare_volume_batch(volume, device, image_size=224, channels_last=False)`
  does `div_(255)`, gray→RGB `repeat`, bilinear resize to `image_size`, and **hardcoded
  ImageNet mean/std** normalization.
- `program.md` declares `dataloader.py` and `train.py`'s eval/metric extraction the "fixed
  harness" for the autoresearch loop.

### 1.3 Why these two models

MedSigLIP and MedGemma-4B share the **same SigLIP vision encoder** trained on de-identified
medical images (chest X-ray, dermatology, ophthalmology, histopathology). MedSigLIP is that
encoder standalone (image/text embeddings). MedGemma-4B is that encoder plus a Gemma-3 4B
language model. Path 1 consumes the raw image embeddings; path 2 makes the LM classify from
them. Neither was trained on knee MRI, so both require fitting on MRNet — path 1 fits a small
head, path 2 fits LoRA adapters.

### 1.4 Constraints (from brainstorming)

| Constraint | Value | Consequence |
|---|---|---|
| Compute | Apple Silicon Mac, MPS only | No `bitsandbytes` 4-bit. VLM runs bf16 + LoRA adapters. |
| Unified memory | ≥ 32 GB | MedGemma-4B bf16 LoRA is feasible on PyTorch/MPS; no MLX needed. |
| Dataset | `MRNet-v1.0/` present locally | End-to-end smoke tests possible; a synthetic fixture still added for CI. |
| HF gated access | Available (`HF_TOKEN`, licenses accepted) | MedSigLIP + MedGemma are the primary models; no ungated fallbacks. |
| Path 2 objective | AUC-comparable classifier | Fixed-format JSON target; scored by `1`-vs-`0` token probability, not generation. |

---

## 2. Goals / Non-goals

### Goals

1. `--model_type medsiglip` in `train.py` that trains a frozen-encoder `FastMRNet` variant
   using all existing pooling/fusion/head/EMA/early-stop machinery.
2. Optional on-disk per-slice embedding cache so head/fusion sweeps run at seconds/epoch on MPS.
3. A standalone MedGemma-4B LoRA fine-tuning script producing an adapter and per-task + mean
   val AUC in the same output format as `train.py`.
4. Deterministic, generation-free AUC scoring for path 2 (`1`-vs-`0` token probability at
   three fixed JSON value slots).
5. A test suite that runs without the 6 GB dataset (synthetic fixture), with real-weight
   tests marked `slow`.
6. Documentation: `MEDICAL_MODELS.md` + a README section.

### Non-goals

- No changes to `dataloader.py` label/volume logic or `autoresearch_loop.py`.
- No free-text radiology report generation (chosen objective is AUC-comparable classification).
- No RadImageNet / BiomedCLIP / Qwen / MLX fallbacks.
- No multi-GPU / distributed training.
- Path 2 is **not** added to the autoresearch search space.

---

## 3. Path 1 — MedSigLIP frozen encoder

### 3.1 New module: `medical_encoders.py`

```python
class MedSigLIPEncoder(nn.Module):
    """Frozen SigLIP vision tower from google/medsiglip-448.

    forward(flat_inputs: [N, 3, 448, 448]) -> [N, 1152]  pooled image embeddings
    """
    feature_dim = 1152

    def __init__(self, model_id="google/medsiglip-448"):
        # AutoModel.from_pretrained(model_id) -> SiglipModel; keep .vision_model only.
        # Prefer the model's pooled image-feature path (get_image_features equivalent)
        # so output is a single [N, 1152] vector per slice.
        # requires_grad_(False) on all params.

    def train(self, mode=True):
        # Override: always keep the wrapped tower in eval() regardless of parent mode
        # (no dropout / stochastic depth during feature extraction).
        super().train(mode)
        self._tower.eval()
        return self

    @torch.no_grad()
    def forward(self, flat_inputs):
        ...
```

Notes:
- The wrapped tower is a submodule so `model.to(device)` / `.state_dict()` behave normally,
  but its weights are excluded from checkpoints via `requires_grad` filtering at save time is
  **not** done (repo saves full `state_dict`); acceptable — adds ~1.8 GB to the `.pth`. To
  avoid that, `MedSigLIPEncoder.state_dict()` is overridden to return `{}` so the frozen
  tower is not serialized; it is always reloaded from HF on `build_model`. `train.py`'s
  `maybe_load_init_checkpoint` already tolerates missing keys (`strict=False`).
- Downloaded once to the HF cache (~1.8 GB). `HF_TOKEN` required.

### 3.2 `lightweight_models.py` changes

`build_backbone(name, pretrained)` gains:

```python
elif name == "medsiglip":
    from medical_encoders import MedSigLIPEncoder
    encoder = MedSigLIPEncoder()
    return encoder, MedSigLIPEncoder.feature_dim
```

`_strip_classifier` is not involved (the encoder already returns pooled features). Error
message listing supported backbones is updated to include `medsiglip`.

`FastMRNet` is otherwise **unchanged**: `_encode_planes` already does
`encoded = self.encoder(flat_inputs)` expecting `[N, feature_dim]`, then reshapes to
`[B, slices, feature_dim]` and pools. `feature_dim = 1152` flows through projection / fusion
/ classifier automatically.

### 3.3 Preprocessing generalization

`utils.prepare_volume_batch` signature becomes:

```python
def prepare_volume_batch(volume, device, image_size=224, channels_last=False,
                         mean=IMAGENET_MEAN, std=IMAGENET_STD):
```

- Defaults preserve current behavior byte-for-byte for existing model types.
- `mean` / `std` accept `[1,1,3,1,1]` tensors. New constants in `utils.py`:
  `SIGLIP_MEAN = torch.full((1,1,3,1,1), 0.5)`, `SIGLIP_STD = torch.full((1,1,3,1,1), 0.5)`.

`train.py` gains `resolve_input_spec(args) -> dict(image_size=int, mean=tensor, std=tensor)`:

| `args.model_type` | `image_size` | mean / std |
|---|---|---|
| `medsiglip` | 448 (ignores `--image_size`) | `SIGLIP_MEAN` / `SIGLIP_STD` |
| anything else | `args.image_size` | `IMAGENET_MEAN` / `IMAGENET_STD` |

`prepare_inputs(volumes, device, args)` reads the spec once and passes it through. This keeps
`dataloader.py` untouched; only tensor normalization generalizes, so the "fixed harness"
holds.

### 3.4 Freezing & optimizer

- `build_model(args)`: `--freeze_backbone` is tri-state with default `auto`:
  - `auto` (default) → freeze the encoder **only** for `model_type == medsiglip`; leave every
    other backbone fully trainable. This preserves current behavior byte-for-byte for
    `mobilenet_v3_small` / `resnet18` / `efficientnet_b0`.
  - `1` → force `model.encoder.requires_grad_(False)` for any backbone.
  - `0` → force the encoder trainable (no-op for `medsiglip`, whose `MedSigLIPEncoder`
    froze its own weights in `__init__` and is not un-frozen here).
- `train.py` optimizer construction changes to:
  ```python
  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
  ```
  Safe for every existing model type (all params trainable there today).
- `ModelEMA` already filters on `requires_grad` — frozen encoder is naturally excluded.

### 3.5 Embedding cache (opt-in)

New arg `--embed_cache_dir PATH` (default: unset → disabled).

When set and `model_type == medsiglip`:

1. **Precompute pass** (once, before epoch 0): iterate `MRMultiPlaneDataset` with
   augmentation forced off (`transform=None`), run the frozen encoder per slice, store
   per-`(exam_id, plane)` a `float16 [num_slices, 1152]` array at
   `PATH/medsiglip-448/{split}/{plane}/{exam_id}.npy`.
2. **Training**: a `CachedEmbeddingEncoder` replaces `MedSigLIPEncoder` inside `FastMRNet`.
   It is handed the already-loaded `[B, slices, 1152]` batch by a cache-aware
   `prepare_inputs` path and simply returns it (identity), so `FastMRNet`'s pooling /
   projection / fusion / head train normally at ~seconds/epoch.
3. **Cache key / validation**: `medsiglip-448` + `exam_id` + `plane` + `num_slices`. On
   mismatch (e.g. slice count differs) the entry is recomputed. A `cache_manifest.json`
   records model id, image size, normalization, and dataset root; a mismatch aborts with a
   clear message rather than silently using stale vectors.

Trade-off (documented in `MEDICAL_MODELS.md`): the cache disables image augmentation. Default
is **cache off** (augmentation preserved). Turn it on for head/fusion hyperparameter sweeps.

Approx storage: `1130 train + 120 val` exams × 3 planes × ~25 slices × 1152 × 2 bytes
≈ **210 MB**.

### 3.6 New `train.py` args (path 1)

| arg | type | default | purpose |
|---|---|---|---|
| `--embed_cache_dir` | str | `None` | Enable/point the per-slice embedding cache. |
| `--freeze_backbone` | `auto`/`0`/`1` | `auto` | `auto` = freeze only for `medsiglip`; preserves current behavior for all other backbones. `1`/`0` force freeze/trainable. |

`--model_type` choices gain `medsiglip`. `--image_size` is ignored for `medsiglip` (forced
448) with a one-line stderr note.

### 3.7 Path 1 usage

```bash
export HF_TOKEN=...
python train.py \
  --prefix_name medsiglip_gem_attn \
  --model_type medsiglip \
  --pooling gem \
  --plane_fusion plane_attention \
  --fusion_depth 3 --hidden_dim 192 --dropout 0.15 \
  --lr 3e-4 --weight_decay 5e-4 \
  --data_root MRNet-v1.0

# fast head/fusion sweep (augmentation disabled):
python train.py --prefix_name medsiglip_sweep --model_type medsiglip \
  --embed_cache_dir ~/.mrnet_embed_cache --pooling attention --data_root MRNet-v1.0
```

---

## 4. Path 2 — MedGemma-4B LoRA classifier

Standalone. Reuses `MRMultiPlaneDataset` and `train.py:compute_auc` by import; adds no
coupling to `train.py`'s loop.

### 4.1 `vlm_common.py` — pure helpers (no model state)

```python
LABEL_ORDER = ("abnormal", "acl", "meniscus")   # single source of truth for column order

def sample_slice_indices(num_slices: int, k: int, strategy: str = "uniform") -> list[int]:
    """Return k sorted, in-bounds slice indices.
    strategy="uniform": evenly spaced across the stack.
    strategy="center":  Gaussian-weighted toward the middle of the stack.
    TODO(user): ~5-8 line implementation. Affects what the model sees.
    """

def build_montage(volume: torch.Tensor, indices: list[int],
                  grid: tuple[int, int] = (3, 2), cell: int = 448) -> "PIL.Image":
    """Lay chosen slices into a grid image. Per-slice min-max -> uint8, resize to cell,
    grayscale->RGB. Output size = (cell*cols, cell*rows)."""

def build_messages(has_images: bool = True) -> list[dict]:
    """Chat structure:
      system:  "You are a musculoskeletal radiologist analyzing a knee MRI exam."
      user:    [sagittal img][coronal img][axial img] + instruction text (below)
      (assistant target is appended by the dataset / scorer, not here)
    """

INSTRUCTION = (
    "Sagittal, coronal, and axial series are shown as slice montages. "
    "Reply with only a JSON object with integer keys abnormal, acl, meniscus and "
    "values 0 or 1. abnormal = any abnormality present; acl = ACL tear; "
    "meniscus = meniscal tear."
)

def format_answer(labels: tuple[int, int, int]) -> str:
    """Deterministic: '{"abnormal": %d, "acl": %d, "meniscus": %d}' — fixed key order,
    one space after each colon, so value-token positions are stable."""

def mask_prompt_tokens(input_ids, assistant_start_idx):
    """Return labels tensor = input_ids with positions < assistant_start_idx set to -100."""

def score_exam(model, processor, messages, images) -> "np.ndarray[3]":
    """Deterministic AUC scoring — one teacher-forced forward, no generation.
    1. answer = format_answer((0, 0, 0))  # placeholder zeros
    2. tokenize prompt+answer; locate the 3 value-token indices; assert each is digit '0'
    3. one forward pass; at each value index take logits[idx-1], restrict to the token ids
       for '0' and '1', softmax -> p1 = P('1') / (P('0') + P('1'))
    4. return [p1_abnormal, p1_acl, p1_meniscus]
    TODO(user): ~6 line implementation of step 3's logit -> probability mapping.
    Alternatives to weigh: full-vocab softmax then read P('1'); include ' 1'/'1' variant
    ids; temperature. Renormalized two-way softmax is the standard MCQ-eval choice.
    """
```

### 4.2 `vlm_dataset.py` — `MRVLMDataset`

- Wraps `MRMultiPlaneDataset(root_dir, train=..., mmap=..., cache_size=...)` — same split,
  labels, `exam_id`. No `MRVolumeAugmentor` (montage rasterization + 4B LoRA: augmentation
  adds cost/noise for little gain; revisit later if needed).
- `__getitem__(i)` →
  ```python
  {
    "messages": build_messages(has_images=True),
    "images":   [sag_montage, cor_montage, ax_montage],   # 3 PIL images
    "label":    torch.tensor([...], dtype=torch.float32), # order == LABEL_ORDER
    "exam_id":  "0123",
  }
  ```
- `k` (slices per montage), `grid`, `cell`, `strategy` are constructor args.
- `collate_fn` keeps batch size 1 (matches the whole repo) and runs the `processor` to
  produce `input_ids`, `attention_mask`, `pixel_values` plus (for training) `labels` from
  `mask_prompt_tokens`, and (always) the raw `label` tensor + `exam_id`.

Token budget: 3 montages ≈ 3 × ~256 image tokens + ~150 text ≈ ~950 tokens/exam.

### 4.3 `vlm_finetune.py` — train / val / eval entrypoint

Arg style mirrors `train.py`.

| arg | default | notes |
|---|---|---|
| `--prefix_name` | (required) | run name; adapter dir + logs |
| `--data_root` | `MRNet-v1.0` | |
| `--base_model` | `google/medgemma-4b-it` | overridable for stub tests |
| `--epochs` | 3 | |
| `--lr` | 1e-4 | adapter LR |
| `--batch_size` | 1 | enforced == 1 |
| `--grad_accum` | 8 | effective batch |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | 16 / 32 / 0.05 | |
| `--slices_per_plane` | 6 | montage `k` |
| `--slice_strategy` | `uniform` | `uniform` \| `center` |
| `--warmup_ratio` | 0.03 | cosine schedule |
| `--patience` | 3 | epochs w/o mean-val-AUC improvement |
| `--time_budget_minutes` | `None` | wall-clock cap (same semantics as `train.py`) |
| `--max_train_batches` / `--max_val_batches` | `None` | |
| `--eval_only` | `None` | path to an adapter dir; skip training, run §5, print metrics |
| `--dump_predictions` | `None` | write `exam_id, p_abnormal, p_acl, p_meniscus, y_*` TSV |

**Model load:**
```python
processor = AutoProcessor.from_pretrained(base_model)
model = AutoModelForImageTextToText.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map={"": "mps"})
for p in model.parameters():
    p.requires_grad_(False)
peft_config = LoraConfig(
    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)   # vision_tower + multi_modal_projector frozen
```
Trainable ≈ 20–30M adapter params.

**Training step:** processor batch → `model(**batch).loss` (teacher-forced CE on the masked
assistant JSON). AdamW on adapter params, cosine schedule w/ warmup, `--grad_accum`,
`clip_grad_norm_(1.0)`. Weights already bf16 → no AMP context. `torch.mps.synchronize()`
around epoch timing (reuse `train.py:maybe_sync` via import or a local copy).

**Per-epoch validation:** run §5 scoring over the val split → per-task + mean AUC. Also track
mean teacher-forced val CE as `best_val_loss`. Save adapter with
`model.save_pretrained("models/<prefix>_medgemma_lora_valauc_{mean:.4f}/")` whenever mean val
AUC improves; delete prior dirs whose name contains `<prefix>` (mirrors `train.py`'s
one-best-per-run). Early stop on `--patience`.

**`--eval_only`:** load base + adapter (`PeftModel.from_pretrained`), run §5, print metrics.

**Output block (keys aligned with `train.py`):**
```
---
best_val_auc:        0.xxxxxx
best_val_loss:       0.xxxxxx        # mean teacher-forced val CE
per_task_auc:        abnormal=0.xxxx acl=0.xxxx meniscus=0.xxxx
best_epoch:          N
epochs_ran:          M
trainable_params_M:  0.03
model_type:          medgemma-4b-lora
device:              mps
```

### 4.4 Path 2 usage

```bash
export HF_TOKEN=...
export PYTORCH_ENABLE_MPS_FALLBACK=1

python vlm_finetune.py \
  --prefix_name medgemma_lora_v1 \
  --data_root MRNet-v1.0 \
  --epochs 3 --grad_accum 8 --slices_per_plane 6 \
  --time_budget_minutes 120

python vlm_finetune.py --eval_only models/medgemma_lora_v1_medgemma_lora_valauc_0.8543 \
  --data_root MRNet-v1.0 --dump_predictions preds.tsv
```

---

## 5. AUC-comparable scoring (path 2)

`score_exam` (§4.1) per val exam → `[p1_abnormal, p1_acl, p1_meniscus]`:

1. Build assistant string via `format_answer((0, 0, 0))` (placeholder zeros).
2. Tokenize `prompt + answer`; record the 3 value-token indices; assert each decodes to `0`.
3. **One teacher-forced forward.** At each value index `idx`, take `logits[idx - 1]`,
   restrict to the token ids for `"0"` and `"1"`, softmax → `p1 = P("1") / (P("0") + P("1"))`.
4. Return the 3 probabilities.

**Eval loop:** iterate val split, stack predictions → `[N, 3]`. Per-task
`compute_auc(y_true[:, i], y_pred[:, i])` (imported from `train.py`) + mean. Deterministic,
generation-free: ~120 exams × 1 forward ≈ 1–3 min on MPS.

**Design decision left to the user (`score_exam` step 3, ~6 lines):** two-way renormalized
softmax over `{"0","1"}` is the recommended default (standard for MCQ-style LLM eval,
well-behaved AUC). Alternatives to consider: full-vocab softmax then read `P("1")`; include
`" 1"`/`"1"` tokenizer variants; apply a temperature.

---

## 6. Dependencies

`requirements.txt` additions (MPS-compatible, conservative pins):

```
transformers>=4.53
peft>=0.13
accelerate>=1.0
pillow
sentencepiece
huggingface_hub>=0.26
```

No `bitsandbytes`, `open_clip`, or `mlx`. First run downloads ~1.8 GB (MedSigLIP) + ~8 GB
(MedGemma-4B) to the HF cache. Requires `HF_TOKEN` and both licenses accepted on
huggingface.co. `MEDICAL_MODELS.md` documents `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## 7. Testing

### 7.1 Synthetic fixture — `tests/conftest.py`

Writes a throwaway MRNet-shaped tree to a `tmp_path`: 6 train + 4 valid exams; each plane a
random `float32 [s, 256, 256]` npy with `s` random in `[12, 28]`; the 6
`{split}-{task}.csv` files with random 0/1 labels. A `mrnet_fixture` fixture returns the root
path. Lets the whole suite run without the 6 GB dataset.

### 7.2 Tests

| file | assertions |
|---|---|
| `tests/test_input_spec.py` | `resolve_input_spec` → `448` + `SIGLIP_*` for `medsiglip`, `args.image_size` + `IMAGENET_*` otherwise. `prepare_volume_batch` with custom mean/std → correct shape `[B, s, 3, size, size]` and expected value range. |
| `tests/test_medsiglip_backbone.py` | With a monkeypatched stub tower: `FastMRNet(backbone_name="medsiglip")` forward on `[1, s, 3, 448, 448]` → `[1, 3]`; all encoder params `requires_grad is False`; after `model.train()` the tower is still in `eval()` mode; `MedSigLIPEncoder.state_dict() == {}`. Real-weights variant behind `@pytest.mark.slow`. |
| `tests/test_vlm_common.py` | `build_montage` output size `== (cell*cols, cell*rows)`; `sample_slice_indices(n, k)` returns `k` sorted unique in-bounds ints for `uniform` and `center`; `format_answer` string is byte-exact; `score_exam` on hand-built logits returns the algebraically expected `p1`; value-slot digit assertion fires on malformed formatting. |
| `tests/test_vlm_dataset.py` | With `mrnet_fixture`: `__getitem__` returns 3 `PIL.Image` of the montage size, `label` shape `[3]` with column order `== LABEL_ORDER`, and `messages` well-formed (system+user, 3 image placeholders). `collate_fn` yields batch size 1 with `labels` masked (`-100` before the assistant turn). |
| `tests/test_vlm_finetune_smoke.py` | `@pytest.mark.slow`. `vlm_finetune.py --eval_only <stub-adapter> --max_val_batches 2` against a tiny random Gemma-3 model id → the metrics block prints with all keys. |

`pytest -m "not slow"` is the default fast gate. `slow` tests need `HF_TOKEN` + network.

---

## 8. File inventory

**New**

- `medical_encoders.py` — `MedSigLIPEncoder`, `CachedEmbeddingEncoder`.
- `vlm_common.py` — prompt/montage/slice/scoring helpers, `LABEL_ORDER`.
- `vlm_dataset.py` — `MRVLMDataset`, `collate_fn`.
- `vlm_finetune.py` — LoRA train / per-epoch val / `--eval_only`.
- `MEDICAL_MODELS.md` — both workflows, env vars, caveats.
- `tests/conftest.py` + `tests/test_input_spec.py` + `tests/test_medsiglip_backbone.py`
  + `tests/test_vlm_common.py` + `tests/test_vlm_dataset.py`
  + `tests/test_vlm_finetune_smoke.py`.

**Modified**

- `lightweight_models.py` — `build_backbone` gains a `medsiglip` branch; updated error text.
- `utils.py` — `prepare_volume_batch` gains `mean`/`std` params; add `SIGLIP_MEAN`/`SIGLIP_STD`.
- `train.py` — `resolve_input_spec`; `prepare_inputs` uses it; optimizer filters
  `requires_grad`; `build_model` honors `--freeze_backbone`; new args
  `--embed_cache_dir`, `--freeze_backbone`; `medsiglip` added to `--model_type` choices;
  embedding-cache precompute + `CachedEmbeddingEncoder` swap path.
- `requirements.txt` — six additions (§6).
- `README.md` — "Pretrained medical models" section linking `MEDICAL_MODELS.md`.

**Untouched:** `dataloader.py`, `autoresearch_loop.py`, `research_controller.py`,
`research_priors.py`, `pretrain_ssl.py`, `experiment_*.py`, `resnet.py`, `alexnet.py`,
`vit.py`, `advanced_vit.py`.

---

## 9. Risks & mitigations

1. **MedGemma-4B bf16 LoRA on MPS is slow** (~10–25 min/epoch on 1,130 train exams).
   Mitigations built in: `--time_budget_minutes`, `--max_train_batches`, `--grad_accum`.
   First realistic target is a few hundred steps, not 30 epochs.
2. **MedSigLIP live encoding on MPS** — ~90 ViT-L forwards/exam. `--embed_cache_dir` is the
   answer for head/fusion sweeps; live mode is for final augmented runs.
3. **MPS dtype quirks** — some `transformers` ops error or CPU-fallback in bf16 on MPS.
   `PYTORCH_ENABLE_MPS_FALLBACK=1` documented; smoke tests catch regressions.
4. **Small val set (120 exams)** — AUC has real variance. Both paths report per-task AUC so
   the spread is visible.
5. **Class imbalance** (`abnormal` ~80% positive) — path 1 inherits the repo's `pos_weight`.
   Path 2's JSON-token CE has no reweighting, but threshold-free AUC scoring makes this a
   training-signal concern, not a metric artifact. If `abnormal` recall lags, revisit with a
   weighted target or prompt tweak.
6. **Full-`state_dict` checkpoint bloat for path 1** — mitigated by `MedSigLIPEncoder`
   returning an empty `state_dict()`; the frozen tower always reloads from the HF cache.
7. **Tokenizer formatting drift** — if a `transformers` upgrade changes how the digit tokens
   or chat template render, `score_exam`'s digit-slot assertion fails loudly rather than
   producing silently wrong AUC.

---

## 10. Open items for implementation

- `sample_slice_indices` strategy bodies (`vlm_common.py`) — user contribution.
- `score_exam` logit→probability mapping (`vlm_common.py`) — user contribution.
- Confirm MedSigLIP pooled-feature access path (`get_image_features` vs `vision_model`
  pooler output) against the installed `transformers` version during implementation.
- Confirm MedGemma LoRA `target_modules` names match the released checkpoint's module paths.
