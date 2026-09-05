# Multi-Encoder Per-Task Feature-Bank Classifier for MRNet — Design Spec

**Date:** 2026-09-05
**Status:** Draft — §11 open questions resolved (see §11.1); awaiting user review
**Scope:** A single-run, three-head MRNet classifier that consumes **cached features from a
bank of 2–3 frozen pretrained encoders**, with per-task specialization (plane emphasis,
feature level, pooling, augmentation, decision threshold) and a dedicated multi-scale head
for meniscus. Extends `train.py`'s existing pipeline; reuses its EMA, scheduler, AUC eval,
and checkpointing unchanged.

**Goal (the number we are chasing):** beat the published **120-exam validation-set**
per-task AUCs — push **meniscus past ~0.91** and **ACL past ~0.97** on `valid-*.csv` —
measured locally, no competition submission. `abnormal` (~0.94 in the literature) is the
easy task and mostly a sanity check.

**Sibling specs:** `2026-09-01-medsiglip-encoder-design.md` (frozen MedSigLIP encoder inside
`FastMRNet`) and `2026-09-01-medgemma-lora-design.md` (MedGemma-4B LoRA). This spec is
"Path 3." It **reuses** the MedSigLIP encoder wrapper from Path 1 and the `dataloader.TASKS`
label order; it is otherwise independent.

**Explicitly NOT in scope:** competition-portal submission packaging; full backbone
fine-tuning; the MedGemma LoRA path (frozen at pooled AUC 0.8000, iter 5 of the Sep-4/5
chain); the separate augmentation work for `vlm_dataset.py` (tracked independently).

---

## 1. Context

### 1.1 The task and the two metrics

MRNet: exam-level, 3 series (`sagittal`/`coronal`/`axial`, ~12–40 grayscale `256×256`
slices), 3 independent binary labels — `abnormal` (~80% positive), `acl` (~23%),
`meniscus` (~37%). Split: 1,130 train / 120 validation (`valid-*.csv`) / 120 hidden test
(unavailable).

- **Primary reported number (this spec):** the **per-task AUC** on the 120-val set —
  `roc_auc_score` computed separately for each of the three columns. This is what every
  MRNet paper reports and what our goal is stated against.
- **Parity number:** the **pooled micro-AUC** over the flattened `[120, 3]` matrix — the
  identical construction as `train.py`'s `best_val_auc` (`train.py:242-248` flatten +
  `compute_auc`). Printed alongside so Path 3 sits in the same table as the CNN baselines
  and the MedGemma run.

### 1.2 Why the published SOTA is hard to match on this hardware

- Every SOTA result (MRNet 2018, ELNet, MRPyrNet, attention/SSL variants) **fine-tunes the
  vision backbone** on the 3-plane volume and usually trains **per-task specialists**.
- We are MPS-only: ~32 GB unified memory, one run at a time, hours per epoch if a big
  backbone is in the training loop. Full fine-tuning of even one ViT-B across ~100 slices ×
  3 planes is impractical for the sweep volume we need.
- **The bet:** strong *medical* encoders (MedSigLIP, BiomedCLIP) and strong *self-supervised*
  encoders (DINOv2) did not exist when most MRNet papers were written. Their **frozen**
  features, **combined** across encoders and fed to well-designed per-task heads, may close
  most of the gap that frozen-feature methods historically had against full fine-tuning.
- **Honest caveat:** this is a bet, not a guarantee. Frozen features may plateau below
  0.91 meniscus / 0.97 ACL. §9 defines the go/no-go checkpoints so we find out cheaply.

### 1.3 What we reuse from the repo

| Component | Source | Reuse |
| --- | --- | --- |
| Split logic, label construction, `exam_id`, `mmap` | `dataloader.MRMultiPlaneDataset` | wrapped by the new feature-cache dataset |
| `TASKS = ("abnormal", "acl", "meniscus")` | `dataloader` | imported, single source of label order |
| `PLANES = ("sagittal", "coronal", "axial")` | `dataloader` | imported |
| `MRVolumeAugmentor` (study-consistent aug) | `dataloader` | drives the cache-variant generation (§4.3) |
| Frozen MedSigLIP tower wrapper | `medical_encoders.MedSigLIPEncoder` | one entry in the encoder bank |
| Training loop, EMA, `ReduceLROnPlateau`, best-ckpt, run summary block | `train.py` | **unchanged**; new `--model_type` slots into `build_model` |
| `compute_auc`, pooled + per-task eval | `train.py` / mirrored helper | unchanged |

---

## 2. Architecture overview

```
OFFLINE (one-time, slow, per encoder × per cache-variant):
  raw exam volumes ──► MRVolumeAugmentor(variant) ──► slice sample ──► frozen encoder
                                                                          │
                          ┌───────────────────────────────────────────────┤
                          ▼                                               ▼
                  pooled per-slice emb                          patch-token grid
                  [S, D_pool]                                    [S, D_patch, h, w]
                          └──────────────► feature_cache/<enc>/<variant>/<split>/<exam>.pt

TRAIN (one run, seconds/epoch — features are cached):
  cache ──► FeatureCacheDataset ──► FeatBankMRNet
                                      ├── abnormal head : pooled, 3 planes, GeM→plane-attn→MLP
                                      ├── acl head      : pooled, sagittal+coronal, GeM→plane-attn→MLP
                                      └── meniscus head : pooled + patch-grid, sagittal+coronal,
                                                          multi-scale pyramid head (§5.3)
                                      loss = Σ w_task · focal_task
  reuse train.py: EMA, scheduler, per-task + pooled AUC, best checkpoint

EVAL:
  per cache-variant → head logits → average over variants (TTA)
  report per-task AUC (primary) + pooled micro-AUC (parity)
  per-task threshold calibrated on a train-side CV fold (§6)
```

Per-task specialization lives **entirely in the trainable heads and the eval config**, never
in the encoder weights (they are shared, frozen, and cached once per encoder). This is the
resolution of the Q&A point: separate *encoder weights* per task would be pointless when
frozen; separate *heads* consuming a *shared multi-encoder feature bank* is the useful move.

---

## 3. The encoder bank

All encoders are **frozen** (`requires_grad_(False)`, `.eval()`, `torch.no_grad()` forward),
run in slice micro-batches to bound MPS memory, and are **never** part of the training loop —
they exist only in the offline caching script.

### 3.1 Encoder roster (phased)

| Phase | Encoder | HF id | Feature outputs | Notes |
| --- | --- | --- | --- | --- |
| 1 | **MedSigLIP-448** | `google/medsiglip-448` | pooled 1152-d; patch grid 32×32×1152 | medical image–text; reuse `MedSigLIPEncoder`; gated (HF_TOKEN); input forced 448 |
| 1 | **DINOv2 ViT-B/14** | `facebook/dinov2-base` | pooled 768-d (CLS); patch grid 16×16×768 | self-supervised; strong local/dense features; ungated |
| 2 (if phase 1 underperforms) | **BiomedCLIP ViT-B/16** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | pooled 512-d; patch grid 16×16×768 | medical image–text; loaded via `open_clip`; input 224 |

Phase 1 = MedSigLIP + DINOv2 (one medical, one self-supervised — maximally complementary).
Add BiomedCLIP only if the §9 phase-1 checkpoint misses its bar.

### 3.2 Encoder wrapper contract

New file `feature_bank.py` (transformers/open_clip imports isolated here, mirroring
`medical_encoders.py`). One class per encoder, all implementing:

```python
class FrozenEncoder(Protocol):
    name: str                    # "medsiglip" | "dinov2" | "biomedclip"
    pooled_dim: int
    patch_dim: int
    patch_grid: tuple[int, int]  # (h, w) of the token grid
    input_size: int              # 448 | 224 | ...

    @torch.no_grad()
    def encode_slices(
        self, slices_u8: Tensor,          # [S, H, W] uint8 grayscale, any H/W
    ) -> dict:                            # resized + 3-ch replicated + normalized internally
        # returns {"pooled": [S, pooled_dim] float32,
        #          "patch":  [S, patch_dim, h, w] float32}   # patch omitted if not requested
```

`MedSigLIPEncoder` is adapted (not rewritten) to also expose the patch grid
(`last_hidden_state` reshaped) behind a `want_patch` flag.

---

## 4. Offline feature cache

### 4.1 Script

`scripts/build_feature_cache.py`:

```
python scripts/build_feature_cache.py \
  --data_root MRNet-v1.0 \
  --encoders medsiglip,dinov2 \
  --variants clean,hflip,rotp,rotn,slicesB,slicesC \
  --slices_per_plane 32 --slice_strategy uniform \
  --want_patch_for meniscus \
  --out feature_cache/
```

Cache at 32 slices/plane (§11.1 R2); `FeatureCacheDataset` subsamples to `--slices_used`
(default 24) at load time, so slice count is a CV knob rather than a re-cache.

Idempotent: skips an `<enc>/<variant>/<split>/<exam_id>.pt` that already exists and passes a
version/shape check. Resumable after an MPS OOM kill (per-exam granularity).

### 4.2 Cache layout & format

```
feature_cache/
  manifest.json                     # encoders, variants, slice params, dims, git sha, schema v
  medsiglip/
    clean/train/0000.pt             # {"sagittal": {...}, "coronal": {...}, "axial": {...}}
    clean/valid/1130.pt
    hflip/train/0000.pt
    ...
  dinov2/
    ...
```

Each `<exam>.pt` per plane holds:
- `pooled`: `float16` `[S, pooled_dim]` (S = `slices_per_plane` = 32 on disk; the dataset
  subsamples to `slices_used` at load)
- `patch`: `float16` `[S, patch_dim, h, w]` — **only** for planes/encoders enabled via
  `--want_patch_for` (meniscus branch uses sagittal+coronal patch grids; that is the only
  patch consumer, to keep disk sane).

### 4.3 Cache variants (this IS the augmentation for Path 3)

Variants are generated by `MRVolumeAugmentor` with fixed per-variant plans, applied
**study-consistently** across the three planes (the augmentor already does this via
`sample_plan`):

| variant | transform | rationale |
| --- | --- | --- |
| `clean` | identity | the reference features |
| `hflip` | horizontal flip (`dims=[2]`) | left/right knee symmetry — safe for all 3 labels |
| `rotp` / `rotn` | ±8–10° in-plane rotation | scanner positioning variance |
| `slicesB` / `slicesC` | identity image, **different slice sampling** (offset / `center` strategy) | per-"epoch" montage diversity — the highest-value aug for a frozen encoder; attacks overfitting to one fixed slice set |

**No** vertical flip (anatomically invalid), **no** elastic/grid distortion (destroys small
meniscal-horn geometry), **no** aggressive intensity aug at cache time (kept for the optional
live-aug polish run, §8). Train-time consumes all variants as extra samples; eval-time
averages head logits over variants (TTA).

### 4.4 Disk budget

Per encoder, per variant, per split, pooled-only (at the 32-slice cache, §11.1 R2):
~1,250 exams × 3 planes × 32 slices × 768–1152 dims × 2 B (fp16) ≈ **0.17–0.27 GB**.
6 variants × 2 encoders pooled ≈ **2.1–3.2 GB**. Patch grids (meniscus only, 2 planes,
2 encoders, 6 variants): DINOv2 16×16×768 fp16 ≈ 0.4 MB/slice → ~1,250 × 32 × 2 × 2 × 6 ×
0.4 MB ≈ **8 GB**. **Total ≈ 10–12 GB.**

This is tight against the current ~11 GB free (§ ledger notes disk fluctuates 8–16 GB).
**Cache-build gate:** require ≥ 16 GB free before the patch-grid pass, or apply a fallback
in this order: (a) patch grids for `clean` + `hflip` variants only (rot variants get
pooled-only) → ~3 GB patch, total ~6 GB; (b) fp8-quantize patch features; (c) drop
`rotp/rotn` entirely. Pooled-only phase 1 (C1) needs just ~1 GB and has no gate.

---

## 5. The model — `FeatBankMRNet`

New class in `lightweight_models.py` (or `feature_bank.py` if it grows large; decide at
plan time). No backbone — it starts from cached features.

### 5.1 Inputs (from `FeatureCacheDataset`, batch_size configurable now that it is cheap)

```
batch = {
  "pooled": {enc: {plane: [B, S, pooled_dim_enc]}},   # all encoders, all 3 planes
  "patch":  {enc: {plane: [B, S, patch_dim_enc, h, w]}},  # meniscus planes/encoders only
  "label":  [B, 3],                                    # TASKS order
  "exam_id":[B],
}
```

### 5.2 Shared trunk (per plane, per encoder)

For each `(encoder, plane)`: `LayerNorm → Linear(pooled_dim_enc → d_model=256) → GELU`.
This projects every encoder into a common `d_model` so heads can concatenate/gate across
encoders without caring about native dims.

### 5.3 Per-task heads

**abnormal head** — planes = all 3; features = pooled only.
- per (enc, plane): GeM slice-pool (`p` learnable, from `lightweight_models.GeMPooling`) →
  `[B, d_model]`
- per plane: mean over encoders (or a 2-logit softmax gate) → `[B, 3 planes, d_model]`
- `PlaneAttentionFusion(d_model)` → `[B, d_model]` → MLP(`d_model→128→1`)

**acl head** — planes = sagittal + coronal (ACL is a sagittal-plane call, coronal
corroborates); features = pooled only. Same structure as abnormal over 2 planes.

**meniscus head** — planes = sagittal + coronal; features = pooled **+ patch grid**.
- pooled path: as acl head → `[B, d_model]`
- **pyramid path** (the MRPyrNet insight, small-structure sensitive):
  - patch grid `[B, S, patch_dim, h, w]` → per-slice: 2 parallel depthwise-sep conv blocks
    at strides {1, 2} → GAP each → concat → `Linear(→ d_model)` → `[B, S, d_model]`
  - attention slice-pool over S (a small `AttentionPooling`, already in
    `lightweight_models`) → `[B, d_model]` per (enc, plane)
  - mean over encoders, `PlaneAttentionFusion` over the 2 planes → `[B, d_model]`
- concat(pooled path, pyramid path) → `[B, 2·d_model]` → MLP(`2·d_model→128→1`)

All three heads are tiny (≈0.1–0.5 M params each). Total trainable ≈ 1–3 M.

### 5.4 `forward(...)` signature

To slot into `train.py` without touching the loop, `FeatBankMRNet.forward` accepts the
batch dict (not `(sagittal, coronal, axial)` tensors). This requires a **1-line branch** in
`train.py`'s train/val step: if `model.consumes_feature_batch` is truthy, call
`model(batch)`; else the existing `model(sag, cor, ax)`. This is the only `train.py` change.
Returns `[B, 3]` logits in TASKS order — identical downstream contract.

---

## 6. Training & selection

- **Loss:** `Σ_task w_task · focal(logit_task, y_task)`, `w` = inverse-prevalence-ish,
  tunable. Optionally per-task `label_smoothing`.
- **Optimizer:** AdamW, per-task-group LR allowed (heads are separate module groups).
- **Schedule / EMA / best-ckpt:** unchanged `train.py` machinery. Selection metric =
  **mean of the 3 val per-task AUCs**. `train.py` gains
  `--select_metric {loss,pooled_auc,per_task_mean}`; the default is resolved after arg
  parsing — `per_task_mean` when `model_type == "featbank"`, today's behaviour otherwise
  (§11.1 R4).
- **Model selection guardrail (critical):** the 120-val set is tiny → picking hyperparams
  by watching val AUC overfits it. Use **5-fold CV on the 1,130 train exams** for all
  hyperparameter and threshold decisions (cheap: features are cached, a fold trains in
  minutes). The 120-val AUC is looked at **once per phase** as the honest number, not as a
  tuning signal. `scripts/cv_select.py` runs the folds and writes `cv_results.tsv`.
- **Thresholds:** per-task operating threshold chosen on CV out-of-fold predictions; AUC is
  threshold-free so this only affects a reported F1/accuracy sidebar, not the headline.

---

## 7. Integration points (the full change list)

| File | Change |
| --- | --- |
| `feature_bank.py` (new) | frozen encoder wrappers (MedSigLIP patch-grid adapter, DINOv2, later BiomedCLIP); `FrozenEncoder` protocol |
| `scripts/build_feature_cache.py` (new) | offline caching pass; idempotent, resumable; writes `manifest.json` |
| `scripts/cv_select.py` (new) | 5-fold CV over cached features for hyperparam/threshold selection |
| `dataloader.py` | add `FeatureCacheDataset` (reads cache, subsamples 32→`--slices_used` slices, yields batch dict, shuffles variant per epoch); no change to existing classes |
| `lightweight_models.py` | add `FeatBankMRNet` (+ `consumes_feature_batch = True`); reuse `GeMPooling`, `AttentionPooling`, `PlaneAttentionFusion` |
| `train.py` | `build_model`: `elif args.model_type == "featbank": model = FeatBankMRNet(...)`; 1-line `forward` branch on `consumes_feature_batch`; new args `--encoders`, `--feature_cache`, `--cache_variants`, `--slices_used`, `--slices_used_meniscus`, `--d_model`, `--select_metric {loss,pooled_auc,per_task_mean}` (default resolved post-parse: `per_task_mean` iff `model_type == "featbank"`) |
| `medical_encoders.py` | MedSigLIP wrapper gains optional `want_patch` returning the token grid (backward compatible) |
| `MEDICAL_MODELS.md` | new "Path 3 — Multi-encoder feature bank" section: setup, cache build, train, expected numbers |
| `tests/` | fast-suite tests (fakes, no downloads): cache format round-trip, `FeatureCacheDataset` batch shape, `FeatBankMRNet` forward shape per head, `train.py` forward-branch dispatch, CV splitter determinism. Real-encoder paths `@pytest.mark.slow`. |

**Fast-suite invariants preserved:** no test downloads a model, needs MPS, or needs
`HF_TOKEN`. All new fast tests use synthetic cached tensors and a fake encoder.

---

## 8. Optional polish run (only if we are close)

If a phase clears its §9 bar and we want the last ~0.5 AUC point: one **no-cache** run with
the encoders in the loop under `torch.no_grad()` and **live** `MRVolumeAugmentor` (full
`knee_mri_research` policy) so augmentation is not frozen to 6 variants. Slow
(~30–90 min/epoch), few epochs, EMA on. This is a stretch goal, not part of the core plan.

---

## 9. Go / no-go checkpoints

| Checkpoint | Bar | If missed |
| --- | --- | --- |
| **C1 — single encoder sanity** (MedSigLIP pooled only, 3-head, no pyramid, CV) | mean per-task AUC ≥ **0.85** (≈ current MedGemma 0.80 + a clear margin) | frozen features are too weak → stop, reconsider (fine-tune a small backbone instead) |
| **C2 — phase 1 full** (MedSigLIP + DINOv2, pyramid meniscus head, TTA, CV→120-val once) | meniscus ≥ **0.88**, ACL ≥ **0.95**, abnormal ≥ **0.93** | add BiomedCLIP (phase 2); revisit pyramid head; try patch grids for ACL too |
| **C3 — phase 2 / polish** | **meniscus ≥ 0.91, ACL ≥ 0.97** (the goal) | accept the near-miss as the result; document the frozen-feature ceiling honestly |

Each checkpoint is one CV sweep = hours on MPS, not days.

---

## 10. Risks

1. **Frozen ceiling** — the headline risk (§1.2, §9-C1). Mitigated by cheap early go/no-go.
2. **120-val overfitting** — mitigated by mandatory CV-for-selection (§6); the 120 number is
   read once per phase.
3. **Disk** — cache is 8–10 GB against an 8–16 GB free envelope. Mitigated by patch-grid
   scope limits and fp16; fallbacks in §4.4.
4. **MedSigLIP patch-grid adapter** — touching a Path-1 file. Mitigated: additive, behind a
   flag, Path-1 tests must stay green.
5. **`train.py` forward branch** — a real change to the shared loop. Mitigated: 1 line,
   guarded by an attribute default-false for every existing model; covered by a dispatch
   test.
6. **Encoder input mismatch** — MedSigLIP wants 448, DINOv2 14-patch/518-ideal, BiomedCLIP
   224. Each wrapper owns its own resize; cache stores post-encoder features so the training
   side never sees the discrepancy.

---

## 11. Open questions for review

1. **Encoder roster** — start MedSigLIP + DINOv2, or include BiomedCLIP from the start
   (more disk, more caching time, but skips a phase)?
2. **Slice count** — 24/plane at cache time is a memory/detail trade. Meniscus may want
   more (32–40). Cache the max and subsample in the dataset?
3. **`d_model`** — 256 is a guess; could sweep {192, 256, 384} in CV.
4. **Selection metric default** — leave `train.py` default unchanged and require
   `--select_metric per_task_mean` for Path 3 (proposed), or make per-task-mean the default
   when `model_type == featbank`?
5. **Is C3 worth attempting** if C2 lands at, say, meniscus 0.89 / ACL 0.96 — or do we
   declare that the result and stop?

### 11.1 Resolutions (2026-09-05)

Rulings adopted for the implementation plan. Each is reversible mid-project if a
checkpoint contradicts it; the cost of a wrong call here is one extra CV sweep or one
re-cache, both bounded.

1. **Encoder roster — phased. Phase 1 = MedSigLIP-448 + DINOv2 ViT-B/14 only.**
   BiomedCLIP is deferred to phase 2 and added only if the C2 checkpoint (§9) misses its
   bar. Rationale: each encoder adds ~1 GB cache and a multi-hour caching pass on MPS; the
   medical + self-supervised pair is the most complementary starting point; C2 is the
   honest signal for whether a third view is needed. `--encoders` still accepts a
   comma list so phase 2 is a re-run, not a code change.

2. **Slice count — cache 32/plane, subsample in the dataset.** `build_feature_cache.py`
   runs at `--slices_per_plane 32` (`uniform`). `FeatureCacheDataset` takes
   `--slices_used` (default 24) and subsamples the cached 32 → 24 by uniform stride at
   load time; the meniscus head may request the full 32 via `--slices_used_meniscus 32`.
   Rationale: re-caching is the expensive, irreversible-in-practice op; over-caching by
   33 % costs ~0.05 GB/encoder/variant (well inside §4.4) and makes slice count a free
   CV knob instead of a recache.

3. **`d_model` — default 256, swept `{192, 256, 384}` in the C1 and C2 CV.** Exposed as
   `--d_model`. Not a blocker; the sweep rides along in the CV that runs for C1/C2 anyway.

4. **Selection metric — `per_task_mean` is the default when `model_type == "featbank"`;
   unchanged for every other model type.** `train.py` gains `--select_metric
   {loss,pooled_auc,per_task_mean}`; its default is resolved *after* arg parsing:
   `featbank` → `per_task_mean`, all others → today's behaviour. Rationale: Path 3's whole
   purpose is per-task AUC vs the published per-task numbers; requiring an opt-in flag is a
   silent-footgun where a forgotten flag invalidates a multi-hour run. Other paths see no
   behaviour change.

5. **C3 is attempted only if C2 lands within 0.02 of every C3 bar** — i.e. C2 gives
   meniscus ≥ 0.89 **and** ACL ≥ 0.95 **and** abnormal ≥ 0.93. Otherwise C2 is declared
   the result and the frozen-feature ceiling is documented (§9-C3 "if missed" row).
   Rationale: bounded MPS compute; a >0.02 gap after phase 2 is a ceiling, not a
   tuning-away distance, and chasing it indefinitely has no stopping rule.
