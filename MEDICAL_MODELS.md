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
- `MedSigLIPEncoder` loads the full `google/medsiglip-448` with `AutoModel` and holds a
  reference to it so `get_image_features` stays available as a fallback (used only if a
  future `transformers` release stops returning `pooler_output` from the vision tower).
  That keeps the unused ~1.8 GB text tower resident for the whole run. It is inert for
  the current model — which always returns `pooler_output` — but it counts against the
  ≥32 GB unified-memory requirement.
- Encoding ~75–120 slices per exam through a 400M ViT on MPS is slow (minutes/epoch).
  Use `--max_train_batches` / `--time_budget_minutes` for short loops.
- Resize is bicubic + antialias to approximate the SigLIP processor; it is not a
  byte-exact match to `transformers`' PIL pipeline.

## MedGemma-4B LoRA (`vlm_finetune.py`)

A standalone LoRA fine-tune of [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it)
that classifies a knee MRI exam into the three MRNet labels. Each exam's three planes become
slice montages fed to the vision tower; the answer is a fixed JSON
(`{"abnormal": d, "acl": d, "meniscus": d}`). Scoring is deterministic and generation-free:
one teacher-forced forward per exam reads `P("1")` at each value slot via a two-way softmax,
so the reported `best_val_auc` is the **pooled micro-AUC** constructed identically to
`train.py`'s — its numbers sit in the same table as the CNN baselines. `per_task_auc` is a
supplementary line, not part of `train.py`'s block.

LoRA attaches to the **language model only** (`language_model.`-anchored target regex); the
SigLIP `vision_tower` and `multi_modal_projector` stay frozen (asserted at build time; a
`< 60M` trainable-param tripwire guards against a leak).

### One-time setup

1. Accept the MedGemma license on the model page.
2. `export HF_TOKEN=...`
3. `export PYTORCH_ENABLE_MPS_FALLBACK=1`
4. `pip install -r requirements.txt` (adds `peft`, `accelerate`, `sentencepiece`)

### Usage

```bash
# train (few hundred optimizer steps, not full epochs — ~10–25 min/epoch on MPS)
python vlm_finetune.py --prefix_name medgemma --epochs 3 --grad_accum 8 \
   --max_train_batches 300 --patience 3 --time_budget_minutes 90

# reload a saved best adapter and score validation only
python vlm_finetune.py --prefix_name medgemma --eval_only models/medgemma_medgemma_lora_valauc_0.xxxx \
   --dump_predictions out.tsv

# constrained-hardware multi-epoch training: one epoch per process, each resuming
# the best adapter on disk (a macOS OOM kill then costs <=1 epoch, not the whole run)
scripts/chain_finetune.sh 8 medgemma_full /path/to/MRNet-v1.0
# or a single manual resume step:
python vlm_finetune.py --prefix_name medgemma_full --epochs 1 \
   --resume_adapter models/medgemma_full_medgemma_lora_valauc_0.xxxx
```

### Flags (key ones)

| flag | default | meaning |
| --- | --- | --- |
| `--base_model` | `google/medgemma-4b-it` | HF id of the base multimodal model |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | `16` / `32` / `0.05` | LoRA adapter hyperparameters |
| `--slices_per_plane` / `--slice_strategy` | `6` / `uniform` | montage cells per plane and sampling strategy |
| `--patience` | `3` | early-stop patience on val AUC (`0` disables) |
| `--eval_only` / `--dump_predictions` | — | reload a saved adapter / write the prediction TSV |
| `--resume_adapter` | — | warm-start training from a saved LoRA adapter dir (kept trainable); `best_val_auc` is seeded from its `valauc_<f>` tag so a worse epoch can't overwrite it |
| `--time_budget_minutes` / `--max_train_batches` | — | budget guards for slow MPS runs |

### Notes

- `attn_implementation="eager"` and `do_pan_and_scan=False` are set everywhere: eager keeps the
   Gemma-3 sliding-window attention stable on MPS, and disabling pan-and-scan avoids a 3–5×
   token/memory blowup. Weights load in bf16 (`dtype=`), no `bitsandbytes` 4-bit — needs
   **≥32 GB** unified memory.
- **Two deliberate divergences from `train.py`** (spec §5.4): (1) early stopping keys on the
   **pooled val AUC** — `--patience` epochs with no AUC gain — whereas `train.py` early-stops
   on val *loss*; here val AUC is the entire point of the run. (2) The LR schedule is
   **cosine-with-warmup** across the whole run (`get_cosine_schedule_with_warmup`), not
   `train.py`'s `ReduceLROnPlateau` — a few-hundred-step LoRA run never plateaus meaningfully.
- Only the LoRA adapter is saved (`model.save_pretrained`); the ~4B base reloads from the HF
   cache. One best per run: on each new best the adapter is written first, *then* prior
   `<prefix>_medgemma_lora_valauc_*` dirs are pruned (the new dir is passed as `keep=`), so a
   crash mid-save never leaves zero checkpoints. On a `--resume_adapter` run `best_val_auc`
   starts at the checkpoint's own score, so an epoch that regresses saves nothing and the
   better prior adapter is left untouched. Fast tests use a hand-built fake processor/model
   in `conftest.py`; the real-model paths are `@pytest.mark.slow` (see
   `tests/test_vlm_real.py`).
