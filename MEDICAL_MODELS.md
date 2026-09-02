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
