#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

OUT_DIR="full_training_outputs"
MODEL_DIR="$OUT_DIR/models"
mkdir -p "$OUT_DIR" "$MODEL_DIR"

run_locked() {
  local name="$1"
  shift

  echo "[$(date)] starting ${name}"
  MRNET_MODEL_DIR="$MODEL_DIR" python3 -u train.py \
    --prefix_name "$name" \
    --data_root MRNet-v1.0 \
    --model_type mobilenet_v3_small \
    --pretrained 1 \
    --pooling gem \
    --plane_fusion plane_attention \
    --plane_transformer_heads 4 \
    --fusion_depth 3 \
    --fusion_gate none \
    --projection_dim 128 \
    --hidden_dim 192 \
    --image_size 224 \
    --cache_size 32 \
    --num_workers 0 \
    --batch_size 1 \
    --mmap 1 \
    --amp 1 \
    --channels_last 1 \
    --epochs 36 \
    --patience 10 \
    --save_model 1 \
    --aug_policy knee_mri_plus \
    --aug_cutout_frac 0.18 \
    --aug_noise_std 0.04 \
    --aug_slice_dropout 0.03 \
    --aug_gamma_jitter 0.12 \
    --aug_spatial_shift_frac 0.03 \
    --loss_type focal \
    --focal_gamma 1.5 \
    --label_smoothing 0.03 \
    --ema_decay 0.995 \
    "$@" 2>&1 | tee "${OUT_DIR}/${name}.log"
}

# Architecture is fixed. Only optimization / regularization knobs vary.
run_locked gt95_locked_baseline_20260430 \
  --dropout 0.15 \
  --lr 0.00008 \
  --weight_decay 0.0005 \
  --val_tta_mode none

run_locked gt95_locked_low_lr_20260430 \
  --dropout 0.15 \
  --lr 0.00006 \
  --weight_decay 0.0005 \
  --val_tta_mode none

run_locked gt95_locked_low_wd_20260430 \
  --dropout 0.15 \
  --lr 0.00008 \
  --weight_decay 0.0003 \
  --val_tta_mode none

run_locked gt95_locked_light_reg_20260430 \
  --dropout 0.12 \
  --lr 0.00008 \
  --weight_decay 0.0003 \
  --val_tta_mode none

run_locked gt95_locked_stronger_reg_20260430 \
  --dropout 0.18 \
  --lr 0.00006 \
  --weight_decay 0.0007 \
  --val_tta_mode none
