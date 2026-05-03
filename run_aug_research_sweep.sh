#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

OUT_DIR="full_training_outputs"
MODEL_DIR="$OUT_DIR/models"
mkdir -p "$OUT_DIR" "$MODEL_DIR"

run_aug() {
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
    --epochs 30 \
    --patience 8 \
    --save_model 1 \
    --loss_type focal \
    --focal_gamma 1.5 \
    --label_smoothing 0.03 \
    --ema_decay 0.995 \
    --dropout 0.15 \
    --lr 0.00008 \
    --weight_decay 0.0005 \
    --val_tta_mode none \
    "$@" 2>&1 | tee "${OUT_DIR}/${name}.log"
}

run_aug aug_research_baseline_20260430 \
  --aug_policy knee_mri_plus \
  --aug_cutout_frac 0.18 \
  --aug_noise_std 0.04 \
  --aug_slice_dropout 0.03 \
  --aug_gamma_jitter 0.12 \
  --aug_spatial_shift_frac 0.03

run_aug aug_research_bias_blur_20260430 \
  --aug_policy knee_mri_research \
  --aug_cutout_frac 0.18 \
  --aug_noise_std 0.04 \
  --aug_slice_dropout 0.03 \
  --aug_gamma_jitter 0.12 \
  --aug_spatial_shift_frac 0.03 \
  --aug_bias_field_std 0.10 \
  --aug_blur_sigma 0.65 \
  --aug_motion_prob 0.0

run_aug aug_research_bias_blur_motion_20260430 \
  --aug_policy knee_mri_research \
  --aug_cutout_frac 0.14 \
  --aug_noise_std 0.035 \
  --aug_slice_dropout 0.02 \
  --aug_gamma_jitter 0.10 \
  --aug_spatial_shift_frac 0.025 \
  --aug_bias_field_std 0.12 \
  --aug_blur_sigma 0.75 \
  --aug_motion_prob 0.15

run_aug aug_research_light_motion_20260430 \
  --aug_policy knee_mri_research \
  --aug_cutout_frac 0.10 \
  --aug_noise_std 0.03 \
  --aug_slice_dropout 0.01 \
  --aug_gamma_jitter 0.08 \
  --aug_spatial_shift_frac 0.02 \
  --aug_bias_field_std 0.08 \
  --aug_blur_sigma 0.45 \
  --aug_motion_prob 0.10
