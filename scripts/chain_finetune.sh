#!/bin/zsh
# Chained one-epoch MedGemma-4B LoRA fine-tune.
#
# This Mac cannot sustain more than ~1 epoch of MedGemma-4B training before macOS
# memory-kills the process (exit 137). So instead of one long multi-epoch run we
# invoke vlm_finetune.py once per epoch: each call warm-starts from the best
# adapter on disk (--resume_adapter), trains a single epoch, evaluates, and saves
# only if it beat that checkpoint. A kill costs <=1 epoch; the next iteration
# resumes from whatever the last one managed to save.
#
# LR note: each 1-epoch process restarts the cosine scheduler from step 0, so the
# effective schedule across the chain is ~constant lr with a tiny warmup each
# epoch -- a fine regime for LoRA SFT.
#
# Usage:  scripts/chain_finetune.sh [ITERS] [PREFIX] [DATA_ROOT]
set -u

ITERS=${1:-8}
PREFIX=${2:-medgemma_full}
DATA=${3:-/Users/Apple/projects/MRNet/MRNet-v1.0}
LOG=/tmp/${PREFIX}_chain.log

echo "=== chain start $(date) | iters=$ITERS prefix=$PREFIX ===" | tee -a "$LOG"

for i in $(seq 1 "$ITERS"); do
    # highest-AUC adapter dir for this prefix, by the valauc_<float> tag in its name
    latest=$(ls -d models/${PREFIX}_*_valauc_* 2>/dev/null \
             | awk -F'valauc_' '{print $2, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    resume=()
    [[ -n "$latest" ]] && resume=(--resume_adapter "$latest")

    echo "=== iter $i/$ITERS $(date) | resume='${latest:-<none, fresh init>}' ===" | tee -a "$LOG"

    python -u vlm_finetune.py \
        --prefix_name "$PREFIX" --data_root "$DATA" \
        --epochs 1 --grad_accum 8 --lr 5e-5 --lora_dropout 0.1 \
        --max_train_batches 400 --max_val_batches 90 \
        "${resume[@]}" >> "$LOG" 2>&1
    rc=$?
    echo "=== iter $i exit rc=$rc $(date) ===" | tee -a "$LOG"
    if [[ $rc -ne 0 && $rc -ne 137 ]]; then
        echo "=== non-OOM failure (rc=$rc); stopping chain ===" | tee -a "$LOG"
        exit $rc
    fi
done

final=$(ls -d models/${PREFIX}_*_valauc_* 2>/dev/null \
        | awk -F'valauc_' '{print $2, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
echo "=== chain done $(date) | best adapter: ${final:-<none>} ===" | tee -a "$LOG"
