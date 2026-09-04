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
# Seed: run() re-seeds to --seed at the top of every process, so a fixed seed makes
# the shuffle=True train loader draw the SAME --max_train_batches exams every
# iteration -- two iters resuming the same checkpoint then produce bit-identical
# output and the chain never progresses. We pass --seed=(SEED_BASE + i) so each
# iteration sees a genuinely different slice; ~8 iters of 400/1130 covers ~97% of
# the training set.
#
# Val: evaluated on the full 120-exam val set (not the unshuffled first-90 slice,
# which is an easier sample and inflated "best" by ~0.05), so the saved best
# reflects the true pooled AUC.
#
# Usage:  scripts/chain_finetune.sh [ITERS] [PREFIX] [DATA_ROOT] [SEED_BASE]
set -u

ITERS=${1:-8}
PREFIX=${2:-medgemma_full}
DATA=${3:-/Users/Apple/projects/MRNet/MRNet-v1.0}
SEED_BASE=${4:-2000}
LOG=/tmp/${PREFIX}_chain.log

echo "=== chain start $(date) | iters=$ITERS prefix=$PREFIX ===" | tee -a "$LOG"

for i in $(seq 1 "$ITERS"); do
    # highest-AUC adapter dir for this prefix, by the valauc_<float> tag in its name
    latest=$(ls -d models/${PREFIX}_*_valauc_* 2>/dev/null \
             | awk -F'valauc_' '{print $2, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    resume=()
    [[ -n "$latest" ]] && resume=(--resume_adapter "$latest")

    seed=$((SEED_BASE + i))
    echo "=== iter $i/$ITERS $(date) | seed=$seed | resume='${latest:-<none, fresh init>}' ===" | tee -a "$LOG"

    python -u vlm_finetune.py \
        --prefix_name "$PREFIX" --data_root "$DATA" \
        --epochs 1 --grad_accum 8 --lr 5e-5 --lora_dropout 0.1 \
        --seed "$seed" --max_train_batches 400 --max_val_batches 120 \
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
