#!/bin/zsh
# Chained one-epoch MedGemma-4B LoRA fine-tune.
#
# A long multi-epoch MedGemma-4B run on this Mac gets memory-killed (exit 137)
# once accumulated MPS cache + swap outgrows headroom. So instead we invoke
# vlm_finetune.py once per epoch: each call warm-starts from the best adapter on
# disk (--resume_adapter), trains a single full epoch, evaluates, and saves only
# if it beat that checkpoint. A kill costs <=1 epoch; the next iteration resumes
# from whatever the last one managed to save.
#
# LR note: each 1-epoch process runs its cosine schedule from 5e-5 down to ~0 over
# that epoch's ~142 steps, then the next iteration resets to 5e-5 -- a warm-restart
# (SGDR-style) schedule across the chain.
#
# Epochs: each iteration is one FULL pass over all 1130 training exams (no
# --max_train_batches cap). The earlier cap existed to dodge OOM kills that were
# actually caused by a full disk (since fixed); a true full pass gives a much
# stronger per-epoch gradient than a random 35% slice.
#
# Seed: run() re-seeds to --seed at the top of every process, so a fixed seed makes
# every iteration shuffle identically -- two iters resuming the same checkpoint
# then produce bit-identical output and the chain stalls. --seed=(SEED_BASE + i)
# gives each iteration a different shuffle order (and dropout draw) so a resumed
# epoch is genuinely new work.
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
        --seed "$seed" --max_val_batches 120 \
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
