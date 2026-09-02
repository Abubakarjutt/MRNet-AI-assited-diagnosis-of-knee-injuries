"""MedGemma-4B LoRA SFT for MRNet (spec section 5).

Standalone LoRA fine-tune of google/medgemma-4b-it that classifies a knee MRI exam into the
three MRNet labels, scored with a deterministic teacher-forced token-probability readout so its
pooled micro-AUC is constructed identically to train.py's best_val_auc.

Heavy deps (peft / transformers / accelerate) are imported behind a guard so this module imports
without them; the real-model paths assert their presence. The fast suite exercises the pure logic
(vlm_common / vlm_dataset) and skips the model-dependent paths when peft is unavailable.
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.optim as optim

from dataloader import TASKS
from vlm_common import compute_auc, score_exam
from vlm_dataset import MRVLMDataset, collate_fn
import utils

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers import get_cosine_schedule_with_warmup
    _HAVE_DEPS = True
except Exception:
    _HAVE_DEPS = False


# Copied verbatim from train.py:50-54 (do NOT import train.py: it runs module-level
# side effects). Synchronizes MPS/CUDA after each epoch for stable timing.
def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# Anchored on `language_model.` but tolerant of the `model.` wrapper that
# Gemma3ForConditionalGeneration adds (real names: model.language_model.layers.N...).
# PEFT uses re.fullmatch, so the leading `(?:.*\.)?` is load-bearing. Still leak-proof:
# the SigLIP tower and multi_modal_projector have no `language_model.` segment.
LORA_TARGET_MODULES = (
    r"(?:.*\.)?language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
_VISION_FROZEN_SUBSTRINGS = ("vision_tower", "multi_modal_projector")
_TRAINABLE_TRIPWIRE_M = 60.0        # LoRA leak guard (spec section 9.3)

processor_ref = {"processor": None}    # filled by run() so build_loaders can close over it


# collate_fn attaches these bookkeeping keys next to the processor's real outputs
# (input_ids / attention_mask / pixel_values / labels / token_type_ids / ...). They must
# not reach model.forward — the real Gemma3 signature rejects unknown kwargs. A denylist
# (not an allowlist) keeps any *valid* future processor output flowing through untouched.
_COLLATE_EXTRA_KEYS = frozenset({"label", "exam_id", "images"})


def _forward_kwargs(batch):
    return {k: v for k, v in batch.items() if k not in _COLLATE_EXTRA_KEYS}


def load_base(base_model, device):
    """Load the processor + bf16 base model (eager attention, MPS device map)."""
    if not _HAVE_DEPS:
        raise RuntimeError("transformers/peft not available; cannot load the base model")
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,               # `dtype=`, not the deprecated `torch_dtype=`
        attn_implementation="eager",        # Gemma-3 sliding-window stability, esp. on MPS
        device_map={"": device.type},       # MPS only (no bitsandbytes 4-bit)
    )
    return processor, model


def apply_lora(model, args):
    """Freeze the base and attach LoRA adapters to the language model only."""
    if not _HAVE_DEPS:
        raise RuntimeError("peft not available; cannot apply LoRA")
    for p in model.parameters():
        p.requires_grad_(False)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    return get_peft_model(model, peft_config)


def assert_freeze(model):
    """Vision tower + projector must stay frozen; trainable size below the leak tripwire."""
    for name, p in model.named_parameters():
        if any(sub in name for sub in _VISION_FROZEN_SUBSTRINGS):
            assert not p.requires_grad, f"{name} should be frozen but requires grad"
    trainable_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    assert trainable_M < _TRAINABLE_TRIPWIRE_M, (
        f"trainable {trainable_M:.2f}M exceeds the {_TRAINABLE_TRIPWIRE_M}M leak tripwire"
    )
    return trainable_M


def build_model(args, device):
    """Full section 5.2 pipeline: load base, apply LoRA, verify the freeze."""
    processor, model = load_base(args.base_model, device)
    model = apply_lora(model, args)
    trainable_M = assert_freeze(model)
    return processor, model, trainable_M


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loaders(args):
    from functools import partial

    def make(train):
        ds = MRVLMDataset(
            args.data_root, train=train, k=args.slices_per_plane,
            grid=(3, 2), cell=448, slice_strategy=args.slice_strategy,
            mmap=bool(args.mmap), cache_size=args.cache_size,
        )
        collate = partial(collate_fn, processor=processor_ref["processor"], train=train)
        return torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=train, num_workers=0, collate_fn=collate
        )

    return make(True), make(False)


def _write_predictions(path, exam_ids, y_pred, y_true):
    """TSV: exam_id, p_abnormal, p_acl, p_meniscus, y_abnormal, y_acl, y_meniscus.
    Rows come straight from evaluate()'s single scoring pass — no second forward."""
    header = "exam_id\t" + "\t".join(TASKS) + "\t" + "\t".join(TASKS) + "\n"
    with open(path, "w") as handle:
        handle.write(header)
        for exam_id, prow, yrow in zip(exam_ids, y_pred, y_true):
            handle.write(
                f"{exam_id}\t"
                + "\t".join(f"{float(p):.6f}" for p in prow) + "\t"
                + "\t".join(f"{int(y)}" for y in yrow) + "\n"
            )


def evaluate(model, processor, val_loader, device, max_val_batches=None,
             dump_predictions=None):
    """Per-exam teacher-forced scoring -> pooled micro-AUC + per-task AUC + mean val CE.

    Two forwards per exam: (1) the generation-free readout (score_exam) that yields the
    AUC rows, and (2) a teacher-forced masked forward (train-style collation) whose .loss
    is averaged into best_val_loss (spec §5.4)."""
    model.eval()
    y_true_rows, y_pred_rows, exam_ids = [], [], []
    ce_sum, ce_n = 0.0, 0
    for batch_index, batch in enumerate(val_loader):
        if max_val_batches is not None and batch_index >= max_val_batches:
            break
        # score_exam re-templates from images (all-zeros teacher-forced readout); the
        # val loader's collated input_ids/pixel_values are intentionally unused here.
        probs = score_exam(model, processor, batch["images"])          # np.ndarray[3]
        y_true_rows.append(batch["label"].int().tolist())
        y_pred_rows.append(list(probs))
        exam_ids.append(batch["exam_id"])

        tf_item = {"images": batch["images"], "label": batch["label"],
                   "exam_id": batch["exam_id"]}
        tf_batch = _batch_to_model(collate_fn([tf_item], processor, train=True), model)
        with torch.no_grad():
            ce = model(**_forward_kwargs(tf_batch)).loss
        if ce is not None:
            ce_sum += float(ce)
            ce_n += 1

    y_true = np.array(y_true_rows, dtype=np.int64)
    y_pred = np.array(y_pred_rows, dtype=np.float64)
    pooled_auc = compute_auc(y_true.reshape(-1).tolist(), y_pred.reshape(-1).tolist())
    per_task = [
        float(compute_auc(y_true[:, i].tolist(), y_pred[:, i].tolist()))
        for i in range(len(TASKS))
    ]
    val_ce = (ce_sum / ce_n) if ce_n else None
    if dump_predictions:
        _write_predictions(dump_predictions, exam_ids, y_pred, y_true)
    return float(pooled_auc), per_task, val_ce, y_pred, y_true


def _batch_to_model(batch, model):
    """Move collated tensors onto the model's device and cast floating tensors
    (pixel_values) to its compute dtype. Int tensors (input_ids/attention_mask/labels)
    and non-tensor entries (label/exam_id/images) pass through untouched."""
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    dtype = getattr(model, "dtype", None)
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif dtype is not None and value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def train_one_epoch(model, optimizer, scheduler, train_loader, device, args,
                    global_step, start_time):
    """One epoch. Optimizer steps every grad_accum samples; honors batch + time budgets."""
    model.train()
    running_loss = 0.0
    n_seen = 0
    accumulator = 0
    for batch_index, batch in enumerate(train_loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        if (args.time_budget_minutes is not None
                and (time.time() - start_time) / 60.0 >= args.time_budget_minutes):
            print(f"Stopping training early after reaching the "
                  f"{args.time_budget_minutes:.2f} minute budget.")
            break
        batch = _batch_to_model(batch, model)
        loss = model(**_forward_kwargs(batch)).loss / float(args.grad_accum)
        loss.backward()
        running_loss += loss.item()
        n_seen += 1
        accumulator += 1
        if accumulator >= args.grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulator = 0
            global_step += 1
    if accumulator > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
    return running_loss / max(n_seen, 1), global_step


def _adapter_dir(prefix, pooled_auc):
    return os.path.join("models", f"{prefix}_medgemma_lora_valauc_{pooled_auc:.4f}")


def _prune_prior_adapter_dirs(prefix):
    """One-best-per-run: delete prior adapter dirs whose name contains prefix."""
    models_dir = os.path.abspath("models")
    if not os.path.isdir(models_dir):
        return
    for name in os.listdir(models_dir):
        if prefix in name and os.path.isdir(os.path.join(models_dir, name)):
            shutil.rmtree(os.path.join(models_dir, name), ignore_errors=True)


def _rebuild_with_adapter(args, device):
    """For --eval_only: load base + adapter via PeftModel.from_pretrained."""
    processor, model = load_base(args.base_model, device)
    model = PeftModel.from_pretrained(model, args.eval_only)
    trainable_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return processor, model, trainable_M


def _print_metrics(args, device, best_val_auc, best_val_loss, per_task,
                   training_seconds, epochs_ran, best_epoch, full_M, trainable_M):
    per_task_str = " ".join(f"{t}={a:.4f}" for t, a in zip(TASKS, per_task))
    print("---")
    print(f"best_val_auc:         {best_val_auc:.6f}")
    print(f"best_val_loss:        {best_val_loss:.6f}")
    print(f"per_task_auc:         {per_task_str}")
    print(f"training_seconds:     {training_seconds:.2f}")
    print(f"epochs_ran:           {epochs_ran}")
    print(f"best_epoch:           {best_epoch}")
    print(f"num_params_M:         {full_M:.2f}")
    print(f"trainable_params_M:   {trainable_M:.2f}")
    print(f"model_type:          medgemma-4b-lora")
    print(f"device:               {device.type}")


def run(args):
    if args.batch_size != 1:
        raise ValueError("this pipeline is built for batch_size == 1")

    torch.manual_seed(args.seed)
    device = utils.get_device()

    if args.eval_only is not None:
        processor, model, trainable_M = _rebuild_with_adapter(args, device)
        processor_ref["processor"] = processor          # BEFORE build_loaders
        full_M = sum(p.numel() for p in model.parameters()) / 1e6
        _, val_loader = build_loaders(args)
        pooled_auc, per_task, val_ce, _, _ = evaluate(
            model, processor, val_loader, device,
            max_val_batches=args.max_val_batches,
            dump_predictions=args.dump_predictions,
        )
        print(f"--- eval_only adapter: {args.eval_only}")
        _print_metrics(args, device, pooled_auc,
                       (val_ce if val_ce is not None else float("nan")),
                       per_task, 0.0, 0, -1, full_M, trainable_M)
        return float(pooled_auc)

    processor, model, trainable_M = build_model(args, device)
    processor_ref["processor"] = processor
    full_M = sum(p.numel() for p in model.parameters()) / 1e6
    train_loader, val_loader = build_loaders(args)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    total_steps = int(np.ceil(len(train_loader) / args.grad_accum)) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
        num_training_steps=max(1, total_steps),
    )

    best_val_auc = 0.0
    best_per_task = [0.5] * len(TASKS)
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    epochs_ran = 0
    global_step = 0
    t_start_training = time.time()

    for epoch in range(args.epochs):
        if (args.time_budget_minutes is not None
                and (time.time() - t_start_training) / 60.0 >= args.time_budget_minutes):
            print(f"Stopping early after reaching the "
                  f"{args.time_budget_minutes:.2f} minute budget.")
            break

        train_loss, global_step = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, args,
            global_step, t_start_training,
        )
        maybe_sync(device)

        pooled_auc, per_task, val_ce, _, _ = evaluate(
            model, processor, val_loader, device, max_val_batches=args.max_val_batches,
        )
        epochs_ran = epoch + 1
        print(
            f"epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.4f} "
            f"| val auc: {pooled_auc:.4f} | per-task: "
            + " ".join(f"{t}={a:.4f}" for t, a in zip(TASKS, per_task))
        )

        if pooled_auc > best_val_auc:
            best_val_auc = float(pooled_auc)
            best_epoch = epoch + 1
            best_per_task = per_task
            _prune_prior_adapter_dirs(args.prefix_name)
            out_dir = _adapter_dir(args.prefix_name, best_val_auc)
            os.makedirs(out_dir, exist_ok=True)
            model.save_pretrained(out_dir)
            print(f"Best adapter saved (val AUC {best_val_auc:.4f}) -> {out_dir}")
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping after {patience_counter} epochs "
                      f"without val-AUC improvement.")
                break

        if val_ce is not None and val_ce < best_val_loss:
            best_val_loss = float(val_ce)

    training_seconds = time.time() - t_start_training
    _print_metrics(args, device, best_val_auc, best_val_loss, best_per_task,
                   training_seconds, epochs_ran, best_epoch, full_M, trainable_M)
    return best_val_auc


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="MRNet-v1.0")
    parser.add_argument("--base_model", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--slices_per_plane", type=int, default=6)
    parser.add_argument("--slice_strategy", type=str, default="uniform",
                        choices=["uniform", "center"])
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--time_budget_minutes", type=float, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--eval_only", type=str, default=None)
    parser.add_argument("--dump_predictions", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mmap", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cache_size", type=int, default=32)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
