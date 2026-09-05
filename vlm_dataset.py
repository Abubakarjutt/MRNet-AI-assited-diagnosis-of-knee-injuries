"""MRNet exam -> MedGemma-4B LoRA dataset + collate (spec section 4).

`MRVLMDataset` wraps the existing `MRMultiPlaneDataset` (no augmentation) and turns each
exam's three plane volumes into three slice-montage images. `collate_fn` runs the chat
processor for a batch of exactly one exam (the whole repo runs batch_size == 1) and, for
training, builds teacher-forced labels by masking the prompt up to the assistant turn.

The `assistant_start` offset is derived by re-encoding the *prompt-only* conversation with
`add_generation_prompt=True` and taking its length -- the first token of the assistant's
answer content. (Spec section 11 flags this as an open item; the real Gemma-3 path may
instead use the template's `return_assistant_tokens_mask`, but the offset method is
self-contained and testable.)
"""

import torch
from torch.utils.data import Dataset

from dataloader import MRMultiPlaneDataset, MRVolumeAugmentor, TASKS
from vlm_common import (
    build_conversation,
    build_montage,
    format_answer,
    mask_prompt_tokens,
    sample_slice_indices,
)


class MRVLMDataset(Dataset):
    """Wraps ``MRMultiPlaneDataset`` and turns each exam into three slice-montage images.

    Training augmentation (``train=True`` and ``augment=True``, the defaults):
      * a study-consistent ``MRVolumeAugmentor`` runs on the raw volumes before montage
        assembly — one sampled plan applied identically to all three planes (flip,
        intensity, and — for ``knee_mri_plus`` — gamma and small spatial shift). The
        per-frame min-max normalisation inside ``build_montage`` re-levels global
        intensity, so flip / spatial-shift / gamma / slice-jitter carry the signal.
      * ``slice_jitter`` perturbs the montage slice indices per read (see
        ``vlm_common.sample_slice_indices``), so each epoch sees a slightly different
        montage.
    Eval (``train=False``) is always unaugmented and deterministic regardless of
    ``augment`` / ``slice_jitter``.
    """

    def __init__(self, root_dir, train, k=6, grid=(3, 2), cell=448,
                 slice_strategy="uniform", mmap=True, cache_size=32,
                 augment=True, aug_policy="knee_mri_plus",
                 aug_gamma_jitter=0.15, aug_spatial_shift_frac=0.05, slice_jitter=2):
        self.train = bool(train)
        self.augment = bool(augment) and self.train
        transform = None
        if self.augment:
            transform = MRVolumeAugmentor(
                policy=aug_policy,
                gamma_jitter=aug_gamma_jitter,
                spatial_shift_frac=aug_spatial_shift_frac,
            )
        self.base = MRMultiPlaneDataset(
            root_dir=root_dir,
            train=train,
            mmap=mmap,
            cache_size=cache_size,
            transform=transform,
        )
        self.k = int(k)
        self.grid = tuple(grid)
        self.cell = int(cell)
        self.slice_strategy = slice_strategy
        self.slice_jitter = int(slice_jitter)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        volumes, label, _weights, exam_id = self.base[i]     # 3 x [s, 256, 256]
        jitter = self.slice_jitter if (self.train and self.slice_jitter > 0) else 0
        images = [
            build_montage(
                volume,
                sample_slice_indices(
                    volume.shape[0], self.k, self.slice_strategy, jitter=jitter,
                ),
                self.grid,
                self.cell,
            )
            for volume in volumes
        ]
        return {
            "images": images,             # sagittal, coronal, axial (PLANES order)
            "label": label,              # tensor[3], order == TASKS
            "exam_id": exam_id,
        }


def _assistant_start(processor, conversation, do_pan_and_scan=False):
    """First token index of the assistant's answer content.

    Re-encode the prompt-only conversation (assistant turn removed) with
    add_generation_prompt=True; its length is the offset at which the answer begins."""
    prompt = [m for m in conversation if m["role"] != "assistant"]
    enc = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=do_pan_and_scan,
    )
    return int(enc["input_ids"].shape[1])


def collate_fn(batch, processor, train: bool):
    """Collate a single-exam batch (len == 1, enforced) through the chat processor.

    - train=True   -> includes the assistant answer, add_generation_prompt=False, and builds
      `labels` (prompt masked to -100, real ids from the assistant turn on).
    - train=False -> omits the assistant answer, add_generation_prompt=True (generation mode);
      no `labels` key.
    Always attaches the raw `label` tensor and `exam_id`.

    Note: on the eval path the scoring readout (vlm_common.score_exam) re-runs the chat
    template from item["images"] to build its own all-zeros teacher-forced conversation,
    so the train=False input_ids / pixel_values here are not consumed by AUC scoring.
    They are kept for parity, debugging, and a future generation-based eval."""
    if len(batch) != 1:
        raise ValueError(f"collate_fn expects batch_size 1, got {len(batch)}")
    item = batch[0]

    answer = format_answer(item["label"].int().tolist()) if train else None
    conv = build_conversation(item["images"], answer=answer)

    enc = processor.apply_chat_template(
        conv,
        add_generation_prompt=not train,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=False,
    )

    out = {k: v for k, v in enc.items() if k not in ("assistant_start",)}
    if train:
        assistant_start = _assistant_start(processor, conv, do_pan_and_scan=False)
        out["labels"] = mask_prompt_tokens(enc["input_ids"][0], assistant_start).unsqueeze(0)
        if out["labels"].shape != out["input_ids"].shape:
            raise ValueError("labels/input_ids shape mismatch after masking")

    out["label"] = item["label"]
    out["exam_id"] = item["exam_id"]
    # score_exam / _write_predictions read batch["images"] (the raw montage list used
    # to build the teacher-forced readout conversation); carry it through both branches.
    out["images"] = item["images"]
    return out
