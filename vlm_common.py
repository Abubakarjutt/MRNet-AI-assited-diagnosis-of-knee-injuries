"""MedGemma-4B LoRA classifier — pure, model-independent helpers.

This module is deliberately free of any transformers / peft import so it can be imported
and unit-tested on the fast suite (``pytest -m "not slow"``) with no model download and no
MPS device. It provides everything ``vlm_dataset`` and ``vlm_finetune`` need *except* the
actual model + processor (which come from ``transformers`` / ``peft`` at real-model runtime).

Spec: docs/superpowers/specs/2026-09-01-medgemma-lora-design.md
"""

import numpy as np
import torch
from sklearn import metrics

from dataloader import TASKS  # ("abnormal", "acl", "meniscus") — single source of label order

# The radiology prompt is fixed and versioned here; both the training conversation and the
# scoring readout use exactly these strings so the token positions line up.
SYSTEM = "You are a musculoskeletal radiologist analyzing a knee MRI exam."
INSTRUCTION = (
    "Sagittal, coronal, and axial series are shown as slice montages. "
    "Reply with only a JSON object with integer keys abnormal, acl, meniscus and "
    "values 0 or 1. abnormal = any abnormality present; acl = ACL tear; "
    "meniscus = meniscal tear."
)

# Deterministic seed for the "center" slice strategy. Kept as a module constant so the
# sampling is reproducible across a run; override via the `seed=` keyword if desired.
_CENTER_SEED = 20260901


# --------------------------------------------------------------------------- #
# Metric — VERBATIM copy of train.py:144-150 (do not import train.py: it runs     #
# module-level side effects). Returns 0.5 on single-class / degenerate input so   #
# smoke runs with --max_val_batches stay well-defined.                            #
# --------------------------------------------------------------------------- #
def compute_auc(y_true, y_pred):
    try:
        if len(np.unique(y_true)) > 1:
            return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        pass
    return 0.5


# --------------------------------------------------------------------------- #
# Slice sampling — which slices per plane become montage cells.                  #
# CONTRACT (spec §3.2): k sorted slice indices in [0, num_slices).              #
#   * num_slices >= k: pick k per `strategy`                                    #
#       "uniform": rounded linspace(0, n-1, k)                                  #
#       "center" : k points from a truncated normal centered at (n-1)/2         #
#   * num_slices <  k: rounded linspace(0, n-1, k) — indices repeat (cells       #
#       duplicate; documented, not an error).                                   #
# This is a *user contribution* (spec §11). The bodies below are the documented  #
# DEFAULT; they are overridable.                                                #
# --------------------------------------------------------------------------- #
def sample_slice_indices(num_slices, k, strategy="uniform", seed=_CENTER_SEED):
    n = int(num_slices)
    k = int(k)
    if n <= 0:
        raise ValueError("num_slices must be positive")
    if k <= 0:
        return []

    if n < k:
        # Not enough slices: repeat via linspace; montage cells simply duplicate.
        return sorted(int(v) for v in np.linspace(0, n - 1, k).round())

    if strategy == "uniform":
        idx = np.linspace(0, n - 1, k).round()
        return sorted(int(v) for v in idx)

    if strategy == "center":
        mu = (n - 1) / 2.0
        sigma = max(n / 6.0, 1e-6)
        rng = np.random.default_rng(seed)
        idx = rng.normal(loc=mu, scale=sigma, size=k)
        idx = np.clip(idx.round(), 0, n - 1).astype(int)
        return sorted(int(v) for v in idx)

    raise ValueError(f"unknown slice strategy: {strategy!r}")


# --------------------------------------------------------------------------- #
# Montage build — per-slice min-max -> uint8, resize to `cell` (bicubic),       #
# place into a rows*cols grid, grayscale -> RGB.                                #
# PIL is imported lazily so `import vlm_common` works even without Pillow.       #
# --------------------------------------------------------------------------- #
def build_montage(volume, indices, grid=(3, 2), cell=448):
    from PIL import Image

    rows, cols = grid
    needed = rows * cols
    if len(indices) != needed:
        raise ValueError(
             f"grid {rows}x{cols} needs {needed} slice indices, got {len(indices)}"
        )

    volume = volume if torch.is_tensor(volume) else torch.as_tensor(volume)
    volume = volume.float()

    resample = getattr(Image, "Resampling", Image).BICUBIC
    frames = []
    for idx in indices:
        frame = volume[int(idx)]
        lo, hi = frame.min(), frame.max()
        if (hi - lo) > 0:
            norm = (frame - lo) / (hi - lo)
        else:
            norm = torch.zeros_like(frame)
        u8 = (norm * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        img = Image.fromarray(u8, mode="L").resize((cell, cell), resample)
        frames.append(img)

    canvas = Image.new("RGB", (cell * cols, cell * rows))
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * cell, r * cell))
    return canvas


# --------------------------------------------------------------------------- #
# Conversation + answer formatting                                             #
# --------------------------------------------------------------------------- #
def format_answer(labels):
    """Exactly: {"abnormal": %d, "acl": %d, "meniscus": %d} — fixed key order (== TASKS),
    one space after each colon, values coerced to {0, 1}."""
    vals = [int(bool(v)) for v in labels]
    if len(vals) != 3:
        raise ValueError(f"expected 3 labels (TASKS order), got {len(vals)}")
    return '{"abnormal": %d, "acl": %d, "meniscus": %d}' % (vals[0], vals[1], vals[2])


def build_conversation(images, answer=None):
    """Chat-template message list. Image order is sagittal, coronal, axial (caller's order).
    The assistant turn is appended only when `answer` is provided (teacher-forced training /
    scoring); at eval-time generation the caller omits it and relies on add_generation_prompt."""
    user = [{"type": "image", "image": img} for img in images]
    user.append({"type": "text", "text": INSTRUCTION})
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def mask_prompt_tokens(input_ids, assistant_start):
    """labels = input_ids.clone(); labels[:assistant_start] = -100; return labels.
    Everything before the assistant turn is masked so the CE only scores the answer tokens."""
    labels = input_ids.clone()
    labels[:assistant_start] = -100
    return labels


# --------------------------------------------------------------------------- #
# Deterministic, generation-free scoring (the load-bearing part of the spec).   #
# We feed the model the answer as all-zeros and read, at the token position     #
# just before each value slot, P("1" | context) via a two-way softmax.          #
# --------------------------------------------------------------------------- #
def _find_last_subsequence(haystack, needle):
    """Index of the last occurrence of `needle` in `haystack` (lists of ints), or None."""
    m = len(needle)
    if m == 0:
        return 0
    for i in range(len(haystack) - m, -1, -1):
        if haystack[i:i + m] == needle:
            return i
    return None


def digit_token_ids(processor):
    """Return (zero_id, one_id) as they render *in the answer context*, derived not assumed.

    Tokenize format_answer((0,0,0)) and format_answer((1,1,1)); the positions that differ are
    exactly the three value slots. Assert all three zero-slots share one id and all three
    one-slots share another, then return that pair. (Gemma splits digits into single tokens,
    so this is stable, but we derive it rather than call encode('0').)
    """
    tok = processor.tokenizer
    zeros = tok(format_answer((0, 0, 0)), add_special_tokens=False)["input_ids"]
    ones = tok(format_answer((1, 1, 1)), add_special_tokens=False)["input_ids"]
    if len(zeros) != len(ones):
        raise ValueError("0-answer and 1-answer tokenizations differ in length")

    zero_ids, one_ids = set(), set()
    for a, b in zip(zeros, ones):
        if a != b:
            zero_ids.add(a)
            one_ids.add(b)
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(
            f"ambiguous digit tokens: zero_ids={sorted(zero_ids)} one_ids={sorted(one_ids)}"
        )
    zero_id, one_id = zero_ids.pop(), one_ids.pop()
    if zero_id == one_id:
        raise ValueError("zero_id and one_id must differ")
    return zero_id, one_id


def locate_answer_slots(processor, input_ids):
    """Absolute positions of the three value (digit) tokens in the PROCESSOR-EXPANDED
    `input_ids` (image placeholders already expanded). Method: tokenize
    format_answer((0,0,0)) alone, find it as the tail subsequence of input_ids, and return the
    positions whose token equals the zero digit id. Assert each decodes (stripped) to "0".
    """
    tok = processor.tokenizer
    zero_id, _one_id = digit_token_ids(processor)
    answer_ids = tok(format_answer((0, 0, 0)), add_special_tokens=False)["input_ids"]

    flat = input_ids.reshape(-1).tolist() if torch.is_tensor(input_ids) else list(input_ids)
    start = _find_last_subsequence(flat, answer_ids)
    if start is None:
        raise ValueError("answer subsequence not found in input_ids")

    slots = []
    for j, tid in enumerate(answer_ids):
        if tid == zero_id:
            abs_pos = start + j
            if tok.decode([tid]).strip() != "0":
                raise ValueError(f"slot token {tid} decoded to {tok.decode([tid])!r}, expected '0'")
            slots.append(abs_pos)

    if len(slots) != 3:
        raise ValueError(f"expected 3 digit slots, found {len(slots)}")
    if any(slots[i] >= slots[i + 1] for i in range(len(slots) - 1)):
        raise ValueError(f"slots not strictly increasing: {slots}")
    if any(s >= len(flat) for s in slots):
        raise ValueError(f"slot index out of bounds: {slots} vs len={len(flat)}")
    return slots


def score_exam(model, processor, images):
    """One teacher-forced forward per exam -> np.ndarray[3] of P(label==1) in TASKS order.

    1. conv = build_conversation(images, answer=format_answer((0,0,0)))
    2. batch = apply_chat_template(conv, add_generation_prompt=False, tokenize=True,
           return_dict=True, return_tensors="pt", do_pan_and_scan=False).to(device)
    3. slots = locate_answer_slots(...); zero_id, one_id = digit_token_ids(...)
    4. logits = model(**batch).logits[0]                          # [T, V]
    5. for each pos in slots: pair = logits[pos-1, [zero_id, one_id]].float();
         p1 = softmax(pair, -1)[1]; collect -> np.array([p1_abnormal, p1_acl, p1_meniscus])

    Step 5 is a *user contribution* (spec §11). DEFAULT = two-way renormalized softmax over
    {zero_id, one_id} — the standard MCQ-eval readout and it gives well-behaved AUC. Alternatives
    to weigh: full-vocab softmax then read P(one_id); a temperature. The two-way form is used here.
    """
    conv = build_conversation(images, answer=format_answer((0, 0, 0)))
    batch = processor.apply_chat_template(
        conv,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        do_pan_and_scan=False,
    )
    model_dtype = getattr(model, "dtype", None)

    def _to_model(value):
        if not torch.is_tensor(value):
            return value
        if model_dtype is not None and value.is_floating_point():
            return value.to(device=model.device, dtype=model_dtype)
        return value.to(device=model.device)

    batch = {k: _to_model(v) for k, v in batch.items()}

    input_ids = batch["input_ids"]
    slots = locate_answer_slots(processor, input_ids[0])
    zero_id, one_id = digit_token_ids(processor)

    with torch.no_grad():
        logits = model(**batch).logits[0]  # [T, V]
        probs = []
        for pos in slots:
            pair = logits[pos - 1, [zero_id, one_id]].float()
            p1 = torch.softmax(pair, dim=-1)[1].item()
            probs.append(p1)

    return np.array([probs[0], probs[1], probs[2]], dtype=np.float64)
