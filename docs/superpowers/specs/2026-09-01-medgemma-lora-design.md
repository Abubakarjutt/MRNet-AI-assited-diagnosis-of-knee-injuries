# MedGemma-4B LoRA Classifier for MRNet — Design Spec

**Date:** 2026-09-01
**Status:** Approved for planning
**Scope:** A standalone LoRA fine-tune of `google/medgemma-4b-it` that classifies a knee MRI
exam into the three MRNet labels, scored with a deterministic teacher-forced token-probability
readout so its **pooled micro-AUC matches `train.py`'s `best_val_auc`** and its numbers sit in
the same table as the CNN baselines.

**Sibling spec:** `2026-09-01-medsiglip-encoder-design.md` (path 1, frozen MedSigLIP
encoder). Independent implementation, independent PR. Path 1 should land first; this spec
shares only the dataset, `dataloader.TASKS`, and `MEDICAL_MODELS.md`.

**Supersedes:** the path-2 half of `2026-09-01-medical-pretrained-models-design.md`, revised
after independent review (metric was mis-described; scoring methodology was under-specified;
`target_modules` leaked into the vision tower; `attn_implementation` and `do_pan_and_scan`
were unaddressed; the "tiny Gemma-3" smoke-test model does not exist).

---

## 1. Context

### 1.1 The task and the metric it must match (corrected)

MRNet: exam-level, 3 series (`sagittal`/`coronal`/`axial`, ~12–40 grayscale `256×256`
slices), 3 independent binary labels — `abnormal` (~80% positive), `acl`, `meniscus`.

**`train.py`'s `best_val_auc` is a single pooled micro-AUC** (verified in source):
`iterate_epoch` flattens each `[1,3]` prediction with `.reshape(-1)` and pools all three
per-task probabilities from all exams into two flat lists (`train.py:242-246`);
`compute_auc` (`train.py:122-128`) runs `sklearn.metrics.roc_auc_score` over those `3 × N`
points and returns **`0.5`** on a single-class input. The final block (`train.py:551-561`)
has **no per-task AUC line**.

**Therefore** this spec's primary reported number is the identical construction:
`compute_auc(y_true.reshape(-1), y_pred.reshape(-1))` over the `[N, 3]` prediction matrix →
printed as `best_val_auc`. Per-task AUCs are printed as a **supplementary**
`per_task_auc:` line, clearly not part of `train.py`'s block.

### 1.2 What we reuse from the repo

- `dataloader.MRMultiPlaneDataset` — split logic, label construction, `exam_id`, `mmap`,
  `cache_size`. Yields `(volumes, label, weights, exam_id)` with `volumes` a 3-tuple of
  `[slices, 256, 256]` float32 tensors.
- `dataloader.TASKS == ("abnormal", "acl", "meniscus")` — **imported**, the single source of
  label-column order (no separate `LABEL_ORDER` constant).
- `compute_auc` — **copied verbatim** (3 lines) into `vlm_common.py`, *not* imported.
  Importing `train.py` runs its module-level side effects
  (`torch.set_float32_matmul_precision`, the fallback `SummaryWriter` class,
  `import advanced_vit/lightweight_models/utils`, `from dataloader import ...`). The copy
  keeps the `0.5`-on-single-class behavior so smoke runs with `--max_val_batches` stay
  well-defined.
- `maybe_sync(device)` pattern (`train.py:50-54`) — copied (2 lines) for MPS epoch timing.

Nothing in `train.py`, `lightweight_models.py`, `dataloader.py`, or the autoresearch loop is
modified.

### 1.3 Why MedGemma-4B

`google/medgemma-4b-it` = Gemma-3 4B multimodal (`Gemma3ForConditionalGeneration`, loaded
via `AutoModelForImageTextToText`): a SigLIP vision tower + `multi_modal_projector` + a
Gemma-3 language model, instruction-tuned with medical-domain data. Not trained on knee MRI
→ fit LoRA adapters on the **language** layers only.

### 1.4 Constraints

Apple Silicon Mac, **MPS only** (no `bitsandbytes` 4-bit → bf16 weights + LoRA adapters),
**≥32 GB** unified memory (MedGemma-4B bf16 + adapters + activations fits, slowly),
`MRNet-v1.0/` local, HF gated access available. Objective fixed at brainstorming:
**AUC-comparable classifier**, not free-text generation.

---

## 2. Goals / Non-goals

### Goals

1. `vlm_common.py` — pure helpers: slice sampling, montage build, chat-message build, fixed
   answer formatting, prompt-token masking, **fully-specified** teacher-forced scoring,
   local `compute_auc`.
2. `vlm_dataset.py` — `MRVLMDataset` wrapping `MRMultiPlaneDataset` + a `collate_fn` that
   runs the processor.
3. `vlm_finetune.py` — LoRA SFT training loop, per-epoch validation via the scoring path,
   best-adapter checkpointing, `--eval_only`, a final metrics block whose `best_val_auc` is
   the pooled micro-AUC.
4. LoRA on the **language model only** (`language_model.`-anchored `target_modules`); vision
   tower + projector verifiably frozen.
5. bf16 + `attn_implementation="eager"` + `device_map={"": "mps"}`, `do_pan_and_scan=False`.
6. Deterministic, generation-free scoring (one teacher-forced forward per exam).
7. Tests that run on `pytest -m "not slow"` with no large download (a hand-built tiny
   `Gemma3` model + processor in `conftest.py`); real-weight paths marked `slow`.
8. A `MEDICAL_MODELS.md` section.

### Non-goals

- No free-text radiology reports (chosen objective is AUC-comparable classification). A
  `--generate` inspection mode is explicitly deferred (§9).
- No changes to `train.py` / `lightweight_models.py` / `dataloader.py` / autoresearch.
- No 4-bit/8-bit quantization, no MLX, no multi-GPU.
- Not added to the autoresearch search space.
- No image augmentation (montage rasterization + 4B LoRA — augmentation adds cost/noise for
  little expected gain; revisit later).

---

## 3. `vlm_common.py` — pure helpers

```python
import numpy as np, torch
from dataloader import TASKS            # ("abnormal", "acl", "meniscus")

SYSTEM = "You are a musculoskeletal radiologist analyzing a knee MRI exam."
INSTRUCTION = (
    "Sagittal, coronal, and axial series are shown as slice montages. "
    "Reply with only a JSON object with integer keys abnormal, acl, meniscus and "
    "values 0 or 1. abnormal = any abnormality present; acl = ACL tear; "
    "meniscus = meniscal tear."
)

def compute_auc(y_true, y_pred) -> float:
    # verbatim copy of train.py:122-128 — returns 0.5 on single-class / degenerate input
    ...

def sample_slice_indices(num_slices: int, k: int, strategy: str = "uniform") -> list[int]:
    """k sorted slice indices in [0, num_slices).
      - if num_slices >= k: pick k of them per `strategy`
          "uniform": np.linspace(0, num_slices-1, k) rounded
          "center" : k points from a truncated normal centered at num_slices/2
      - if num_slices < k: return np.linspace(0, num_slices-1, k).round() (indices repeat) —
          montage cells simply duplicate; documented, not an error.
    TODO(user, ~8 lines): implement the two strategies. This decides what the model sees.
    """

def build_montage(volume: torch.Tensor, indices: list[int],
                  grid: tuple[int, int] = (3, 2), cell: int = 448) -> "PIL.Image.Image":
    """Per-slice min-max -> uint8, resize to `cell` (bicubic), place into a rows*cols grid,
    grayscale -> RGB. Output size == (cell*cols, cell*rows). len(indices) must == rows*cols."""

def format_answer(labels) -> str:
    """Exactly:  {"abnormal": %d, "acl": %d, "meniscus": %d}
    fixed key order (== TASKS), one space after each colon, values in {0,1}."""

def build_conversation(images, answer: str | None):
    """Return the chat list:
        system: SYSTEM
        user:   [{"type":"image","image":img} x3] + {"type":"text","text": INSTRUCTION}
        assistant (only if answer is not None): {"type":"text","text": answer}
    Image order is sagittal, coronal, axial."""

def mask_prompt_tokens(input_ids: torch.Tensor, assistant_start: int) -> torch.Tensor:
    """labels = input_ids.clone(); labels[:assistant_start] = -100; return labels."""

# ---- deterministic scoring -------------------------------------------------------------

def locate_answer_slots(processor, input_ids: torch.Tensor) -> list[int]:
    """Find the 3 value-token absolute positions in the PROCESSOR-EXPANDED input_ids
    (image placeholders already expanded to hundreds of tokens).
    Method: tokenize format_answer((0,0,0)) alone; find that subsequence as the tail of
    input_ids (it is the last turn's content); return the absolute indices of the three
    tokens that correspond to the digit positions. Assert each decodes (stripped) to "0"."""

def digit_token_ids(processor) -> tuple[int, int]:
    """Return (zero_id, one_id) as they render IN THE ANSWER CONTEXT, derived — not assumed:
    tokenize format_answer((0,0,0)) and format_answer((1,1,1)); at each of the 3 slot
    positions the id differs; assert all 3 zero-slots share one id and all 3 one-slots share
    another; return that pair. (Gemma splits digits into single tokens, so this is stable,
    but we derive it rather than call encode('0').)"""

def score_exam(model, processor, images) -> np.ndarray:
    """1. conv = build_conversation(images, answer=format_answer((0,0,0)))
       2. batch = processor.apply_chat_template(conv, add_generation_prompt=False,
              tokenize=True, return_dict=True, return_tensors="pt",
              do_pan_and_scan=False).to(model.device)
       3. slots = locate_answer_slots(processor, batch["input_ids"][0])
          zero_id, one_id = digit_token_ids(processor)
       4. one teacher-forced forward: logits = model(**batch).logits[0]      # [T, V]
       5. for each pos in slots: pair = logits[pos-1, [zero_id, one_id]].float()
              p1 = torch.softmax(pair, dim=-1)[1].item()
          return np.array([p1_abnormal, p1_acl, p1_meniscus])
    TODO(user, ~6 lines): the step-5 logit -> probability mapping. Default = two-way softmax
    over {zero_id, one_id}. Alternatives to weigh: full-vocab softmax then read P(one_id);
    a temperature. The two-way renormalized softmax is the standard MCQ-eval choice and
    gives well-behaved AUC.
    """
```

Rationale for the scoring shape (addresses the review's "load-bearing parts were punted"):
- **Indices are located on the expanded `input_ids`**, after `apply_chat_template` inserts
  image tokens — not on a text-only tokenization — so the `pos-1` logit lookup is aligned.
- **`zero_id` / `one_id` are derived from the actual answer rendering** (diffing
  `(0,0,0)` vs `(1,1,1)` tokenizations), so the `▁0` vs `0` / leading-space ambiguity cannot
  silently invalidate the AUC. The `decode(...).strip() == "0"` assert is a second tripwire.
- One forward, no `generate()` → deterministic, ~120 val exams × 1 forward ≈ 1–3 min on MPS.

---

## 4. `vlm_dataset.py`

```python
class MRVLMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train, k=6, grid=(3, 2), cell=448,
                 slice_strategy="uniform", mmap=True, cache_size=32):
        self.base = MRMultiPlaneDataset(root_dir, train=train, mmap=mmap,
                                        cache_size=cache_size, transform=None)  # no aug
        ...

    def __getitem__(self, i):
        volumes, label, _w, exam_id = self.base[i]                 # 3 x [s,256,256]
        images = [build_montage(v, sample_slice_indices(v.shape[0], self.k,
                                self.slice_strategy), self.grid, self.cell)
                  for v in volumes]                                 # sag, cor, ax
        return {"images": images,
                "label": label,                                    # tensor[3], order == TASKS
                "exam_id": exam_id}
```

**`collate_fn(batch, processor, train: bool)`** — `batch` has length 1 (the whole repo runs
`batch_size == 1`; enforced):
- `conv = build_conversation(item["images"], answer=format_answer(item["label"].int().tolist()) if train else None)`
- `enc = processor.apply_chat_template(conv, add_generation_prompt=not train, tokenize=True,
  return_dict=True, return_tensors="pt", do_pan_and_scan=False)`
- if `train`: compute `assistant_start` (first token index of the assistant turn's content
  via the template's offsets / a marker search) and
  `enc["labels"] = mask_prompt_tokens(enc["input_ids"][0], assistant_start).unsqueeze(0)`
- always attach `label` tensor + `exam_id`.

Token budget with `do_pan_and_scan=False`: 3 montages × ~256 image tokens + ~150 text ≈
**~950 tokens/exam** (pan-and-scan would 3–5× this).

---

## 5. `vlm_finetune.py`

### 5.1 Args (style mirrors `train.py`)

| arg | default | notes |
|---|---|---|
| `--prefix_name` | (required) | run name → adapter dir + logs |
| `--data_root` | `MRNet-v1.0` | |
| `--base_model` | `google/medgemma-4b-it` | override for the stub-model tests |
| `--epochs` | 3 | |
| `--lr` | 1e-4 | adapter LR |
| `--batch_size` | 1 | enforced `== 1` |
| `--grad_accum` | 8 | optimizer step = every 8 samples |
| `--lora_r` / `--lora_alpha` / `--lora_dropout` | 16 / 32 / 0.05 | |
| `--slices_per_plane` | 6 | montage `k`; with `--grid 3x2` must equal rows*cols |
| `--slice_strategy` | `uniform` | `uniform` \| `center` |
| `--warmup_ratio` | 0.03 | cosine schedule over **optimizer steps** = `ceil(len(train)/grad_accum) * epochs` |
| `--patience` | 3 | epochs w/o **pooled val-AUC** improvement (see §5.4) |
| `--time_budget_minutes` | `None` | wall-clock cap (same semantics as `train.py`) |
| `--max_train_batches` / `--max_val_batches` | `None` | smoke-run caps |
| `--eval_only` | `None` | adapter dir; skip training, run §5.4, print block |
| `--dump_predictions` | `None` | TSV: `exam_id, p_abnormal, p_acl, p_meniscus, y_abnormal, y_acl, y_meniscus` |
| `--seed` | 1337 | |

### 5.2 Model load

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained(base_model)
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    dtype=torch.bfloat16,                 # `dtype=`, not deprecated `torch_dtype=`
    attn_implementation="eager",          # Gemma-3 sliding-window stability, esp. on MPS
    device_map={"": "mps"},
)
for p in model.parameters():
    p.requires_grad_(False)

peft_config = LoraConfig(
    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    bias="none", task_type="CAUSAL_LM",
    # anchor to the language model so SigLIP's q/k/v_proj are NOT matched
    target_modules=r"language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
)
model = get_peft_model(model, peft_config)

# hard assertions (also covered by tests)
assert all(not p.requires_grad for n, p in model.named_parameters()
           if "vision_tower" in n or "multi_modal_projector" in n)
trainable_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
# expected ~30-45M at r=16; printed, and asserted < 60M as a leak tripwire
```

### 5.3 Training loop

- `DataLoader(MRVLMDataset(...), batch_size=1, shuffle=True,
  collate_fn=partial(collate_fn, processor=processor, train=True))`.
- Step: `loss = model(**batch).loss` (teacher-forced CE on the masked assistant JSON) /
  `grad_accum`; `loss.backward()`; every `grad_accum` samples:
  `clip_grad_norm_(adapter_params, 1.0)`, `optimizer.step()`, `scheduler.step()`,
  `optimizer.zero_grad(set_to_none=True)`.
- `optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr, betas,
  weight_decay=0.0)`; `scheduler = get_cosine_schedule_with_warmup(...)`.
- Weights are already bf16 → no AMP context. `maybe_sync(device)` around epoch timing.
- Honor `--max_train_batches` and `--time_budget_minutes` exactly as `train.py` does.

### 5.4 Per-epoch validation + checkpointing

- Val loader with `train=False`, `collate_fn(train=False)`.
- For each val exam: `score_exam(model, processor, item["images"])` → row of `[N, 3]`.
  (`model.eval()` + `torch.no_grad()`.) Also accumulate the teacher-forced val CE (rebuild
  the batch with `train=True` masking) for a `best_val_loss` figure.
- `pooled_auc = compute_auc(y_true.reshape(-1).tolist(), y_pred.reshape(-1).tolist())` —
  **this is the selection metric and the reported `best_val_auc`**.
- `per_task = [compute_auc(y_true[:, i], y_pred[:, i]) for i in range(3)]` — supplementary.
- Save when `pooled_auc` improves:
  `model.save_pretrained(f"models/{prefix}_medgemma_lora_valauc_{pooled_auc:.4f}")`
  (adapter only, ~tens of MB), and delete prior dirs whose name contains `prefix` (mirrors
  `train.py`'s one-best-per-run file cleanup).
- **Early stop** on `--patience` epochs without `pooled_auc` improvement. This is a
  **deliberate divergence** from `train.py` (which early-stops on val *loss*): here val AUC
  is the whole point, and there is no `ReduceLROnPlateau` (cosine schedule instead). Stated
  in `MEDICAL_MODELS.md`.

### 5.5 Final metrics block

Keys chosen to overlap `train.py`'s block where meaningful; extras clearly marked:

```
---
best_val_auc:        0.xxxxxx        # pooled micro-AUC over [N,3] — SAME construction as train.py
best_val_loss:       0.xxxxxx        # mean teacher-forced val CE
per_task_auc:        abnormal=0.xxxx acl=0.xxxx meniscus=0.xxxx   # supplementary (not in train.py)
training_seconds:    xxxx.xx
epochs_ran:          N
best_epoch:          M
num_params_M:        ~4300           # full model (MedGemma-4B)
trainable_params_M:  ~30-45          # LoRA adapters only
model_type:          medgemma-4b-lora
device:              mps
```

### 5.6 `--eval_only`

`PeftModel.from_pretrained(base, adapter_dir)` (base loaded exactly as §5.2), then §5.4
scoring, then §5.5 block. `--dump_predictions` writes the TSV.

### 5.7 Usage

```bash
export HF_TOKEN=...
export PYTORCH_ENABLE_MPS_FALLBACK=1

python vlm_finetune.py --prefix_name medgemma_lora_v1 --data_root MRNet-v1.0 \
  --epochs 3 --grad_accum 8 --slices_per_plane 6 --time_budget_minutes 120

python vlm_finetune.py --eval_only models/medgemma_lora_v1_medgemma_lora_valauc_0.8543 \
  --data_root MRNet-v1.0 --dump_predictions preds.tsv
```

---

## 6. Dependencies

`requirements.txt` additions:

```
transformers>=4.56    # `dtype=` kwarg; current Gemma-3 multimodal handling
peft>=0.13
accelerate>=1.0
sentencepiece
pillow
huggingface_hub>=0.26
pytest
```

No `bitsandbytes`, `open_clip`, `mlx`. First run downloads ~8 GB (`google/medgemma-4b-it`)
to the HF cache; `HF_TOKEN` + accepted license required. `PYTORCH_ENABLE_MPS_FALLBACK=1`
must be exported **before** `python` starts (read at `import torch`); documented in
`MEDICAL_MODELS.md` along with `attn_implementation="eager"` and the MPS caveats.

If path 1 already merged, `transformers` / `huggingface_hub` / `pillow` / `pytest` are
already present — this spec adds `peft`, `accelerate`, `sentencepiece`.

---

## 7. Testing

### 7.1 `tests/conftest.py` additions

- Set `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` **before** importing torch
  (shared with path 1's conftest if merged).
- `mrnet_fixture` — seeded synthetic MRNet tree (shared with path 1): 6 train + 4 valid
  exams, `[s,256,256]` npy per plane with `s ∈ [12,28]`, CSVs with **both classes present**
  per task.
- `tiny_medgemma` — a **hand-built** tiny multimodal model + processor, no download:
  - `Gemma3Config` with `text_config` hidden_size 64 / 2 layers / 2 heads, a minimal
    `vision_config` (image_size 32, patch 16, hidden 32), `Gemma3ForConditionalGeneration`
    from that config with random weights.
  - A `Gemma3Processor` assembled from a tiny tokenizer fixture + `Gemma3ImageProcessor`
    with `size=32`, matching chat template.
  - Saved to `tmp_path` and referenced via `--base_model`.
  - If assembling a faithful processor proves infeasible on the pinned `transformers`, the
    dependent tests fall back to `@pytest.mark.slow` + the real model. The spec commits to
    trying the stub first; §9 tracks it.

### 7.2 Tests

| file | assertions (`pytest -m "not slow"`) |
|---|---|
| `tests/test_vlm_common.py` | `format_answer((1,0,1))` byte-exact. `build_montage` output size `== (cell*cols, cell*rows)`; `RGB`. `sample_slice_indices(n,k)`: `k` sorted, in-bounds ints for both strategies; `n<k` returns `k` (possibly repeated) in-bounds ints. `compute_auc` == `roc_auc_score` on a mixed vector; `== 0.5` on single-class. `digit_token_ids` on the tiny processor returns two distinct ints; `locate_answer_slots` returns 3 strictly-increasing indices `< T` and each decodes to `"0"`. `score_exam` on `tiny_medgemma` returns a `(3,)` array in `[0,1]`; monkeypatching the model's `logits` so slot logits favor `one_id` drives `p1 → 1`. |
| `tests/test_vlm_dataset.py` | With `mrnet_fixture`: `__getitem__` → 3 `PIL.Image` of montage size, `label` shape `(3,)` ordered as `TASKS`. `collate_fn(train=True)` → `input_ids`/`attention_mask`/`pixel_values` present, `labels` has `-100` before the assistant turn and real ids after, `labels.shape == input_ids.shape`. `collate_fn(train=False)` → no `labels`, `add_generation_prompt` applied. Batch dim is 1. |
| `tests/test_vlm_lora_wiring.py` | On `tiny_medgemma`: after `get_peft_model` with the anchored regex, **no** param whose name contains `vision_tower` or `multi_modal_projector` has `requires_grad`; **some** `language_model.*.q_proj` LoRA param does; `trainable_params_M` below the leak tripwire. |
| `tests/test_vlm_finetune_smoke.py` | `vlm_finetune.py --base_model <tiny> --eval_only <tiny-adapter> --data_root <fixture> --max_val_batches 3` → the §5.5 block prints with all keys, `best_val_auc` parseable as a float in `[0,1]`. A 2-step train run (`--epochs 1 --max_train_batches 2`) completes and writes an adapter dir. |
| `tests/test_vlm_real.py` | `@pytest.mark.slow`, needs `HF_TOKEN`. Real `google/medgemma-4b-it` load with `dtype=bfloat16, attn_implementation="eager"`; `score_exam` on one fixture exam returns a finite `(3,)` array; `digit_token_ids` returns the real `0`/`1` ids. |

---

## 8. File inventory

**New**
- `vlm_common.py`, `vlm_dataset.py`, `vlm_finetune.py`
- `tests/test_vlm_common.py`, `tests/test_vlm_dataset.py`, `tests/test_vlm_lora_wiring.py`,
  `tests/test_vlm_finetune_smoke.py`, `tests/test_vlm_real.py`
- `tests/conftest.py` + `pytest.ini` — only if path 1 hasn't already added them; otherwise
  extend (`tiny_medgemma` fixture).

**Modified**
- `requirements.txt` — add `peft`, `accelerate`, `sentencepiece` (+ the path-1 set if path 1
  isn't merged).
- `MEDICAL_MODELS.md` — append the MedGemma-4B LoRA section (workflow, env vars,
  `eager` attention, `do_pan_and_scan=False`, the deliberate early-stop divergence, MPS
  runtime expectations).
- `README.md` — one line under the existing "Pretrained medical models" pointer.

**Untouched:** `train.py`, `lightweight_models.py`, `dataloader.py`, `utils.py`,
`advanced_vit.py`, `medical_encoders.py`, `autoresearch_loop.py`, `research_*.py`,
`pretrain_ssl.py`, `experiment_*.py`.

---

## 9. Risks & mitigations

1. **MedGemma-4B bf16 LoRA on MPS is slow** — ~10–25 min/epoch over ~1,130 train exams.
   Built-in mitigations: `--time_budget_minutes`, `--max_train_batches`, `--grad_accum`.
   First realistic target is a few hundred optimizer steps, not 3 full epochs.
2. **`attn_implementation` / SDPA instability** — `eager` is set explicitly; `test_vlm_real`
   (slow) guards the real-model path.
3. **`target_modules` regex must match the released module paths** — verified at build time
   by printing `model.targeted_module_names` (PEFT) and by
   `test_vlm_lora_wiring` on the tiny model; the `< 60M` trainable assert is the leak
   tripwire. If Gemma-3 uses GQA (smaller `k_proj`/`v_proj`), the estimate lands nearer 30M.
4. **Scoring token alignment** — `locate_answer_slots` works on the processor-expanded
   `input_ids` and `digit_token_ids` derives ids from the real rendering; two asserts
   (`decode == "0"`, all-slots-agree) fail loudly rather than produce a silently invalid
   AUC.
5. **`do_pan_and_scan` default-on** would 3–5× the token budget and memory — set `False`
   everywhere (dataset + scorer).
6. **Tiny stub processor may be hard to assemble faithfully** — if so, the affected tests
   drop to `slow`; the fast suite still covers `vlm_common` (pure) and `vlm_dataset`
   (montage/label shape) which need no model. Tracked here as the main implementation
   unknown.
7. **Small val set (120 exams)** — pooled AUC still has variance; `per_task_auc` +
   `--dump_predictions` expose it.
8. **Class imbalance** (`abnormal` ~80% positive) — the JSON-token CE has no reweighting;
   threshold-free AUC scoring makes this a training-signal concern, not a metric artifact.
   If `acl`/`meniscus` recall lags, revisit with class-balanced sampling or a prompt tweak
   (out of scope for v1).
9. **`compute_auc` copy drift** — if `train.py`'s version ever changes, the copy must be
   updated; a comment in `vlm_common.py` points at `train.py:122-128`.

---

## 10. Deferred / future work

- `--generate` inspection mode (free-text rationale alongside the JSON) for spot-checking —
  not scored, purely diagnostic.
- Class-balanced or focal-style weighting of the answer-token loss.
- `center` slice strategy tuning; per-plane `k`.
- Sharing one `conftest.py` cleanly between path 1 and path 2 (whichever merges second
  extends it).

---

## 11. Open items for implementation

- `sample_slice_indices` strategy bodies (`vlm_common.py`) — **user contribution**.
- `score_exam` step-5 logit→probability mapping (`vlm_common.py`) — **user contribution**;
  default two-way softmax specified.
- Confirm `AutoProcessor.apply_chat_template(..., return_dict=True, tokenize=True)` returns
  `pixel_values` for MedGemma on the pinned `transformers`; confirm how to get
  `assistant_start` (template offset mapping vs marker search).
- Confirm `Gemma3` exposes `vision_tower` / `multi_modal_projector` under those names in the
  installed version (adjust the freeze assert + regex if not).
- Confirm the tiny-`Gemma3` stub + processor is assemblable; else mark the dependent tests
  `slow`.
