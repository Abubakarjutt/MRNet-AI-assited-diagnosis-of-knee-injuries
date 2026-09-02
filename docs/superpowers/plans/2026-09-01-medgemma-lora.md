# MedGemma-4B LoRA Classifier — Implementation Plan

> **Spec:** `docs/superpowers/specs/2026-09-01-medgemma-lora-design.md` (authoritative; read alongside).
> **Sibling:** `docs/superpowers/plans/2026-09-01-medsiglip-encoder.md` (Path 1, already merged to `main`).

**Goal:** Add a standalone LoRA fine-tune of `google/medgemma-4b-it` that classifies a knee MRI
exam into the three MRNet labels, scored with a deterministic teacher-forced token-probability
readout so its **pooled micro-AUC is constructed identically to `train.py`'s `best_val_auc`**.

**Branch:** `feat/medgemma-lora` (off `main`, after Path 1). Independent PR.

## Global constraints

- Apple Silicon / **MPS only**; no `bitsandbytes` (bf16 base + bf16/fp16 LoRA adapters).
- `HF_TOKEN` + accepted `google/medgemma-4b-it` license at *real-model* runtime.
- **Fast suite** (`pytest -m "not slow"`) must pass with **no model download** and **no MPS**.
  Model-dependent paths are gated behind `@pytest.mark.slow` (needs `HF_TOKEN`) or a hand-built
  fake processor/model fixture in `conftest.py`.
- **Nothing** in `train.py`, `lightweight_models.py`, `dataloader.py`, `utils.py`,
  `medical_encoders.py`, `advanced_vit.py`, `research_*.py`, `autoresearch_loop.py`,
  `pretrain_ssl.py`, `experiment_*.py` is modified.
- `compute_auc` is **copied verbatim** from `train.py:144-150` into `vlm_common.py` (not imported;
  importing `train.py` triggers module-level side effects). A comment points at the source line.
- `dataloader.TASKS` is **imported**, the single source of label-column order.

## File structure

**New**
- `vlm_common.py` — pure, model-independent helpers (prompt, `compute_auc` copy, slice sampling,
  montage, answer formatting, conversation build, prompt masking, and the deterministic
  teacher-forced scoring: `digit_token_ids`, `locate_answer_slots`, `score_exam`).
- `vlm_dataset.py` — `MRVLMDataset(MRMultiPlaneDataset)` + `collate_fn(processor, train)`.
- `vlm_finetune.py` — LoRA SFT loop, per-epoch val-AUC selection + early stop, best-adapter
  checkpointing, `--eval_only`, the `§5.5` metrics block.
- `tests/test_vlm_common.py`, `tests/test_vlm_dataset.py`, `tests/test_vlm_lora_wiring.py`,
  `tests/test_vlm_finetune_smoke.py`, `tests/test_vlm_real.py`.
- Extend `tests/conftest.py` with a `fake_processor` + `fake_model` fixture (no download).
- `requirements.txt` — add `peft>=0.13`, `accelerate>=1.0`, `sentencepiece`.
- `MEDICAL_MODELS.md` — append the MedGemma-4B LoRA section; one-line README pointer.

## Tasks

- [ ] **T1 — `vlm_common.py` (pure).** Implement `SYSTEM`/`INSTRUCTION`, `compute_auc` (verbatim
      copy + `# train.py:144-150` comment), `sample_slice_indices` (uniform + center; n<k repeats),
      `build_montage` (per-slice min-max→uint8→bicubic→RGB grid; lazy `PIL` import so the module
      imports without Pillow), `format_answer` (byte-exact fixed JSON), `build_conversation`,
      `mask_prompt_tokens`, `digit_token_ids`, `locate_answer_slots`, `score_exam`.
- [ ] **T2 — `conftest.py` fixture.** `fake_processor` (char-level tokenizer with distinct `0`/`1`
      tokens + a minimal `apply_chat_template` that expands images to a fixed placeholder count and
      appends the assistant answer subsequence contiguously) and `fake_model` (returns
      `.logits [1,T,V]`; monkeypatchable for the `p1→1` test). No network, no MPS.
- [ ] **T3 — tests `test_vlm_common.py`.** Pure assertions (format_answer byte-exact;
      sample_slice_indices contract for both strategies + n<k; compute_auc == roc_auc_score and
      ==0.5 single-class; build_montage size/RGB). Scoring: `digit_token_ids` distinct;
      `locate_answer_slots` 3 strictly-increasing in-bounds indices each decoding to `"0"`;
      `score_exam` returns `(3,)` in `[0,1]` and favoring `one_id` drives `p1→1`.
- [ ] **T4 — `vlm_dataset.py` + `test_vlm_dataset.py`.** `MRVLMDataset.__getitem__` → 3 PIL
      montages + `label[3]` (TASKS order) + `exam_id`; `collate_fn(processor, train=True)` builds
      `input_ids`/`attention_mask`/`pixel_values` + masked `labels` (`-100` before the assistant
      turn, shape == `input_ids`); `train=False` omits `labels` and applies `add_generation_prompt`.
      Gated on the fake processor.
- [ ] **T5 — `vlm_finetune.py` + `test_vlm_finetune_smoke.py` + `test_vlm_lora_wiring.py`.** LoRA
      on `language_model.` only; vision tower + projector frozen (assert); `trainable_params_M`
      < 60M tripwire. bf16 + `attn_implementation="eager"` + `device_map={"":"mps"}`,
      `do_pan_and_scan=False`. Per-epoch val: pooled micro-AUC is the selection metric + early stop
      on `--patience`; `model.save_pretrained(...)` best-adapter + one-best-per-run cleanup.
      `--eval_only` + `--dump_predictions`. `§5.5` metrics block. Gated on `peft` import.
- [ ] **T6 — `test_vlm_real.py` (`@slow`)** + `MEDICAL_MODELS.md` section + README pointer.
- [ ] **T7 — Fast suite green; commit per task.**

## Open items (from spec §11) — decisions surfaced to the user

1. **`sample_slice_indices` strategy bodies** — *user contribution*. Default implemented:
   `uniform` = rounded `linspace`; `center` = deterministic (fixed-seed) normal about the mid-slice,
   clipped/rounded/sorted. Clearly marked overridable.
2. **`score_exam` step-5 logit→probability mapping** — *user contribution*. Default = two-way
   softmax over `{zero_id, one_id}` (the MCQ-eval standard); alternatives (full-vocab softmax read
   `P(one_id)`, a temperature) documented in a comment.
3. **Confirm** `apply_chat_template(return_dict=True, tokenize=True)` yields `pixel_values` for
   MedGemma on the pinned `transformers`, and how to obtain `assistant_start`
   (offset mapping vs marker search). → real-model slow test.
4. **Confirm** `Gemma3` exposes `vision_tower` / `multi_modal_projector` under those names in the
   installed version (adjust freeze assert + regex if not). → real-model slow test.
5. **Tiny-`Gemma3` fixture assemblability** — the spec's #1 implementation unknown. In this
   sandbox (no network, MPS off) the real fixture is unavailable; a hand-built *fake* processor/
   model stands in for the fast suite and the real path is `@slow`.
