import numpy as np
import pytest
import torch
from PIL import Image
from sklearn import metrics

import vlm_common
from dataloader import TASKS


# --- format_answer ------------------------------------------------------------- #
def test_format_answer_byte_exact():
    assert vlm_common.format_answer((1, 0, 1)) == '{"abnormal": 1, "acl": 0, "meniscus": 1}'
    assert vlm_common.format_answer([0, 0, 0]) == '{"abnormal": 0, "acl": 0, "meniscus": 0}'
    # values are coerced to {0, 1} and key order == TASKS
    assert vlm_common.format_answer((2, 0, 3)) == '{"abnormal": 1, "acl": 0, "meniscus": 1}'


def test_format_answer_requires_three_labels():
    with pytest.raises(ValueError):
        vlm_common.format_answer([0, 1])


# --- sample_slice_indices ------------------------------------------------------ #
@pytest.mark.parametrize("strategy", ["uniform", "center"])
def test_sample_slice_indices_contract(strategy):
    n, k = 24, 6
    idx = vlm_common.sample_slice_indices(n, k, strategy)
    assert len(idx) == k
    assert idx == sorted(idx)                 # sorted
    assert all(0 <= i < n for i in idx)        # in-bounds ints
    assert all(isinstance(i, int) for i in idx)


def test_sample_slice_indices_uniform_is_spread():
    idx = vlm_common.sample_slice_indices(100, 5, "uniform")
    assert idx[0] == 0 and idx[-1] == 99
    assert idx == sorted(set(idx)) or len(set(idx)) == 5   # near-uniform, may collide at small n


def test_sample_slice_indices_fewer_than_k_repeats():
    idx = vlm_common.sample_slice_indices(3, 6, "uniform")
    assert len(idx) == 6
    assert idx == sorted(idx)
    assert all(0 <= i < 3 for i in idx)
    assert len(set(idx)) <= 3            # must repeat


def test_sample_slice_indices_unknown_strategy_raises():
    with pytest.raises(ValueError):
        vlm_common.sample_slice_indices(10, 4, "bogus")


# --- compute_auc (verbatim copy of train.py:144-150) --------------------------- #
def test_compute_auc_matches_sklearn_on_mixed():
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.3, 0.7, 0.6, 0.2]
    assert abs(vlm_common.compute_auc(y_true, y_pred) - metrics.roc_auc_score(y_true, y_pred)) < 1e-9


def test_compute_auc_single_class_is_half():
    assert vlm_common.compute_auc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5


# --- build_montage ------------------------------------------------------------- #
def test_build_montage_shape_and_mode():
    vol = torch.rand(6, 256, 256, dtype=torch.float32) * 255.0
    indices = vlm_common.sample_slice_indices(vol.shape[0], 6, "uniform")
    img = vlm_common.build_montage(vol, indices, grid=(3, 2), cell=32)
    assert img.size == (32 * 2, 32 * 3)     # (cell*cols, cell*rows)
    assert img.mode == "RGB"


def test_build_montage_grid_length_must_match():
    vol = torch.rand(6, 16, 16, dtype=torch.float32)
    with pytest.raises(ValueError):
        vlm_common.build_montage(vol, [0, 1, 2], grid=(3, 2), cell=16)   # 3 != 6


# --- conversation / masking ---------------------------------------------------- #
def test_build_conversation_order_and_answer():
    conv = vlm_common.build_conversation([None, None, None], answer=vlm_common.format_answer((1, 0, 1)))
    roles = [m["role"] for m in conv]
    assert roles == ["system", "user", "assistant"]
    user_parts = conv[1]["content"]
    assert [p["type"] for p in user_parts[:3]] == ["image", "image", "image"]
    assert conv[2]["content"][0]["text"] == '{"abnormal": 1, "acl": 0, "meniscus": 1}'


def test_build_conversation_no_answer_at_eval():
    conv = vlm_common.build_conversation([None, None, None], answer=None)
    assert [m["role"] for m in conv] == ["system", "user"]


def test_mask_prompt_tokens():
    ids = torch.tensor([1, 2, 3, 4, 5])
    labels = vlm_common.mask_prompt_tokens(ids, assistant_start=3)
    assert labels.tolist() == [-100, -100, -100, 4, 5]
    assert labels is not ids            # cloned, not aliased


# --- scoring (fake processor + fake model) ------------------------------------- #
def test_digit_token_ids_distinct(fake_processor):
    zero_id, one_id = vlm_common.digit_token_ids(fake_processor)
    assert isinstance(zero_id, int) and isinstance(one_id, int)
    assert zero_id != one_id
    # the fake is char-level, so these are exactly the '0'/'1' char ids
    assert fake_processor.tokenizer.decode([zero_id]) == "0"
    assert fake_processor.tokenizer.decode([one_id]) == "1"


def test_locate_answer_slots(fake_processor):
    images = [torch.zeros(4, 32, 32) for _ in range(3)]
    conv = vlm_common.build_conversation(images, answer=vlm_common.format_answer((0, 0, 0)))
    batch = fake_processor.apply_chat_template(
        conv, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt", do_pan_and_scan=False,
     )
    T = batch["input_ids"].shape[1]
    slots = vlm_common.locate_answer_slots(fake_processor, batch["input_ids"][0])
    assert len(slots) == 3
    assert all(slots[i] < slots[i + 1] for i in range(2))
    assert all(s < T for s in slots)
    tok = fake_processor.tokenizer
    assert all(tok.decode([batch["input_ids"][0][s].item()]).strip() == "0" for s in slots)


def test_score_exam_shape_and_range(fake_processor, fake_model):
    images = [torch.rand(4, 32, 32) for _ in range(3)]
    probs = vlm_common.score_exam(fake_model, fake_processor, images)
    assert probs.shape == (3,)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_score_exam_favors_one_id(fake_processor, fake_model):
    images = [torch.rand(4, 32, 32) for _ in range(3)]
    zero_id, one_id = vlm_common.digit_token_ids(fake_processor)
    conv = vlm_common.build_conversation(images, answer=vlm_common.format_answer((0, 0, 0)))
    batch = fake_processor.apply_chat_template(
        conv, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt", do_pan_and_scan=False,
     )
    T, V = batch["input_ids"].shape[1], fake_model.vocab_size
    logits = torch.full((1, T, V), -100.0)
    logits[:, :, one_id] = 100.0
    fake_model._logits = logits
    probs = vlm_common.score_exam(fake_model, fake_processor, images)
    assert probs.shape == (3,)
    assert all(p > 0.99 for p in probs)     # P(one_id) -> 1 at every slot


# --- C4: score_exam casts pixel_values to the model dtype -------------------- #
def test_score_exam_casts_pixel_values_to_model_dtype(fake_processor):
    import torch
    from types import SimpleNamespace
    import vlm_common

    seen = []

    class _M:
        device = torch.device("cpu")
        dtype = torch.float64

        def __call__(self, **batch):
            seen.append(batch["pixel_values"].dtype)
            T = batch["input_ids"].shape[-1]
            return SimpleNamespace(logits=torch.zeros(1, T, 512))

        def eval(self):
            return self

    imgs = [object(), object(), object()]        # fake_processor ignores image content
    out = vlm_common.score_exam(_M(), fake_processor, imgs)
    assert out.shape == (3,)
    assert seen and all(d == torch.float64 for d in seen)


# --- I7: the pos-1 causal alignment in score_exam is under test ----------------- #
def _readout_ids(fake_processor):
    imgs = [Image.new("RGB", (32, 32)) for _ in range(3)]
    conv = vlm_common.build_conversation(
        imgs, answer=vlm_common.format_answer((0, 0, 0))
    )
    batch = fake_processor.apply_chat_template(
        conv, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt", do_pan_and_scan=False,
    )
    return imgs, batch["input_ids"][0]


def test_score_exam_reads_logits_at_slot_minus_one(fake_model, fake_processor):
    """Causal-LM convention: P(1) for a value slot is read at pos-1. Boosting one_id at
    exactly the slot-minus-one rows must drive every returned p1 -> 1."""
    imgs, ids = _readout_ids(fake_processor)
    slots = vlm_common.locate_answer_slots(fake_processor, ids)
    _zero_id, one_id = vlm_common.digit_token_ids(fake_processor)
    T = int(ids.shape[0])
    logits = torch.zeros(1, T, fake_model.vocab_size)
    for s in slots:
        logits[0, s - 1, one_id] = 50.0
    fake_model._logits = logits
    probs = vlm_common.score_exam(fake_model, fake_processor, imgs)
    assert np.allclose(probs, 1.0, atol=1e-3), probs


def test_score_exam_ignores_logits_at_the_slot_itself(fake_model, fake_processor):
    """Negative control: boosting one_id AT the slot row (pos, not pos-1) must NOT move
    p1 — rules out an off-by-one in the other direction."""
    imgs, ids = _readout_ids(fake_processor)
    slots = vlm_common.locate_answer_slots(fake_processor, ids)
    _zero_id, one_id = vlm_common.digit_token_ids(fake_processor)
    T = int(ids.shape[0])
    logits = torch.zeros(1, T, fake_model.vocab_size)
    for s in slots:
        logits[0, s, one_id] = 50.0
    fake_model._logits = logits
    probs = vlm_common.score_exam(fake_model, fake_processor, imgs)
    assert np.allclose(probs, 0.5, atol=1e-3), probs
