"""Fast-suite tests for vlm_dataset (fake processor, synthetic fixture; no model download)."""

import pytest
from PIL import Image

import vlm_dataset
from dataloader import TASKS


def _item(mrnet_fixture, fake_processor):
    ds = vlm_dataset.MRVLMDataset(
        mrnet_fixture, train=True, k=6, grid=(3, 2), cell=32, mmap=True, cache_size=4
    )
    return ds[0]


def test_getitem_returns_montages_and_label(mrnet_fixture, fake_processor):
    item = _item(mrnet_fixture, fake_processor)
    assert isinstance(item["exam_id"], str)
    assert len(item["images"]) == 3
    for img in item["images"]:
        assert isinstance(img, Image.Image)
        assert img.size == (32 * 2, 32 * 3)     # cell*cols, cell*rows
        assert img.mode == "RGB"
    assert item["label"].shape == (3,)


def test_collate_train_builds_masked_labels(mrnet_fixture, fake_processor):
    import vlm_common
    from vlm_dataset import _assistant_start

    item = _item(mrnet_fixture, fake_processor)
    out = vlm_dataset.collate_fn([item], fake_processor, train=True)
    for key in ("input_ids", "attention_mask", "pixel_values", "labels", "label", "exam_id"):
        assert key in out
    assert out["labels"].shape == out["input_ids"].shape
    assert out["input_ids"].shape[0] == 1

    conv = vlm_common.build_conversation(
        item["images"], answer=vlm_common.format_answer(item["label"].int().tolist())
    )
    start = _assistant_start(fake_processor, conv)
    labels0 = out["labels"][0]
    assert 0 < start < labels0.shape[0]
    assert (labels0[:start] == -100).all()          # whole prompt masked, exact boundary
    assert (labels0[start:] != -100).all()          # every answer token scored


def test_collate_eval_omits_labels(mrnet_fixture, fake_processor):
    import vlm_common

    item = _item(mrnet_fixture, fake_processor)
    out = vlm_dataset.collate_fn([item], fake_processor, train=False)
    assert "labels" not in out
    for key in ("input_ids", "attention_mask", "pixel_values"):
        assert key in out
    assert out["input_ids"].shape[0] == 1

    # add_generation_prompt=True must have taken effect: exactly one extra (assistant-role)
    # token vs the same prompt templated WITHOUT a generation prompt.
    prompt_conv = vlm_common.build_conversation(item["images"], answer=None)
    without = fake_processor.apply_chat_template(
        prompt_conv, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt", do_pan_and_scan=False,
    )
    assert out["input_ids"].shape[1] == without["input_ids"].shape[1] + 1


def test_collate_requires_single_batch(mrnet_fixture, fake_processor):
    item = _item(mrnet_fixture, fake_processor)
    with pytest.raises(ValueError):
        vlm_dataset.collate_fn([item, item], fake_processor, train=True)
