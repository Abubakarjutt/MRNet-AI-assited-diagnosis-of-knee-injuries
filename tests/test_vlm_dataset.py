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
    item = _item(mrnet_fixture, fake_processor)
    out = vlm_dataset.collate_fn([item], fake_processor, train=True)
    for key in ("input_ids", "attention_mask", "pixel_values", "labels", "label", "exam_id"):
        assert key in out
    assert out["labels"].shape == out["input_ids"].shape
    assert out["input_ids"].shape[0] == 1           # batch dim is 1
    labels0 = out["labels"][0]
    assert int(labels0[0]) == -100                  # prompt is masked
    assert int(labels0[-1]) != -100                 # answer tokens are scored
    assert (labels0[: out["labels"].shape[1] - 1] == -100).any()


def test_collate_eval_omits_labels(mrnet_fixture, fake_processor):
    item = _item(mrnet_fixture, fake_processor)
    out = vlm_dataset.collate_fn([item], fake_processor, train=False)
    assert "labels" not in out
    for key in ("input_ids", "attention_mask", "pixel_values"):
        assert key in out
    assert out["input_ids"].shape[0] == 1


def test_collate_requires_single_batch(mrnet_fixture, fake_processor):
    item = _item(mrnet_fixture, fake_processor)
    with pytest.raises(ValueError):
        vlm_dataset.collate_fn([item, item], fake_processor, train=True)
