import torch
from torch.utils.data import DataLoader

from dataloader import FeatureCacheDataset, featbank_collate


def _ds(cache, mrnet_fixture, **kw):
    return FeatureCacheDataset(
        str(cache), "train", encoders=["fake"], variants=["clean", "hflip", "slicesB"],
        data_root=str(mrnet_fixture), slices_used=4, slices_used_meniscus=6, seed=0, **kw,
    )


def test_item_payload_shapes(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    payload, label, weights, exam_id = ds[0]
    assert label.shape == (3,)
    assert payload["pooled"]["fake"]["sagittal"].shape == (4, 16)
    assert payload["patch"]["fake"]["coronal"].shape == (6, 8, 4, 4)
    assert "axial" not in payload["patch"]["fake"]
    assert ds.encoder_dims["fake"] == 16 and ds.patch_dims["fake"] == 8


def test_eval_dataset_is_deterministic_and_uses_first_variant(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture, train=False)
    a = ds[1][0]["pooled"]["fake"]["axial"]
    b = ds[1][0]["pooled"]["fake"]["axial"]
    torch.testing.assert_close(a, b)


def test_set_epoch_changes_variant_selection(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    seen = set()
    for e in range(6):
        ds.set_epoch(e)
        seen.add(ds[0][0]["pooled"]["fake"]["sagittal"].sum().item())
    assert len(seen) > 1                     # different variant => different features


def test_collate_stacks_batch(feature_cache_fixture, mrnet_fixture):
    ds = _ds(feature_cache_fixture, mrnet_fixture)
    loader = DataLoader(ds, batch_size=3, collate_fn=featbank_collate)
    payload, labels, weights, ids = next(iter(loader))
    assert labels.shape == (3, 3)
    assert payload["pooled"]["fake"]["sagittal"].shape == (3, 4, 16)
    assert payload["patch"]["fake"]["sagittal"].shape == (3, 6, 8, 4, 4)
    assert len(ids) == 3


def test_slices_used_over_cap_raises(feature_cache_fixture, mrnet_fixture):
    import pytest
    with pytest.raises(AssertionError):
        FeatureCacheDataset(str(feature_cache_fixture), "train", encoders=["fake"],
                            variants=["clean"], data_root=str(mrnet_fixture), slices_used=999)
