import os
import torch
import pytest
import feature_cache_io as fcio


def _exam(with_patch=False):
    planes = {p: {"pooled": torch.randn(8, 12)} for p in ("sagittal", "coronal", "axial")}
    if with_patch:
        for p in ("sagittal", "coronal"):
            planes[p]["patch"] = torch.randn(8, 4, 3, 3)
    return planes


def test_save_load_round_trips_shape_and_is_float32(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "fake", "clean", "train", "0007")
    src = _exam(with_patch=True)
    fcio.save_exam(path, src)
    assert os.path.isfile(path)
    out = fcio.load_exam(path)
    assert set(out) == {"sagittal", "coronal", "axial"}
    assert out["sagittal"]["pooled"].shape == (8, 12)
    assert out["sagittal"]["pooled"].dtype == torch.float32
    assert out["coronal"]["patch"].shape == (8, 4, 3, 3)
    assert "patch" not in out["axial"]
    torch.testing.assert_close(out["sagittal"]["pooled"], src["sagittal"]["pooled"], rtol=1e-2, atol=1e-2)


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "fake", "clean", "train", "0000")
    fcio.save_exam(path, _exam())
    assert not os.path.exists(path + ".tmp")


def test_exam_cache_path_layout(tmp_path):
    path = fcio.exam_cache_path(str(tmp_path), "dinov2", "hflip", "valid", "1130")
    assert path.replace(str(tmp_path), "").lstrip("/") == "dinov2/hflip/valid/1130.pt"


def test_manifest_write_read_merge_and_compat(tmp_path):
    fcio.write_manifest(
        str(tmp_path),
        encoders={"medsiglip": {"pooled_dim": 1152, "patch_dim": 1152, "patch_grid": [32, 32]}},
        variants=["clean", "hflip"], splits=["train"],
        slices_per_plane=32, slice_strategy="uniform",
        want_patch_for=["meniscus"], git_sha="abc123",
    )
    fcio.write_manifest(
        str(tmp_path),
        encoders={"dinov2": {"pooled_dim": 768, "patch_dim": 768, "patch_grid": [16, 16]}},
        variants=["rotp"], splits=["valid"],
        slices_per_plane=32, slice_strategy="uniform",
        want_patch_for=["meniscus"], git_sha="abc123",
    )
    man = fcio.read_manifest(str(tmp_path))
    assert man["schema_version"] == fcio.SCHEMA_VERSION
    assert set(man["encoders"]) == {"medsiglip", "dinov2"}
    assert set(man["variants"]) == {"clean", "hflip", "rotp"}
    assert set(man["splits"]) == {"train", "valid"}
    assert fcio.manifest_compatible(man, encoders=["medsiglip", "dinov2"], variants=["clean", "rotp"], slices_per_plane=32)
    assert not fcio.manifest_compatible(man, encoders=["biomedclip"], variants=["clean"], slices_per_plane=32)
    assert not fcio.manifest_compatible(man, encoders=["medsiglip"], variants=["clean"], slices_per_plane=24)
