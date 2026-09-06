import json
import os
import pytest

import feature_cache_io as fcio
from scripts import build_feature_cache as bfc


def test_pick_slice_indices_modes():
    assert bfc.pick_slice_indices(20, 4, "uniform") == sorted(bfc.pick_slice_indices(20, 4, "uniform"))
    assert len(bfc.pick_slice_indices(20, 6, "center")) == 6
    assert bfc.pick_slice_indices(20, 4, "uniform") != bfc.pick_slice_indices(20, 4, "uniform_offset")
    assert bfc.pick_slice_indices(3, 6, "uniform") == [0, 1, 2, 2, 2, 2] or len(bfc.pick_slice_indices(3, 6, "uniform")) == 6


def test_build_cache_writes_files_and_manifest(mrnet_fixture, tmp_path):
    out = tmp_path / "fc"
    bfc.build_cache(
        data_root=str(mrnet_fixture), out_dir=str(out),
        encoders=["fake"], variants=["clean", "hflip"], splits=["train"],
        slices_per_plane=8, want_patch_for=["meniscus"], device="cpu",
        allow_low_disk=True,
    )
    p = fcio.exam_cache_path(str(out), "fake", "clean", "train", "0000")
    assert os.path.isfile(p)
    exam = fcio.load_exam(p)
    assert exam["sagittal"]["pooled"].shape == (8, 16)
    assert exam["sagittal"]["patch"].shape == (8, 8, 4, 4)     # meniscus plane -> patch present
    assert "patch" not in exam["axial"]                        # axial not a meniscus plane
    man = json.loads((out / "manifest.json").read_text())
    assert man["encoders"]["fake"]["pooled_dim"] == 16
    assert set(man["variants"]) == {"clean", "hflip"}


def test_build_cache_is_idempotent(mrnet_fixture, tmp_path):
    out = tmp_path / "fc"
    kw = dict(data_root=str(mrnet_fixture), out_dir=str(out), encoders=["fake"],
              variants=["clean"], splits=["train"], slices_per_plane=8,
              want_patch_for=[], device="cpu", allow_low_disk=True)
    bfc.build_cache(**kw)
    p = fcio.exam_cache_path(str(out), "fake", "clean", "train", "0000")
    mtime = os.path.getmtime(p)
    bfc.build_cache(**kw)                                       # second run skips
    assert os.path.getmtime(p) == mtime


def test_build_cache_rejects_shape_incompatible_rebuild(mrnet_fixture, tmp_path):
    out = tmp_path / "fc"
    kw = dict(data_root=str(mrnet_fixture), out_dir=str(out), encoders=["fake"],
              variants=["clean"], splits=["train"], want_patch_for=[],
              device="cpu", allow_low_disk=True)
    bfc.build_cache(slices_per_plane=8, **kw)
    with pytest.raises(RuntimeError):
        bfc.build_cache(slices_per_plane=6, **kw)
    bfc.build_cache(slices_per_plane=8, **kw)                    # same shape -> ok
    bfc.build_cache(slices_per_plane=6, force=True, **kw)        # force -> ok


def test_build_cache_low_disk_gate(monkeypatch, mrnet_fixture, tmp_path):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda _p: __import__("collections").namedtuple("d", "total used free")(0, 0, 1))
    with pytest.raises(RuntimeError):
        bfc.build_cache(data_root=str(mrnet_fixture), out_dir=str(tmp_path / "fc"),
                        encoders=["fake"], variants=["clean"], splits=["train"],
                        slices_per_plane=8, want_patch_for=["meniscus"], device="cpu")
