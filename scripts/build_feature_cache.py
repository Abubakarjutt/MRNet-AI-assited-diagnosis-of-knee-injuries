"""Offline frozen-encoder feature cache for MRNet Path 3 (spec §4).

Usage:
  python scripts/build_feature_cache.py --data_root MRNet-v1.0 \
    --encoders medsiglip,dinov2 --variants clean,hflip,rotp,rotn,slicesB,slicesC \
    --slices_per_plane 32 --want_patch_for meniscus --out feature_cache/
"""

import argparse
import os
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_cache_io as fcio                      # noqa: E402
from dataloader import PLANES, TASKS, MRMultiPlaneDataset   # noqa: E402
from feature_bank import build_encoder               # noqa: E402

MIN_FREE_BYTES_FOR_PATCH = 16 * (1024 ** 3)
MENISCUS_PATCH_PLANES = ("sagittal", "coronal")

VARIANTS = {
    "clean":   {"image": "identity", "slices": "uniform"},
    "hflip":   {"image": "hflip",    "slices": "uniform"},
    "rotp":    {"image": "rot+9",    "slices": "uniform"},
    "rotn":    {"image": "rot-9",    "slices": "uniform"},
    "slicesB": {"image": "identity", "slices": "uniform_offset"},
    "slicesC": {"image": "identity", "slices": "center"},
}


def pick_slice_indices(num_slices, k, mode):
    import numpy as np
    if num_slices <= k:
        base = list(range(num_slices)) + [num_slices - 1] * (k - num_slices)
        return base[:k]
    if mode == "center":
        lo = (num_slices - k) // 2
        return list(range(lo, lo + k))
    stride = num_slices / k
    offset = stride / 3.0 if mode == "uniform_offset" else 0.0
    idx = [int(min(num_slices - 1, offset + i * stride)) for i in range(k)]
    return sorted(idx)


def _apply_image_op(volume, op):
    """volume: float tensor [S,H,W]."""
    if op == "identity":
        return volume
    if op == "hflip":
        return torch.flip(volume, dims=[2])
    if op.startswith("rot"):
        from torchvision.transforms.functional import rotate
        angle = float(op[3:])
        return rotate(volume, angle)
    raise ValueError(f"unknown image op: {op}")


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_cache(*, data_root, out_dir, encoders, variants, splits,
                slices_per_plane=32, slice_strategy="uniform",
                want_patch_for=("meniscus",), chunk_size=32, device="cpu",
                force=False, allow_low_disk=False):
    want_patch_for = list(want_patch_for)
    needs_patch = "meniscus" in want_patch_for
    if needs_patch and not allow_low_disk:
        free = shutil.disk_usage(_parent_that_exists(out_dir)).free
        if free < MIN_FREE_BYTES_FOR_PATCH:
            raise RuntimeError(
                f"{free / 2**30:.1f} GiB free < 16 GiB required for a patch-grid pass; "
                "free disk or pass allow_low_disk=True (spec §4.4)."
            )

    built_encoders = {}
    for enc_name in encoders:
        enc = build_encoder(enc_name, chunk_size=chunk_size,
                            want_patch=needs_patch, device=device)
        built_encoders[enc_name] = {
            "pooled_dim": enc.pooled_dim, "patch_dim": enc.patch_dim,
            "patch_grid": list(enc.patch_grid),
        }
        for split in splits:
            ds = MRMultiPlaneDataset(data_root, train=(split == "train"),
                                     mmap=True, cache_size=0, transform=None)
            for i in range(len(ds)):
                volumes, _label, _w, exam_id = ds[i]
                for variant in variants:
                    path = fcio.exam_cache_path(out_dir, enc_name, variant, split, exam_id)
                    if os.path.isfile(path) and not force:
                        continue
                    recipe = VARIANTS[variant]
                    planes_out = {}
                    for plane, vol in zip(PLANES, volumes):
                        vol_t = _apply_image_op(vol, recipe["image"])
                        idx = pick_slice_indices(vol_t.shape[0], slices_per_plane, recipe["slices"])
                        u8 = vol_t[idx].clamp(0, 255).to(torch.uint8)
                        feats = enc.encode_slices(u8)
                        entry = {"pooled": feats["pooled"].cpu()}
                        if needs_patch and plane in MENISCUS_PATCH_PLANES and "patch" in feats:
                            entry["patch"] = feats["patch"].cpu()
                        planes_out[plane] = entry
                    fcio.save_exam(path, planes_out)

    fcio.write_manifest(
        out_dir, encoders=built_encoders, variants=list(variants), splits=list(splits),
        slices_per_plane=slices_per_plane, slice_strategy=slice_strategy,
        want_patch_for=want_patch_for, git_sha=_git_sha(),
    )


def _parent_that_exists(path):
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", dest="out_dir", default="feature_cache/")
    ap.add_argument("--encoders", default="medsiglip,dinov2")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--splits", default="train,valid")
    ap.add_argument("--slices_per_plane", type=int, default=32)
    ap.add_argument("--slice_strategy", default="uniform")
    ap.add_argument("--want_patch_for", default="meniscus")
    ap.add_argument("--chunk_size", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_low_disk", action="store_true")
    a = ap.parse_args(argv)
    build_cache(
        data_root=a.data_root, out_dir=a.out_dir,
        encoders=a.encoders.split(","), variants=a.variants.split(","),
        splits=[s.replace("valid", "valid") for s in a.splits.split(",")],
        slices_per_plane=a.slices_per_plane, slice_strategy=a.slice_strategy,
        want_patch_for=[x for x in a.want_patch_for.split(",") if x],
        chunk_size=a.chunk_size, device=a.device, force=a.force,
        allow_low_disk=a.allow_low_disk,
    )


if __name__ == "__main__":
    main()
