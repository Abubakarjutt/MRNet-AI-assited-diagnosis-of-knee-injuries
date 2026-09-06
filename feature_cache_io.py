"""Disk I/O for the frozen-encoder feature cache (spec §4.2).

Layout:  <cache_dir>/<encoder>/<variant>/<split>/<exam_id>.pt
Each .pt is  {plane: {"pooled": [S, D] float16, "patch": [S, Dp, h, w] float16?}}.
No model / transformers import here — this module is pure I/O.
"""

import json
import os

import torch

SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"


def exam_cache_path(cache_dir, encoder, variant, split, exam_id):
    return os.path.join(cache_dir, encoder, variant, split, f"{exam_id}.pt")


def save_exam(path, planes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for plane, feats in planes.items():
        entry = {"pooled": feats["pooled"].detach().to(torch.float16).contiguous()}
        if "patch" in feats and feats["patch"] is not None:
            entry["patch"] = feats["patch"].detach().to(torch.float16).contiguous()
        payload[plane] = entry
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_exam(path):
    # cache files hold only tensors -> weights_only=True (no arbitrary unpickling)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for plane, entry in raw.items():
        conv = {"pooled": entry["pooled"].to(torch.float32)}
        if "patch" in entry:
            conv["patch"] = entry["patch"].to(torch.float32)
        out[plane] = conv
    return out


def read_manifest(cache_dir):
    path = os.path.join(cache_dir, _MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def write_manifest(cache_dir, *, encoders, variants, splits, slices_per_plane,
                   slice_strategy, want_patch_for, git_sha):
    os.makedirs(cache_dir, exist_ok=True)
    man = read_manifest(cache_dir)
    merged_encoders = dict(man.get("encoders", {}))
    merged_encoders.update(encoders)
    man.update(
        schema_version=SCHEMA_VERSION,
        encoders=merged_encoders,
        variants=sorted(set(man.get("variants", [])) | set(variants)),
        splits=sorted(set(man.get("splits", [])) | set(splits)),
        slices_per_plane=slices_per_plane,
        slice_strategy=slice_strategy,
        want_patch_for=sorted(set(man.get("want_patch_for", [])) | set(want_patch_for)),
        git_sha=git_sha,
    )
    tmp = os.path.join(cache_dir, _MANIFEST + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(man, handle, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(cache_dir, _MANIFEST))


def manifest_compatible(manifest, *, encoders, variants, slices_per_plane):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if manifest.get("slices_per_plane") != slices_per_plane:
        return False
    have_enc = set(manifest.get("encoders", {}))
    have_var = set(manifest.get("variants", []))
    return set(encoders).issubset(have_enc) and set(variants).issubset(have_var)
