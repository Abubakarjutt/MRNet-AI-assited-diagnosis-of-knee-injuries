"""Frozen pretrained encoders for the MRNet feature bank (spec §3).

All `transformers` imports live here. Each encoder implements the FrozenEncoder
protocol: encode_slices(uint8 [S,H,W]) -> {"pooled": [S,pooled_dim],
"patch": [S,patch_dim,gh,gw]?}. Resize / 3-channel replication / normalisation
are each encoder's own business.
"""

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]


class FrozenEncoder(Protocol):
    name: str
    pooled_dim: int
    patch_dim: int
    patch_grid: tuple
    input_size: int
    want_patch: bool

    def encode_slices(self, slices_u8: torch.Tensor) -> dict: ...


def _to_model_input(slices_u8, size, mean, std, device):
    """uint8 [S,H,W] -> float [S,3,size,size], resized + normalised."""
    x = slices_u8.to(device=device, dtype=torch.float32).div_(255.0)
    x = x.unsqueeze(1)                                  # [S,1,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)                            # [S,3,size,size]
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


class FakeBankEncoder(nn.Module):
    """Deterministic, download-free stand-in for the fast test suite."""

    name = "fake"
    pooled_dim = 16
    patch_dim = 8
    patch_grid = (4, 4)
    input_size = 32

    def __init__(self, *, chunk_size=32, want_patch=False, device="cpu"):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, [0.5] * 3, [0.5] * 3, self.device)
        s = x.shape[0]
        step = self.chunk_size if self.chunk_size > 0 else s
        pooled_chunks, patch_chunks = [], []
        for start in range(0, s, step):
            chunk = x[start:start + step]                        # [c,3,32,32]
            gh, gw = self.patch_grid
            grid = F.adaptive_avg_pool2d(chunk, (gh, gw))        # [c,3,gh,gw]
            # pooled: deterministic function of per-channel grid stats
            flat = grid.mean(dim=(2, 3))                         # [c,3]
            pooled = torch.cat([flat, flat.pow(2), flat.roll(1, 1),
                                flat.flip(1), flat.mean(1, keepdim=True).repeat(1, 4)], dim=1)
            pooled_chunks.append(pooled[:, :self.pooled_dim])
            if self.want_patch:
                patch = grid.repeat(1, 3, 1, 1)[:, :self.patch_dim]   # [c,8,gh,gw]
                patch_chunks.append(patch)
        out = {"pooled": torch.cat(pooled_chunks, dim=0).contiguous()}
        if self.want_patch:
            out["patch"] = torch.cat(patch_chunks, dim=0).contiguous()
        return out


class DINOv2Encoder(nn.Module):
    name = "dinov2"
    pooled_dim = 768
    patch_dim = 768
    patch_grid = (16, 16)
    input_size = 224

    def __init__(self, *, model_id="facebook/dinov2-base", chunk_size=32,
                 want_patch=False, device="cpu"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.chunk_size = int(chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)
        self.model.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, IMAGENET_MEAN, IMAGENET_STD, self.device)
        s = x.shape[0]
        step = self.chunk_size if self.chunk_size > 0 else s
        gh, gw = self.patch_grid
        pooled_chunks, patch_chunks = [], []
        for start in range(0, s, step):
            out = self.model(pixel_values=x[start:start + step])
            pooled_chunks.append(out.pooler_output.float())               # [c,768]
            if self.want_patch:
                tokens = out.last_hidden_state[:, 1:, :].float()          # [c,256,768]
                c = tokens.shape[0]
                patch = tokens.transpose(1, 2).reshape(c, self.patch_dim, gh, gw)
                patch_chunks.append(patch)
        res = {"pooled": torch.cat(pooled_chunks, dim=0).contiguous()}
        if self.want_patch:
            res["patch"] = torch.cat(patch_chunks, dim=0).contiguous()
        return res


def build_encoder(name, *, chunk_size=32, want_patch=False, device="cpu"):
    if name == "fake":
        return FakeBankEncoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    if name == "dinov2":
        return DINOv2Encoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    if name == "medsiglip":
        from feature_bank_medsiglip import MedSigLIPBankEncoder   # wired in Task 3
        return MedSigLIPBankEncoder(chunk_size=chunk_size, want_patch=want_patch, device=device)
    raise ValueError(f"unknown encoder: {name!r}")
