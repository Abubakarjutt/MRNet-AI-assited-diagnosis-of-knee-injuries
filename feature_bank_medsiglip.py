"""MedSigLIP-448 as a FrozenEncoder for the feature bank (spec §3.1).

Wraps the Path-1 medical_encoders.MedSigLIPEncoder; adds resize/normalise and the
uint8 slice contract. transformers is pulled in transitively via medical_encoders.
"""

import torch
import torch.nn as nn

from feature_bank import _to_model_input, SIGLIP_MEAN, SIGLIP_STD
from medical_encoders import MedSigLIPEncoder


class MedSigLIPBankEncoder(nn.Module):
    name = "medsiglip"
    pooled_dim = 1152
    patch_dim = 1152
    patch_grid = (32, 32)
    input_size = 448

    def __init__(self, *, model_id="google/medsiglip-448", chunk_size=32,
                 want_patch=False, device="cpu"):
        super().__init__()
        self.inner = MedSigLIPEncoder(model_id=model_id, chunk_size=chunk_size)
        self.want_patch = bool(want_patch)
        self.device = torch.device(device)
        self.inner.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.inner.eval()
        return self

    @torch.no_grad()
    def encode_slices(self, slices_u8):
        x = _to_model_input(slices_u8, self.input_size, SIGLIP_MEAN, SIGLIP_STD, self.device)
        out = self.inner(x, want_patch=self.want_patch)
        if not self.want_patch:
            return {"pooled": out.float().contiguous()}
        return {"pooled": out["pooled"].float().contiguous(),
                "patch": out["patch"].float().contiguous()}
