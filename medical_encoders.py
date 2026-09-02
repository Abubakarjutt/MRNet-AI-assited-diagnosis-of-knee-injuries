from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel


class MedSigLIPEncoder(nn.Module):
    """Frozen SigLIP vision tower from google/medsiglip-448.

    Contract required by FastMRNet._encode_planes:
        forward(flat_inputs: FloatTensor[N, 3, 448, 448]) -> FloatTensor[N, 1152]
    """

    feature_dim = 1152

    def __init__(self, model_id="google/medsiglip-448", chunk_size=32):
        super().__init__()
        full = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        tower = getattr(full, "vision_model", None)
        if tower is None:
            raise AttributeError(
                f"{type(full).__name__} has no .vision_model; cannot use it as a "
                "frozen vision tower for MedSigLIPEncoder."
            )
        self.tower = tower
        # A bound method, not an nn.Module: nn.Module.__setattr__ stores it plainly
        # so it never enters state_dict()/parameters(). Used only as the
        # pooler_output fallback in forward().
        self._image_features_fallback = getattr(full, "get_image_features", None)
        self.feature_dim = 1152
        self.chunk_size = int(chunk_size)

        for parameter in self.tower.parameters():
            parameter.requires_grad_(False)
        self.tower.eval()

    def train(self, mode=True):
        super().train(mode)
        self.tower.eval()
        return self

    @torch.no_grad()
    def forward(self, flat_inputs):
        flat_inputs = flat_inputs.to(dtype=torch.float32)
        pooled_chunks = []
        for start in range(0, flat_inputs.shape[0], self.chunk_size):
            chunk = flat_inputs[start:start + self.chunk_size]
            output = self.tower(pixel_values=chunk)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                if self._image_features_fallback is None:
                    raise RuntimeError(
                        "SigLIP tower returned no pooler_output and the model exposes "
                        "no get_image_features fallback."
                    )
                pooled = self._image_features_fallback(pixel_values=chunk)
            pooled_chunks.append(pooled.float())
        return torch.cat(pooled_chunks, dim=0)

    def state_dict(self, *args, **kwargs):
        destination = kwargs.get("destination")
        if destination is None and args:
            destination = args[0]
        if destination is None:
            destination = OrderedDict()
        return destination

    def _load_from_state_dict(self, *args, **kwargs):
        return
