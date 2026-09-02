import os

import pytest
import torch

import medical_encoders

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN + accepted license")
def test_real_medsiglip_encoder_forward():
    encoder = medical_encoders.MedSigLIPEncoder(chunk_size=2)
    out = encoder(torch.rand(3, 3, 448, 448))
    assert out.shape == (3, 1152)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
