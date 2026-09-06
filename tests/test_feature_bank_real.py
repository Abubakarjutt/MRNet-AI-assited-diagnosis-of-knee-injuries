import torch
import pytest
import feature_bank as fb

pytestmark = pytest.mark.slow


def test_real_dinov2_encode_slices_shapes():
    enc = fb.build_encoder("dinov2", want_patch=True, chunk_size=2)
    out = enc.encode_slices(torch.randint(0, 256, (3, 200, 180), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 768)
    assert out["patch"].shape == (3, 768, 16, 16)
    assert torch.isfinite(out["pooled"]).all()


@pytest.mark.skipif(not __import__("os").environ.get("HF_TOKEN"),
                    reason="needs HF_TOKEN + accepted MedSigLIP license")
def test_real_medsiglip_encode_slices_shapes():
    enc = fb.build_encoder("medsiglip", want_patch=True, chunk_size=2)
    out = enc.encode_slices(torch.randint(0, 256, (2, 300, 260), dtype=torch.uint8))
    assert out["pooled"].shape == (2, 1152)
    assert out["patch"].shape == (2, 1152, 32, 32)
