import torch
import pytest
import feature_bank as fb


def test_fake_encoder_contract_pooled_only():
    enc = fb.build_encoder("fake", want_patch=False)
    out = enc.encode_slices(torch.randint(0, 256, (5, 40, 50), dtype=torch.uint8))
    assert out["pooled"].shape == (5, 16)
    assert out["pooled"].dtype == torch.float32
    assert "patch" not in out


def test_fake_encoder_contract_with_patch():
    enc = fb.build_encoder("fake", want_patch=True)
    out = enc.encode_slices(torch.randint(0, 256, (3, 30, 30), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 16)
    assert out["patch"].shape == (3, 8, 4, 4)


def test_fake_encoder_is_deterministic():
    enc = fb.build_encoder("fake", want_patch=True)
    x = torch.randint(0, 256, (4, 24, 24), dtype=torch.uint8)
    a = enc.encode_slices(x)
    b = enc.encode_slices(x)
    torch.testing.assert_close(a["pooled"], b["pooled"])
    torch.testing.assert_close(a["patch"], b["patch"])


def test_fake_encoder_respects_chunking():
    enc = fb.build_encoder("fake", want_patch=False, chunk_size=2)
    x = torch.randint(0, 256, (7, 16, 16), dtype=torch.uint8)
    whole = fb.build_encoder("fake", want_patch=False, chunk_size=64).encode_slices(x)
    chunked = enc.encode_slices(x)
    torch.testing.assert_close(whole["pooled"], chunked["pooled"])


def test_build_encoder_unknown_name_raises():
    with pytest.raises(ValueError):
        fb.build_encoder("not-an-encoder")


import types
import feature_bank_medsiglip as fbm


def test_medsiglip_bank_encoder_with_stub_tower(monkeypatch):
    class _StubTower:
        config = types.SimpleNamespace(hidden_size=1152)
        def __call__(self, pixel_values):
            n = pixel_values.shape[0]
            return types.SimpleNamespace(
                pooler_output=torch.randn(n, 1152),
                last_hidden_state=torch.randn(n, 32 * 32, 1152),
            )
        def parameters(self): return iter(())
        def eval(self): return self

    class _StubFull:
        vision_model = _StubTower()
        def get_image_features(self, pixel_values):
            return torch.randn(pixel_values.shape[0], 1152)

    monkeypatch.setattr("medical_encoders.AutoModel",
                        types.SimpleNamespace(from_pretrained=lambda *a, **k: _StubFull()))
    enc = fbm.MedSigLIPBankEncoder(chunk_size=2, want_patch=True)
    out = enc.encode_slices(torch.randint(0, 256, (3, 100, 90), dtype=torch.uint8))
    assert out["pooled"].shape == (3, 1152)
    assert out["patch"].shape == (3, 1152, 32, 32)
