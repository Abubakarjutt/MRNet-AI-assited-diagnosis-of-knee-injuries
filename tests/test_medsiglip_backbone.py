import torch

import medical_encoders


def test_encoder_forward_shape_and_frozen(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    assert encoder.feature_dim == 1152
    assert all(not p.requires_grad for p in encoder.parameters())

    out = encoder(torch.randn(5, 3, 448, 448))
    assert out.shape == (5, 1152)
    assert out.dtype == torch.float32


def test_encoder_train_keeps_tower_in_eval(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    encoder.train()
    assert encoder.training is True
    assert encoder.tower.training is False


def test_encoder_chunk_size_is_invariant(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder(chunk_size=64)
    x = torch.randn(10, 3, 448, 448)

    encoder.chunk_size = 3
    small = encoder(x)
    encoder.chunk_size = 64
    big = encoder(x)

    assert torch.allclose(small, big, atol=1e-5)


def test_encoder_state_dict_is_empty(stub_medsiglip):
    encoder = medical_encoders.MedSigLIPEncoder()
    assert dict(encoder.state_dict()) == {}
