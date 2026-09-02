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


import lightweight_models
from lightweight_models import FastMRNet


def test_build_backbone_medsiglip(stub_medsiglip):
    encoder, feature_dim = lightweight_models.build_backbone("medsiglip", pretrained=1)
    assert isinstance(encoder, medical_encoders.MedSigLIPEncoder)
    assert feature_dim == 1152


def test_fastmrnet_medsiglip_forward(stub_medsiglip):
    model = FastMRNet(
        backbone_name="medsiglip",
        num_classes=3,
        pretrained=True,
        pooling="gem",
        plane_fusion="plane_attention",
        fusion_depth=3,
        hidden_dim=192,
        dropout=0.15,
    )
    model.eval()

    planes = [torch.randn(1, s, 3, 448, 448) for s in (6, 5, 7)]
    logits = model(*planes)
    assert logits.shape == (1, 3)


def test_fastmrnet_medsiglip_checkpoint_excludes_tower(stub_medsiglip, tmp_path):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    state = model.state_dict()
    assert not any(key.startswith("encoder.tower") for key in state)

    path = tmp_path / "m.pth"
    torch.save(state, path)
    reloaded = torch.load(path, weights_only=True)
    result = model.load_state_dict(reloaded, strict=False)
    assert all("encoder.tower" not in key for key in result.unexpected_keys)


def test_encoder_actually_micro_batches(stub_medsiglip):
    import math

    encoder = medical_encoders.MedSigLIPEncoder(chunk_size=4)
    real_forward = encoder.tower.forward
    calls = []

    def counting_forward(*args, **kwargs):
        pv = kwargs.get("pixel_values", args[0] if args else None)
        calls.append(pv.shape[0])
        return real_forward(*args, **kwargs)

    encoder.tower.forward = counting_forward
    out = encoder(torch.randn(10, 3, 448, 448))

    assert out.shape == (10, 1152)
    assert len(calls) == math.ceil(10 / 4)      # 3 micro-batches, not 1
    assert max(calls) <= 4                        # no chunk exceeds chunk_size
    assert sum(calls) == 10                       # every sample processed once
