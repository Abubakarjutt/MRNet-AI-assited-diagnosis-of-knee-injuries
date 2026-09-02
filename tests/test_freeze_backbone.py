from types import SimpleNamespace

import pytest

import train
from lightweight_models import FastMRNet


def _args(**overrides):
    base = dict(
        model_type="mobilenet_v3_small",
        freeze_backbone="auto",
        medsiglip_chunk=32,
        pretrained=1,
        dropout=0.2,
        pooling="max",
        projection_dim=0,
        hidden_dim=256,
        fusion_depth=2,
        fusion_gate="none",
        plane_fusion="concat",
        plane_transformer_heads=4,
        vit_model="vit_b_16",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_auto_leaves_cnn_backbone_trainable():
    model = FastMRNet(backbone_name="mobilenet_v3_small", num_classes=3, pretrained=False)
    train.apply_backbone_freezing(model, _args(model_type="mobilenet_v3_small"))
    assert all(p.requires_grad for p in model.encoder.parameters())


def test_auto_freezes_medsiglip_encoder(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(model, _args(model_type="medsiglip"))
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_explicit_one_freezes_cnn_backbone():
    model = FastMRNet(backbone_name="mobilenet_v3_small", num_classes=3, pretrained=False)
    train.apply_backbone_freezing(model, _args(model_type="mobilenet_v3_small", freeze_backbone="1"))
    assert all(not p.requires_grad for p in model.encoder.parameters())


def test_explicit_zero_on_medsiglip_warns_and_stays_frozen(stub_medsiglip, capsys):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(model, _args(model_type="medsiglip", freeze_backbone="0"))
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert "medsiglip" in capsys.readouterr().out.lower()


def test_build_model_sets_chunk_size(stub_medsiglip):
    model = train.build_model(_args(model_type="medsiglip", medsiglip_chunk=8))
    assert model.encoder.chunk_size == 8


def test_freeze_one_warns_when_no_encoder(capsys):
    from types import SimpleNamespace
    class _NoEncoder:
        pass
    train.apply_backbone_freezing(
        _NoEncoder(), SimpleNamespace(model_type="multiscale", freeze_backbone="1")
    )
    assert "has no .encoder to freeze" in capsys.readouterr().out


def test_argparser_accepts_medsiglip(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--prefix_name", "x", "--model_type", "medsiglip"])
    parsed = train.parse_arguments()
    assert parsed.model_type == "medsiglip"
    assert parsed.freeze_backbone == "auto"
    assert parsed.medsiglip_chunk == 32
