import torch
from types import SimpleNamespace

import train
import utils


def test_siglip_constants_shape_and_value():
    for tensor in (utils.SIGLIP_MEAN, utils.SIGLIP_STD):
        assert tensor.shape == (1, 1, 3, 1, 1)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.full_like(tensor, 0.5))


def test_prepare_volume_batch_default_matches_manual_imagenet():
    """Golden byte-identity guard for the default (non-medsiglip) path.

    Seeded non-constant input at a genuinely different resize (50 -> 32), compared
    against an inlined copy of the pre-clamp implementation. If the F1 clamp (or any
    future change) perturbs the default ImageNet/bilinear/antialias=False path, this
    fails.
    """
    torch.manual_seed(1)
    volume = (torch.rand(1, 5, 50, 50) * 255.0)
    # default path = ImageNet mean/std, bilinear, antialias=False
    out = utils.prepare_volume_batch(volume, device=torch.device("cpu"), image_size=32)
    manual = volume.div(255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1)
    flat = manual.reshape(5, 3, 50, 50)
    flat = torch.nn.functional.interpolate(
        flat, size=(32, 32), mode="bilinear", align_corners=False, antialias=False
    ).reshape(1, 5, 3, 32, 32)
    flat = (flat - utils.IMAGENET_MEAN) / utils.IMAGENET_STD
    assert torch.equal(out, flat)


def test_prepare_volume_batch_siglip_spec_maps_to_unit_range():
    volume = torch.zeros((1, 3, 8, 8))
    volume[..., :, :] = 255.0
    out = utils.prepare_volume_batch(
        volume,
        device=torch.device("cpu"),
        image_size=16,
        mean=utils.SIGLIP_MEAN,
        std=utils.SIGLIP_STD,
        interp_mode="bicubic",
        antialias=True,
    )
    assert out.shape == (1, 3, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert out.max().item() <= 1.0 + 1e-4
    assert out.min().item() >= -1.0 - 1e-4


def test_prepare_volume_batch_siglip_clamps_bicubic_overshoot():
    torch.manual_seed(0)
    volume = (torch.rand(1, 6, 40, 40) * 255.0)
    out = utils.prepare_volume_batch(
        volume,
        device=torch.device("cpu"),
        image_size=48,
        mean=utils.SIGLIP_MEAN,
        std=utils.SIGLIP_STD,
        interp_mode="bicubic",
        antialias=True,
    )
    # bicubic ringing on high-frequency data overshoots [0,1] pre-norm;
    # the clamp must pull it back so post-norm stays in [-1, 1].
    assert out.max().item() <= 1.0 + 1e-6
    assert out.min().item() >= -1.0 - 1e-6


def test_resolve_input_spec_medsiglip():
    args = SimpleNamespace(model_type="medsiglip", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 448
    assert spec["interp_mode"] == "bicubic"
    assert spec["antialias"] is True
    assert spec["mean"] is utils.SIGLIP_MEAN
    assert spec["std"] is utils.SIGLIP_STD


def test_resolve_input_spec_default_backbone():
    args = SimpleNamespace(model_type="mobilenet_v3_small", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 224
    assert spec["interp_mode"] == "bilinear"
    assert spec["antialias"] is False
    assert spec["mean"] is utils.IMAGENET_MEAN
    assert spec["std"] is utils.IMAGENET_STD


def test_prepare_inputs_uses_medsiglip_spec(mrnet_fixture):
    from dataloader import MRMultiPlaneDataset

    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=False)
    volumes, _label, _w, _id = dataset[0]
    args = SimpleNamespace(model_type="medsiglip", image_size=224, channels_last=0)

    sagittal, coronal, axial = train.prepare_inputs(volumes, torch.device("cpu"), args)
    for plane in (sagittal, coronal, axial):
        assert plane.shape[-2:] == (448, 448)
        assert torch.isfinite(plane).all()
        assert plane.min().item() >= -1.0 - 1e-6
        assert plane.max().item() <= 1.0 + 1e-6


def test_resolve_input_spec_medsiglip_warns_on_nondefault_image_size(capsys, monkeypatch):
    monkeypatch.setattr(train, "_warned_medsiglip_image_size", False)
    args = SimpleNamespace(model_type="medsiglip", image_size=384)
    train.resolve_input_spec(args)
    assert "ignored; forced to 448" in capsys.readouterr().err


def test_resolve_input_spec_medsiglip_silent_on_default(capsys, monkeypatch):
    monkeypatch.setattr(train, "_warned_medsiglip_image_size", False)
    args = SimpleNamespace(model_type="medsiglip", image_size=224)
    train.resolve_input_spec(args)
    assert capsys.readouterr().err == ""
