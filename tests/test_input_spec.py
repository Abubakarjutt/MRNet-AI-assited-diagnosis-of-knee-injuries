import torch
from types import SimpleNamespace

import train
import utils
import utils as utils_module


def test_siglip_constants_shape_and_value():
    for tensor in (utils.SIGLIP_MEAN, utils.SIGLIP_STD):
        assert tensor.shape == (1, 1, 3, 1, 1)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.full_like(tensor, 0.5))


def test_prepare_volume_batch_default_matches_manual_imagenet():
    volume = torch.full((1, 4, 8, 8), 128.0)  # [B, slices, H, W]
    out = utils.prepare_volume_batch(volume, device=torch.device("cpu"), image_size=8)

    expected_r = (128.0 / 255.0 - 0.485) / 0.229
    assert out.shape == (1, 4, 3, 8, 8)
    assert torch.allclose(out[0, 0, 0], torch.full((8, 8), expected_r), atol=1e-5)


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


def test_resolve_input_spec_medsiglip():
    args = SimpleNamespace(model_type="medsiglip", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 448
    assert spec["interp_mode"] == "bicubic"
    assert spec["antialias"] is True
    assert spec["mean"] is utils_module.SIGLIP_MEAN
    assert spec["std"] is utils_module.SIGLIP_STD


def test_resolve_input_spec_default_backbone():
    args = SimpleNamespace(model_type="mobilenet_v3_small", image_size=224)
    spec = train.resolve_input_spec(args)
    assert spec["image_size"] == 224
    assert spec["interp_mode"] == "bilinear"
    assert spec["antialias"] is False
    assert spec["mean"] is utils_module.IMAGENET_MEAN
    assert spec["std"] is utils_module.IMAGENET_STD


def test_prepare_inputs_uses_medsiglip_spec(mrnet_fixture):
    from dataloader import MRMultiPlaneDataset

    dataset = MRMultiPlaneDataset(str(mrnet_fixture), train=False)
    volumes, _label, _w, _id = dataset[0]
    args = SimpleNamespace(model_type="medsiglip", image_size=224, channels_last=0)

    sagittal, coronal, axial = train.prepare_inputs(volumes, torch.device("cpu"), args)
    for plane in (sagittal, coronal, axial):
        assert plane.shape[-2:] == (448, 448)
        assert plane.min().item() >= -1.0 - 1e-3
        assert plane.max().item() <= 1.0 + 1e-3
