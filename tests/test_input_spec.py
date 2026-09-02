import torch

import utils


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
