from types import SimpleNamespace

import torch

import train
from lightweight_models import FastMRNet


def _complexity_args(**overrides):
    base = dict(
        model_type="mobilenet_v3_small",
        pooling="max",
        aug_policy="none",
        fusion_depth=1,
        fusion_gate="none",
        plane_fusion="concat",
        val_tta_mode="none",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_model_complexity_matches_legacy_formula():
    # legacy inline values, recomputed here as the reference
    def legacy(args):
        model_c = {"mobilenet_v3_small": 0.0, "resnet18": 0.2, "efficientnet_b0": 0.4}.get(
            args.model_type, 0.0
        )
        pooling_c = {"max": 0.0, "mean": 0.0, "lse": 0.2, "gem": 0.3, "attention": 0.5}.get(
            args.pooling, 0.3
        )
        aug_c = {
            "none": 0.0, "light": 0.05, "strong": 0.12,
            "knee_mri": 0.18, "knee_mri_plus": 0.24, "knee_mri_research": 0.30,
        }.get(args.aug_policy, 0.08)
        fusion_c = 0.2 * max(int(args.fusion_depth) - 1, 0)
        gate_c = 0.15 * (1 if args.fusion_gate == "se" else 0)
        plane_c = {"concat": 0.0, "plane_attention": 0.12, "plane_transformer": 0.22}.get(
            args.plane_fusion, 0.0
        )
        tta_c = 0.05 if args.val_tta_mode != "none" else 0.0
        return model_c + pooling_c + fusion_c + gate_c + aug_c + plane_c + tta_c

    for args in (
        _complexity_args(),
        _complexity_args(model_type="resnet18", pooling="gem", fusion_depth=3),
        _complexity_args(model_type="efficientnet_b0", aug_policy="knee_mri_plus",
                         fusion_gate="se", plane_fusion="plane_transformer",
                         val_tta_mode="flip"),
    ):
        assert train.model_complexity_score(args) == legacy(args)


def test_model_complexity_medsiglip_branch():
    assert train.model_complexity_score(_complexity_args(model_type="medsiglip")) == 0.6


def test_param_counts_reports_trainable_subset(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(
        model, SimpleNamespace(model_type="medsiglip", freeze_backbone="auto")
    )
    total_m, trainable_m = train.param_counts(model)
    assert trainable_m < total_m
    assert trainable_m > 0.0


def test_build_optimizer_only_sees_trainable_params(stub_medsiglip):
    model = FastMRNet(backbone_name="medsiglip", num_classes=3, pretrained=True)
    train.apply_backbone_freezing(
        model, SimpleNamespace(model_type="medsiglip", freeze_backbone="auto")
    )
    args = SimpleNamespace(lr=3e-4, weight_decay=1e-4)
    optimizer = train.build_optimizer(model, args)

    in_optimizer = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < in_optimizer == trainable < total
