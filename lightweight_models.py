import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)


def _strip_classifier(backbone, name):
    if name == "resnet18":
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == "mobilenet_v3_small":
        feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return backbone, feature_dim


def build_backbone(name, pretrained=True):
    if name == "resnet18":
        backbone = models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif name == "mobilenet_v3_small":
        backbone = models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif name == "efficientnet_b0":
        backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError(
            "Unsupported backbone. Choose from: resnet18, mobilenet_v3_small, efficientnet_b0"
        )

    return _strip_classifier(backbone, name)


class FastMRNet(nn.Module):
    def __init__(self, backbone_name="resnet18", num_classes=3, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, self.feature_dim = build_backbone(backbone_name, pretrained)
        fused_dim = self.feature_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, max(256, self.feature_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(256, self.feature_dim // 2), num_classes),
        )

    def _encode_planes(self, plane_tensors):
        batch_size = plane_tensors[0].shape[0]
        flat_inputs = torch.cat(
            [plane.reshape(-1, *plane.shape[-3:]) for plane in plane_tensors],
            dim=0,
        )
        if flat_inputs.device.type in {"cuda", "mps"}:
            flat_inputs = flat_inputs.contiguous(memory_format=torch.channels_last)
        encoded = self.encoder(flat_inputs)

        pooled_features = []
        offset = 0
        for plane in plane_tensors:
            slice_count = plane.shape[1]
            plane_features = encoded[offset: offset + batch_size * slice_count]
            plane_features = plane_features.reshape(batch_size, slice_count, self.feature_dim)
            pooled_features.append(torch.amax(plane_features, dim=1))
            offset += batch_size * slice_count

        return pooled_features

    def forward(self, sagittal, coronal, axial):
        sagittal_features, coronal_features, axial_features = self._encode_planes(
            (sagittal, coronal, axial)
        )
        fused = torch.cat(
            [sagittal_features, coronal_features, axial_features],
            dim=1,
        )
        return self.classifier(fused)
