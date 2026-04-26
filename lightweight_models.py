import math

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


class AttentionMILPool(nn.Module):
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, slice_features):
        logits = self.attention(slice_features)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights * slice_features, dim=1)


class GeMPool1D(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, slice_features):
        positive = torch.clamp(torch.nn.functional.gelu(slice_features), min=self.eps)
        pooled = positive.pow(self.p).mean(dim=1).pow(1.0 / self.p)
        return pooled


class SEFusionGate(nn.Module):
    def __init__(self, fused_dim, reduction=8):
        super().__init__()
        hidden_dim = max(fused_dim // reduction, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, fused):
        return fused * self.net(fused)


class PlaneAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        hidden_dim = max(feature_dim // 2, 32)
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, plane_features):
        planes = torch.stack(plane_features, dim=1)
        weights = torch.softmax(self.scorer(planes), dim=1)
        attended = planes * weights
        return attended.reshape(attended.shape[0], -1)


class PlaneTransformerFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=max(1, min(num_heads, feature_dim)),
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, plane_features):
        planes = torch.stack(plane_features, dim=1)
        attn_input = self.pre_norm(planes)
        attended, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        planes = planes + attended
        planes = planes + self.ffn(planes)
        return self.out_norm(planes).reshape(planes.shape[0], -1)


def build_pooler(pooling, feature_dim):
    if pooling == "attention":
        return AttentionMILPool(feature_dim=feature_dim)
    if pooling == "gem":
        return GeMPool1D()
    return None


def simple_pool(slice_features, pooling):
    if pooling == "max":
        return torch.amax(slice_features, dim=1)
    if pooling == "mean":
        return torch.mean(slice_features, dim=1)
    if pooling == "lse":
        return torch.logsumexp(slice_features, dim=1) - math.log(slice_features.shape[1])
    raise ValueError(f"Unsupported pooling mode: {pooling}")


class FastMRNet(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        num_classes=3,
        pretrained=True,
        dropout=0.2,
        pooling="max",
        projection_dim=0,
        hidden_dim=256,
        fusion_depth=2,
        fusion_gate="none",
        plane_fusion="concat",
        plane_transformer_heads=4,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pooling = pooling
        self.fusion_gate_name = fusion_gate
        self.encoder, self.feature_dim = build_backbone(backbone_name, pretrained)
        self.projection_dim = projection_dim if projection_dim > 0 else self.feature_dim
        self.hidden_dim = hidden_dim
        self.fusion_depth = fusion_depth
        self.plane_fusion_name = plane_fusion
        self.slice_pooler = build_pooler(pooling, self.feature_dim)

        # Simpler projection when projection_dim matches feature_dim
        self.plane_projection = (
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.projection_dim),
                nn.GELU(),
            )
            if self.projection_dim != self.feature_dim
            else nn.Identity()
        )

        fused_dim = self.projection_dim * 3
        self.plane_fusion = (
            PlaneAttentionFusion(self.projection_dim)
            if plane_fusion == "plane_attention"
            else PlaneTransformerFusion(self.projection_dim, num_heads=plane_transformer_heads, dropout=dropout)
            if plane_fusion == "plane_transformer"
            else None
        )
        self.fusion_gate = SEFusionGate(fused_dim) if fusion_gate == "se" else nn.Identity()
        self.classifier = self._build_classifier(
            fused_dim=fused_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_depth=fusion_depth,
        )

    def _build_classifier(self, fused_dim, num_classes, hidden_dim, dropout, fusion_depth):
        layers = [nn.LayerNorm(fused_dim), nn.Dropout(dropout)]

        if fusion_depth <= 1:
            layers.append(nn.Linear(fused_dim, num_classes))
            return nn.Sequential(*layers)

        in_features = fused_dim
        for _ in range(fusion_depth - 1):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        return nn.Sequential(*layers)

    def _pool_slices(self, slice_features):
        if self.slice_pooler is not None:
            return self.slice_pooler(slice_features)
        return simple_pool(slice_features, self.pooling)

    def _encode_planes(self, plane_tensors):
        batch_size = plane_tensors[0].shape[0]
        flat_inputs = torch.cat(
            [plane.reshape(-1, *plane.shape[-3:]) for plane in plane_tensors],
            dim=0,
        )
        if flat_inputs.device.type == "cuda":
            flat_inputs = flat_inputs.contiguous(memory_format=torch.channels_last)
        encoded = self.encoder(flat_inputs)

        pooled_features = []
        offset = 0
        for plane in plane_tensors:
            slice_count = plane.shape[1]
            plane_features = encoded[offset: offset + batch_size * slice_count]
            plane_features = plane_features.reshape(batch_size, slice_count, self.feature_dim)
            pooled = self._pool_slices(plane_features)
            pooled_features.append(self.plane_projection(pooled))
            offset += batch_size * slice_count

        return pooled_features

    def forward(self, sagittal, coronal, axial):
        sagittal_features, coronal_features, axial_features = self._encode_planes(
            (sagittal, coronal, axial)
        )
        plane_features = [sagittal_features, coronal_features, axial_features]
        if self.plane_fusion is not None:
            fused = self.plane_fusion(plane_features)
        else:
            fused = torch.cat(plane_features, dim=1)
        fused = self.fusion_gate(fused)
        return self.classifier(fused)
