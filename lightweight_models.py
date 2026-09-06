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
    elif name == "medsiglip":
        from medical_encoders import MedSigLIPEncoder

        encoder = MedSigLIPEncoder()  # `pretrained` intentionally ignored
        return encoder, encoder.feature_dim
    else:
        raise ValueError(
            "Unsupported backbone. Choose from: resnet18, mobilenet_v3_small, "
            "efficientnet_b0, medsiglip"
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


class _PlaneFuse(nn.Module):
    """PlaneAttentionFusion (returns [B, n_planes*d]) followed by a projection back
    to [B, d] so head shapes match the spec's sketches."""

    def __init__(self, d_model, n_planes):
        super().__init__()
        self.fuse = PlaneAttentionFusion(d_model)
        self.proj = nn.Linear(d_model * n_planes, d_model)

    def forward(self, plane_feats):                       # list of [B, d]
        return self.proj(self.fuse(plane_feats))


class _PooledVec(nn.Module):
    """GeM slice-pool per (enc,plane) -> mean over encoders -> plane fusion -> [B, d_model]. No classifier."""

    def __init__(self, encoders, planes, d_model):
        super().__init__()
        self.encoders = list(encoders)
        self.planes = list(planes)
        self.pool = nn.ModuleDict({e: GeMPool1D() for e in self.encoders})
        self.fuse = _PlaneFuse(d_model, len(self.planes))

    def forward(self, projected):                         # projected[enc][plane] = [B,S,d]
        plane_feats = []
        for plane in self.planes:
            per_enc = [self.pool[e](projected[e][plane]) for e in self.encoders]  # [B,d] each
            plane_feats.append(torch.stack(per_enc, dim=0).mean(dim=0))
        return self.fuse(plane_feats)                     # [B, d_model]


class _PooledHead(nn.Module):
    """_PooledVec + MLP -> [B] logit."""

    def __init__(self, encoders, planes, d_model, head_hidden, dropout):
        super().__init__()
        self.vec = _PooledVec(encoders, planes, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, projected):
        return self.mlp(self.vec(projected)).squeeze(-1)  # [B]


class _PyramidPath(nn.Module):
    """Per-slice depthwise-separable conv at strides {1,2} -> GAP -> concat ->
    Linear(d_model) -> attention slice-pool -> mean over enc -> plane fusion."""

    def __init__(self, patch_dims, planes, d_model):
        super().__init__()
        self.encoders = list(patch_dims)
        self.planes = list(planes)
        self.blocks = nn.ModuleDict()
        for e, dp in patch_dims.items():
            self.blocks[e] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dp, dp, 3, stride=s, padding=1, groups=dp),
                    nn.Conv2d(dp, d_model, 1), nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                ) for s in (1, 2)
            ])
        self.merge = nn.ModuleDict({e: nn.Linear(2 * d_model, d_model) for e in self.encoders})
        self.slice_pool = nn.ModuleDict({e: AttentionMILPool(d_model) for e in self.encoders})
        self.fuse = _PlaneFuse(d_model, len(self.planes))

    def forward(self, patch):                             # patch[enc][plane] = [B,Sm,Dp,h,w]
        plane_feats = []
        for plane in self.planes:
            per_enc = []
            for e in self.encoders:
                x = patch[e][plane]
                b, sm, dp, h, w = x.shape
                x = x.reshape(b * sm, dp, h, w)
                scales = [blk(x).flatten(1) for blk in self.blocks[e]]      # [b*sm, d_model] each
                slc = self.merge[e](torch.cat(scales, dim=1)).reshape(b, sm, -1)
                per_enc.append(self.slice_pool[e](slc))                     # [B,d]
            plane_feats.append(torch.stack(per_enc, dim=0).mean(dim=0))
        return self.fuse(plane_feats)                                       # [B,d]


class FeatBankMRNet(nn.Module):
    consumes_feature_batch = True

    _HEAD_PLANES = {
        "abnormal": ("sagittal", "coronal", "axial"),
        "acl": ("sagittal", "coronal"),
        "meniscus": ("sagittal", "coronal"),
    }

    def __init__(self, encoder_dims, patch_dims, *, d_model=256, dropout=0.15,
                 head_hidden=128, want_patch_encoders=None):
        super().__init__()
        self.encoders = list(encoder_dims)
        self.d_model = int(d_model)
        if want_patch_encoders is None:
            want_patch_encoders = tuple(patch_dims)
        self.patch_encoders = list(want_patch_encoders)

        self.proj = nn.ModuleDict({
            f"{e}::{p}": nn.Sequential(
                nn.LayerNorm(encoder_dims[e]), nn.Linear(encoder_dims[e], d_model), nn.GELU())
            for e in self.encoders for p in ("sagittal", "coronal", "axial")
        })
        self.abnormal_head = _PooledHead(self.encoders, self._HEAD_PLANES["abnormal"],
                                         d_model, head_hidden, dropout)
        self.acl_head = _PooledHead(self.encoders, self._HEAD_PLANES["acl"],
                                    d_model, head_hidden, dropout)
        self.meniscus_pyramid = None
        if self.patch_encoders:
            self.meniscus_vec = _PooledVec(self.encoders, self._HEAD_PLANES["meniscus"], d_model)
            self.meniscus_pyramid = _PyramidPath(
                {e: patch_dims[e] for e in self.patch_encoders},
                self._HEAD_PLANES["meniscus"], d_model)
            self.meniscus_mlp = nn.Sequential(
                nn.LayerNorm(2 * d_model), nn.Dropout(dropout),
                nn.Linear(2 * d_model, head_hidden), nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.meniscus_head = _PooledHead(self.encoders, self._HEAD_PLANES["meniscus"],
                                             d_model, head_hidden, dropout)

    def _project(self, pooled):
        return {e: {p: self.proj[f"{e}::{p}"](pooled[e][p])
                    for p in ("sagittal", "coronal", "axial")}
                for e in self.encoders}

    def forward(self, payload):
        projected = self._project(payload["pooled"])
        abnormal = self.abnormal_head(projected)
        acl = self.acl_head(projected)

        if self.meniscus_pyramid is not None:
            pooled_vec = self.meniscus_vec(projected)              # [B,d]
            pyr_vec = self.meniscus_pyramid(payload["patch"])      # [B,d]
            meniscus = self.meniscus_mlp(torch.cat([pooled_vec, pyr_vec], dim=1)).squeeze(-1)
        else:
            meniscus = self.meniscus_head(projected)

        return torch.stack([abnormal, acl, meniscus], dim=1)     # [B,3] TASKS order
