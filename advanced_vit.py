import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_H_14_Weights, ViT_L_16_Weights


def _build_vit_backbone(model_name, pretrained):
    if model_name == "vit_b_16":
        backbone = models.vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        embedding_dim = 768
    elif model_name == "vit_l_16":
        backbone = models.vit_l_16(
            weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        embedding_dim = 1024
    elif model_name == "vit_h_14":
        backbone = models.vit_h_14(
            weights=ViT_H_14_Weights.IMAGENET1K_V1 if pretrained else None
        )
        embedding_dim = 1280
    else:
        raise ValueError("Unsupported model name. Choose from: vit_b_16, vit_l_16, vit_h_14")

    if hasattr(backbone, "heads"):
        backbone.heads = nn.Identity()
    elif hasattr(backbone, "classifier"):
        backbone.classifier = nn.Identity()

    return backbone, embedding_dim


class _PlaneEncoder(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()
        self.backbone, self.embedding_dim = _build_vit_backbone(model_name, pretrained)

    def forward(self, plane_tensor):
        batch_size, slices, channels, height, width = plane_tensor.shape
        flat_inputs = plane_tensor.reshape(batch_size * slices, channels, height, width)
        slice_features = self.backbone(flat_inputs)
        slice_features = slice_features.reshape(batch_size, slices, self.embedding_dim)
        return torch.max(slice_features, dim=1).values


class AdvancedMRNetViT(nn.Module):
    def __init__(self, num_classes=3, model_name="vit_b_16", pretrained=True):
        super().__init__()
        self.encoder = _PlaneEncoder(model_name=model_name, pretrained=pretrained)
        fused_dim = self.encoder.embedding_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, sagittal, coronal, axial):
        sagittal_features = self.encoder(sagittal)
        coronal_features = self.encoder(coronal)
        axial_features = self.encoder(axial)
        combined_features = torch.cat(
            [sagittal_features, coronal_features, axial_features], dim=1
        )
        return self.classifier(combined_features)


class MultiScaleMRNetViT(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.vit_b_16 = _PlaneEncoder(model_name="vit_b_16", pretrained=pretrained)
        self.vit_l_16 = _PlaneEncoder(model_name="vit_l_16", pretrained=pretrained)
        fused_dim = (self.vit_b_16.embedding_dim + self.vit_l_16.embedding_dim) * 3
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, num_classes),
        )

    def forward(self, sagittal, coronal, axial):
        combined_features = torch.cat(
            [
                self.vit_b_16(sagittal),
                self.vit_b_16(coronal),
                self.vit_b_16(axial),
                self.vit_l_16(sagittal),
                self.vit_l_16(coronal),
                self.vit_l_16(axial),
            ],
            dim=1,
        )
        return self.feature_fusion(combined_features)
