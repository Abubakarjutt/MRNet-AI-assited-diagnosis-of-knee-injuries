import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

class MRNetViT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Use pretrained ViT-B/16
        self.pretrained_model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # The head in torchvision ViT is a Sequential block.
        # We want the representation before the head.
        # For ViT, setting heads to Identity exposes the class token representation (768 dim)
        self.pretrained_model.heads = nn.Identity()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, num_classes)
        )

    def forward(self, sagittal, coronal, axial):
        # Inputs are (Batch=1, Slices, 3, 224, 224)
        # Squeeze the batch dimension
        sagittal = torch.squeeze(sagittal, dim=0)
        coronal = torch.squeeze(coronal, dim=0)
        axial = torch.squeeze(axial, dim=0)

        # Concatenate along the batch dimension (which represents slices here)
        # Shape becomes (Total_Slices, 3, 224, 224)
        x = torch.cat((sagittal, coronal, axial), 0)
        
        # Pass through ViT backbone
        # Output shape: (Total_Slices, 768)
        x = self.pretrained_model(x)
        
        # Max pooling over all slices (planes combined)
        # Shape: (1, 768)
        x = torch.max(x, 0, keepdim=True)[0]
        
        # Classify
        output = self.classifier(x)
        
        return output
