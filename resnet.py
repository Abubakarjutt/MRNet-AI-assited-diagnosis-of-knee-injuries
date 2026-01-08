import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class MRNetResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Use the default pre-trained weights
        self.pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the fully connected layer (fc)
        # ResNet50 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # We want everything up to avgpool
        modules = list(self.pretrained_model.children())[:-1]
        self.features = nn.Sequential(*modules)
        
        # ResNet50 feature dimension after avgpool is 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, sagittal, coronal, axial):
        # Squeeze the extra dimension (batch_size=1)
        sagittal = torch.squeeze(sagittal, dim=0)
        coronal = torch.squeeze(coronal, dim=0)
        axial = torch.squeeze(axial, dim=0)

        # Concatenate along the batch dimension (stack depth)
        # This allows us to process all 3 planes in parallel through the feature extractor
        x = torch.cat((sagittal, coronal, axial), 0)
        
        # Extract features
        x = self.features(x)
        
        # Flatten: Output of avgpool is (Batch, 2048, 1, 1), so we flatten to (Batch, 2048)
        x = x.view(x.size(0), -1)
        
        # Max pooling over the 3 planes (Sagittal, Coronal, Axial)
        # We take the max value for each feature across the 3 views
        x = torch.max(x, 0, keepdim=True)[0]
        
        # Classify
        output = self.classifier(x)
        
        return output
