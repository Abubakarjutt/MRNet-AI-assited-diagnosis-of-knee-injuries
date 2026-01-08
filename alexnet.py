import torch
import torch.nn as nn
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(1000, 3)
        
        # Extract features from the pretrained model
        self.features = self.pretrained_model.features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6 * 3, 4096), # Adjusted input size for concatenated features
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        
        # Re-implementing the specific architecture from the original code
        # The original code had a custom 'net' sequential block that seemed to replicate AlexNet features
        # but with different dropout/pooling. 
        # To match the previous logic exactly but cleaner:
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=12544, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=3, bias=True),
        )

        self.init_weights()

    def init_weights(self):
        # Initialize with pretrained AlexNet weights where possible
        self.net[0].weight = self.pretrained_model.features[0].weight
        self.net[0].bias = self.pretrained_model.features[0].bias
        
        self.net[4].weight = self.pretrained_model.features[3].weight
        self.net[4].bias = self.pretrained_model.features[3].bias
        
        self.net[8].weight = self.pretrained_model.features[6].weight
        self.net[8].bias = self.pretrained_model.features[6].bias
        
        self.net[11].weight = self.pretrained_model.features[8].weight
        self.net[11].bias = self.pretrained_model.features[8].bias
        
        self.net[14].weight = self.pretrained_model.features[10].weight
        self.net[14].bias = self.pretrained_model.features[10].bias

    def forward(self, sagittal, coronal, axial):
        # Squeeze the extra dimension (batch_size=1)
        sagittal = torch.squeeze(sagittal, dim=0)
        coronal = torch.squeeze(coronal, dim=0)
        axial = torch.squeeze(axial, dim=0)

        # Concatenate along the batch dimension (which is actually the stack depth here)
        # The original code concatenated inputs first, then passed through the net
        concatenated_features = torch.cat((sagittal, coronal, axial), 0)
        
        features = self.net(concatenated_features)
        features = features.view(features.size(0), -1)
        
        # Max pooling over the "slices" (batch dimension of the features)
        flattened_features = torch.max(features, 0, keepdim=True)[0]
        
        output = self.classifier(flattened_features)
        
        return output
