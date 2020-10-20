import torch
import torch.nn as nn
from torchvision import models
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)

model = models.alexnet(pretrained = True)

class AlexNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.4)
        self.pretrained_model = models.alexnet(pretrained=True)
        self.classify = nn.Linear(1000, 3)
        self.conv3d = nn.Conv3d(1, 64, (3, 3, 2), stride=(1, 1, 1), padding=(1, 1, 1))
        self.maxpool3d = nn.MaxPool3d(3, stride=2)




        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p = 0.1),
            #nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn. Linear(in_features=12544, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=3, bias=True),

        )

        # classifier is just a name for linear layers
        self.init_weights()
    def init_weights(self):
        self.net[0].weight = self.pretrained_model.features[0].weight
        self.net[4].weight = self.pretrained_model.features[3].weight
        self.net[8].weight = self.pretrained_model.features[6].weight
        self.net[11].weight = self.pretrained_model.features[8].weight
        self.net[14].weight = self.pretrained_model.features[10].weight

        self.net[0].bias = self.pretrained_model.features[0].bias
        self.net[4].bias = self.pretrained_model.features[3].bias
        self.net[8].bias = self.pretrained_model.features[6].bias
        self.net[11].bias = self.pretrained_model.features[8].bias
        self.net[14].bias = self.pretrained_model.features[10].bias




    def forward(self, sagittal, coronal, axial):
        sagittal = torch.squeeze(sagittal, dim=0)
        #sagittal = self.net(sagittal)
        #sagittal_features = self.net(sagittal)
        #sagittal_features = sagittal_features.view(sagittal_features.size(0), -1)

        coronal = torch.squeeze(coronal, dim=0)
        #sagittal = self.net(sagittal)
        #coronal_features = self.net(coronal)
        #coronal_features = coronal_features.view(coronal_features.size(0), -1)

        axial = torch.squeeze(axial, dim=0)
        #sagittal = self.net(sagittal)
        #axial_features = self.net(axial)
        #axial_features = axial_features.view(axial_features.size(0), -1)


        concatenated_features = torch.cat((sagittal, coronal, axial), 0)
        concatenated_features = self.net(concatenated_features)
        #concatenated_features = self.pooling_layer(concatenated_features)
        concatenated_features = concatenated_features.view(concatenated_features.size(0), -1)
        flattened_features = torch.max(concatenated_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)

        return output
