import torch
from torch import nn
import torch.nn.functional as F

from layers.batch_norm import FrozenBatchNorm2d

class ResNet152(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet152 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
        self.resnet152.fc = None
        self.resnet152.avgpool = None
        for param in self.resnet152.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = []
        x = self.resnet152.conv1(x)
        x = self.resnet152.bn1(x)
        x = self.resnet152.relu(x)
        x = self.resnet152.maxpool(x)

        x = self.resnet152.layer1(x)
        outputs.append(x)
        x = self.resnet152.layer2(x)
        outputs.append(x)
        x = self.resnet152.layer3(x)
        outputs.append(x)
        x = self.resnet152.layer4(x)
        outputs.append(x)

        return outputs
        