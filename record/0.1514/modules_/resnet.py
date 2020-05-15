import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, pre_trained):
        super(ResNet, self).__init__()
        model = models.resnet101(pretrained=pre_trained)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.MaxPool2d(7, 7)
        )

    def forward(self, inputs):
        inputs = self.features(inputs)
        return inputs
