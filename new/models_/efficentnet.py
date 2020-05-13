import torch.nn as nn
import torch.nn.functional as f
import efficientnet_pytorch


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b0')
        feature = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features=feature, out_features=num_classes, bias=True)

    def forward(self, x, words, mask):
        x = self.model(x)
        return x
