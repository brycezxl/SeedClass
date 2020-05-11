import torch.nn as nn
import torch.nn.functional as f
from efficientnet_pytorch import EfficientNet


class Efficient(nn.Module):
    def __init__(self, num_classes):
        super(Efficient, self).__init__()
        self.efficient = EfficientNet.from_name('efficientnet-b7')
        feature = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(in_features=feature, out_features=num_classes, bias=True)

    def forward(self, x, words, mask):
        x = self.efficient(x)
        return x
