from torch import nn
from models_.embedding import Embedding
import torch
import torch.nn.functional as f


class AlexNet(nn.Module):
    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.emb = Embedding(args)
        self.args = args
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 374),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, cd):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = f.softmax(x, dim=1)
        return x
