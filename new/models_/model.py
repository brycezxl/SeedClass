import math

from torch import nn
from torch.nn import Parameter
from models_.ml_gcn import MLGCN
from modules_ import *
from utils import *


class Model(nn.Module):
    def __init__(self, args, num_classes, t, pre_trained=True,
                 adj_path=None, mask_path=None, emb_path=None, in_channel=300):
        super(Model, self).__init__()
        self.args = args
        self.num_classes = num_classes
        if args.res:
            self.conv = ResNet(pre_trained=pre_trained)
        else:
            self.conv = AlexNet()
        self.net = MLGCN(args=self.args, num_classes=374, t=0.05, adj_path='../corel_5k/adj.pkl',
                         mask_path='../corel_5k/label_mask.pkl', emb_path='../corel_5k/word2vec.pkl',
                         pre_trained=self.args.pretrain)

    def forward(self, images, cds):
        images = self.conv(images)
        images = images.view(images.size(0), -1)
        outputs = self.net(images, cds)
        return outputs

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.conv.parameters(), 'lr': lr * lrp},
                {'params': self.net.parameters(), 'lr': lr},
                ]
