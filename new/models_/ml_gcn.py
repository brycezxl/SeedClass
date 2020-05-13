import math

from torch import nn
from torch.nn import Parameter

from modules_ import *
from utils import *


class MLGCN(nn.Module):
    def __init__(self, num_classes, t, pre_trained=True, adj_file=None, in_channel=300):
        super(MLGCN, self).__init__()
        self.num_classes = num_classes

        # self.conv = ResNet(pre_trained=pre_trained)
        self.conv = AlexNet()

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        adj = gen_a(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(adj).double(), requires_grad=True)

    def forward(self, feature, inp):
        feature = self.conv(feature)
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj).squeeze(0)

        x = x.transpose(0, 1)
        x = torch.matmul(feature.double(), x)
        x = f.softmax(x, dim=1)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.ones(in_features, out_features, dtype=torch.double) / 10, requires_grad=True)
        self.bias = Parameter(torch.ones(1, 1, out_features, dtype=torch.double) / 10, requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
