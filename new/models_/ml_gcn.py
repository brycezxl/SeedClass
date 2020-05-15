import math

from torch import nn
from torch.nn import Parameter

from modules_ import *
from utils import *


class MLGCN(nn.Module):
    def __init__(self, args, num_classes, t, pre_trained=True,
                 adj_path=None, mask_path=None, emb_path=None, in_channel=300):
        super(MLGCN, self).__init__()
        self.args = args
        self.num_classes = num_classes

        # self.conv = ResNet(pre_trained=pre_trained)
        self.conv = AlexNet()
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.image_fc = nn.Linear(2048, in_channel)
        self.fc = nn.Linear(in_channel * 3, in_channel).double()
        self.cd_emb = nn.Embedding(50, in_channel)

        self.adj = load_cd_adj(num_classes, t).cuda()

        self.label_mask = load_label_mask(mask_path)
        self.words = load_emb(emb_path)

    def forward(self, images, cds):
        images = self.conv(images)
        images = images.view(images.size(0), -1)

        label_mask = self.label_mask[cds].unsqueeze(-1)
        x = self.words * label_mask.ceil()

        x = self.fc(torch.cat((
            x,
            x * self.image_fc(images).unsqueeze(-2),
            x * self.cd_emb(cds).unsqueeze(-2),
        ), dim=-1))

        adj = self.adj[cds]
        adj = gen_cd_adj(adj)

        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = torch.matmul(x, images.unsqueeze(-1).double())
        x = x * label_mask
        x[torch.where(label_mask == 0)] += -1e10
        x = torch.sigmoid(x.squeeze(-1))

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
