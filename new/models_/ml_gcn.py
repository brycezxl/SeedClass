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
        # self.conv = AlexNet()
        self.gc1_1 = GraphConvolution(in_channel, 1024)
        self.gc1_2 = GraphConvolution(in_channel, 1024)
        self.gc2_1 = GraphConvolution(1024, 2048)
        self.gc2_2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.image_fc = nn.Linear(2048, in_channel)
        self.fc = nn.Linear(in_channel * 3, in_channel).double()
        self.cd_emb = nn.Embedding(50, in_channel)

        # self.adj_all = Parameter(load_adj(num_classes, t, adj_path), requires_grad=True)
        self.adj_cd = load_cd_adj(num_classes, t).cuda()

        self.label_mask = load_label_mask(mask_path)
        self.words = load_emb(emb_path)

        self.attn = Attention(args).double()
        # self.out = nn.Linear(2048, 1).double()

    def forward(self, images, cds):
        # images = self.conv(images)
        # images = images.view(images.size(0), -1)

        label_mask = self.label_mask[cds].unsqueeze(-1)
        x = self.words * label_mask.ceil()

        x = self.fc(torch.cat((
            x,
            x * self.image_fc(images).unsqueeze(-2),
            x * self.cd_emb(cds).unsqueeze(-2),
        ), dim=-1))

        adj_cd = self.adj_cd[cds]
        adj_cd = gen_cd_adj(adj_cd)
        adj_emb = torch.matmul(x, x.transpose(-1, -2))
        adj_emb = torch.clamp(adj_emb / torch.max(adj_emb, -1)[0].unsqueeze(-1), min=0, max=1)
        adj_emb = gen_cd_adj(adj_emb)

        x = (self.gc1_1(x, adj_cd), self.gc1_2(x, adj_emb))
        x = (self.relu(x[0]), self.relu(x[1]))
        x = (self.gc2_1(x[0], adj_cd), self.gc2_2(x[1], adj_emb))
        x = (x[0] + x[1]) / 2

        # x = x * images.unsqueeze(1).double()
        # x = self.out(x)

        # x_ = self.attn(images.unsqueeze(1).double(), x, x)
        # x_ = torch.mean((x_ - images.unsqueeze(1).double()) ** 2 / 2048)

        x = torch.matmul(x, images.unsqueeze(-1).double())
        x = x * label_mask.ceil()
        x[torch.where(label_mask == 0)] += -1e10
        x = torch.sigmoid(x.squeeze(-1))

        # return x, x_
        return x


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
