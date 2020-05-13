from torch import nn
import torch.nn.functional as f


class Embedding(nn.Module):
    def __init__(self, args):
        super(Embedding, self).__init__()
        self.args = args
        self.emb = nn.Embedding(50, args.d_model)

    def forward(self, x):
        x = self.emb(x)
        x = f.dropout(x, self.args.dropout, self.training)
        return x
