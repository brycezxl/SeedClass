from torch import nn
import torch.nn.functional as f


class Embedding(nn.Module):
    def __init__(self, d_model):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(51, d_model)

    def forward(self, x, mask):
        x = self.emb(x)
        x = x * mask.unsqueeze(-1)
        x = f.dropout(x, self.args.dropout, self.training)
        return x
