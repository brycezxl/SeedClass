import torch
from torch import nn
from torch.nn import functional as f


class CrossAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t_q = nn.Linear(args.d_model, args.d_model)
        self.t_k = nn.Linear(args.d_model, args.d_model)
        self.t_v = nn.Linear(args.d_model, args.d_model)
        self.g = nn.Linear(2 * args.d_model, args.d_model)

        nn.init.xavier_normal_(self.t_q.weight)
        nn.init.xavier_normal_(self.t_k.weight)
        nn.init.xavier_normal_(self.t_v.weight)
        nn.init.xavier_normal_(self.g.weight)

    def forward(self, q, k, v, mask=None, add=True):
        q0 = q
        q = self.t_q(q)
        k = self.t_k(k)
        v = self.t_v(v)

        w = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        w = f.softmax(w, dim=2)
        x = torch.matmul(w, v)

        gate = self.g(torch.cat((q, x), -1))
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x * torch.sigmoid(gate)
        if add:
            x = x + q0
        return x


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, q, k, v, mask=None):
        w = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        w = f.softmax(w, dim=2)
        x = torch.matmul(w, v)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class CoAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.t_q = nn.Linear(args.d_model, args.d_model)
        self.t_k = nn.Linear(args.d_model, args.d_model)
        self.t_v = nn.Linear(args.d_model, args.d_model)

    def forward(self, x1, x2, mask=None):
        q = self.t_q(x1)
        k = self.t_k(x2)
        v2 = self.t_v(x2)
        v1 = self.t_v(x1)

        a = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if mask is not None:
            m = (1 - mask) * -1e30
            a = (a.transpose(-1, -2) + m.unsqueeze(-1)).transpose(-1, -2)
        a2_ = f.softmax(a, dim=2)
        x2_ = torch.matmul(a2_, v2)
        a1_ = f.softmax(a.transpose(-1, -2), dim=2)
        x1_ = torch.matmul(a1_, v1)
        if mask is not None:
            x1_ = x1_ * mask.unsqueeze(-1)
        x1_ = torch.matmul(a2_, x1_)
        return x1_, x2_
