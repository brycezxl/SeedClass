import torch
import torch.nn.functional as f
from torch import nn


def f1_loss(predict, label):
    predict = f.softmax(predict, dim=1)
    predict = torch.clamp(predict * (1 - label), min=5e-3) + predict * label
    tp = predict * label
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (label.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))

    return -torch.mean(f1) + torch.tensor(1)


class F1Score(object):
    def __init__(self):
        self.tp = torch.zeros(374).cuda()
        self.fp = torch.zeros(374).cuda()
        self.fn = torch.zeros(374).cuda()
        self.best_f1 = 0

    def update(self, predict, label):
        predict_total = torch.zeros(374)
        label_total = torch.zeros(374)
        for i in range(len(predict)):
            for j in range(predict[i].size(0)):
                predict_total += torch.argmax(predict[i][j, :], dim=-1)
                label_total[label[i][j]] += 1

        self.tp += torch.sum(label_total * predict_total, dim=0)
        self.fp += torch.sum((torch.tensor(1) - label_total) * predict_total, dim=0)
        self.fn += torch.sum(label_total * (torch.tensor(1) - predict_total), dim=0)

    def get_f1(self):
        p = self.tp / (self.tp + self.fp)
        r = self.tp / (self.tp + self.fn)
        f1 = 2 * p * r / (p + r)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        return torch.mean(f1)

    def best(self):
        f1_ = self.get_f1()
        if f1_ > self.best_f1:
            self.best_f1 = f1_
            return True
        return False

    def reset(self):
        self.tp = torch.zeros(374).cuda()
        self.fp = torch.zeros(374).cuda()
        self.fn = torch.zeros(374).cuda()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                                             常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha < 1   # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, predicts, labels):
        """
        focal_loss损失计算
        :param predicts:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        predicts = predicts.view(-1, predicts.size(-1))
        self.alpha = self.alpha.to(predicts.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果
        preds_softmax = f.softmax(predicts, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((torch.tensor(1) - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
