import logging
import os
import random
from torch import nn
import numpy as np
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from modules_.f1_loss import f1_loss, F1Score
from get_dataset import DataSet
from models_ import *
from utils import AverageMeter, AnalysisMeter


class Runner:
    def __init__(self, args):
        self.args = args
        self._setup_seed(2020)
        self._build_loader()
        self._build_model()

    def _build_loader(self):
        train = DataSet(train=True)
        test = DataSet(train=False)
        self.classes = 50
        self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=4,
                                       shuffle=True, drop_last=True)
        self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=4,
                                      shuffle=False) if test else None

    @staticmethod
    def _setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _build_model(self):
        # self.model = Simple(num_classes=self.classes)
        self.model = AlexNet(num_classes=self.classes)

        self.device = torch.device('cuda:0')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # 5 6e-4:6681 3e-4:6755 1e-4:6855
        # 3 6753
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)
        self.f1_loss = f1_loss
        self.cross_entropy = torch.nn.CrossEntropyLoss()

        self.f1_score = F1Score()
        self.analysis_meter = AnalysisMeter()

    def train(self):
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            self._train_one_epoch(epoch)
            # path = os.path.join(self.args.model_saved_path, 'model-%d' % epoch)
            # torch.save(self.model.state_dict(), path)
            # logging.info('model saved to %s' % path)
            self.eval()
        # self.analysis_meter.show_difficult_image()
        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        for batch, (images, words, idx, mask, labels, words_label) in enumerate(self.train_loader, 1):

            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images, words, mask)
            loss = self.cross_entropy(outputs, torch.argmax(labels, dim=1))
            # loss += self.f1_loss(outputs, labels) / 4
            loss.backward(loss)
            self.optimizer.step()
            self.scheduler.step(epoch - 1 + batch / len(self.train_loader))
            loss_meter.update(loss.item())
        print('Epoch %2d | Train Loss: %.4f | ' % (
            epoch, loss_meter.avg,
        ), end='')
        loss_meter.reset()

    def eval(self):
        data_loaders = [self.test_loader]
        loss_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for batch, (images, words, idx, mask, labels, words_label) in enumerate(data_loader, 1):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, words, mask)
                    loss = self.cross_entropy(outputs, torch.argmax(labels, dim=1))
                    self.f1_score.update(outputs, labels)
                    loss_meter.update(loss.item())
                    # if self.f1_score.best_f1 > 0.6:
                    #     self.analysis_meter.difficult_image(outputs, labels, idx)
                print('Test Loss: %.4f | F1: %.4f | Best: %s' % (
                    loss_meter.avg, self.f1_score.get_f1(), 'True' if self.f1_score.best() else 'False'
                ))
                loss_meter.reset()
                self.f1_score.reset()
