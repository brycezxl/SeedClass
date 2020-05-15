import logging
import random

import torch.backends.cudnn
from torch.utils.data import DataLoader
from torch import nn
from get_dataset import Corel
from models_ import *
from utils import *


class Runner:
    def __init__(self, args):
        self.args = args
        self._setup_seed(2020)
        self._build_loader()
        self._build_model()

    def _build_loader(self):
        train = Corel(train=True)
        test = Corel(train=False)
        self.classes = 374
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
        self.model = MLGCN(args=self.args, num_classes=374, t=0.05, adj_path='../corel_5k/adj.pkl',
                           mask_path='../corel_5k/label_mask.pkl', emb_path='../corel_5k/word2vec.pkl',
                           pre_trained=self.args.pretrain)

        self.device = torch.device('cuda:0')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5)

        self.f1_loss = f1_loss
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.f1_score_2 = F1Score2()
        self.f1_score_4 = F1Score1()
        self.f1_score_6 = F1Score05()
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
        for batch, ((images, cds), labels) in enumerate(self.train_loader, 1):

            images = images.to(self.device)
            labels = labels.to(self.device)
            cds = cds.to(self.device)
            self.optimizer.zero_grad()
            # outputs, cd_ = self.model(images, cds)
            outputs = self.model(images, cds)
            loss = self.bce(outputs, labels)
            # loss += self.mse(o, o_)
            # loss += self.ce(cd_, cds) * 0.01
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
                for batch, ((images, cds), labels) in enumerate(data_loader, 1):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    cds = cds.to(self.device)
                    # outputs = self.model(images, cds)
                    outputs = self.model(images, cds)
                    loss = self.bce(outputs, labels)
                    self.f1_score_2.update(outputs, labels)
                    self.f1_score_4.update(outputs, labels)
                    self.f1_score_6.update(outputs, labels)
                    loss_meter.update(loss.item())
                    # if self.f1_score.best_f1 > 0.6:
                    #     self.analysis_meter.difficult_image(outputs, labels, idx)
                print('Test Loss: %.4f | F1: %.4f %.4f %.4f | Best: %s' % (
                    loss_meter.avg, self.f1_score_2.get_f1()
                    , self.f1_score_4.get_f1(), self.f1_score_6.get_f1(), 'True' if self.f1_score_2.best() else 'False'
                ))
                loss_meter.reset()
                self.f1_score_2.reset()
