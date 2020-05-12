import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import torch


class Analysis(object):
    def __init__(self, train):
        self.train = train
        self.path = "../corel_5k/"
        self._load_labels()
        self._load_images()

    def _load_labels(self):
        self.train_labels = {}
        self.test_labels = {}
        train_path = self.path + "labels/training_label"
        test_path = self.path + "labels/test_label"
        with open(train_path, "r") as f:
            label = f.readline()
            while label:
                label = label.split()
                self.test_labels[label[0]] = []
                for i in range(1, len(label)):
                    self.test_labels[label[0]].append(int(label[i]))
                label = f.readline()
        with open(test_path, "r") as f:
            label = f.readline()
            while label:
                label = label.split()
                self.train_labels[label[0]] = []
                for i in range(1, len(label)):
                    self.train_labels[label[0]].append(int(label[i]))
                label = f.readline()

    def _load_images(self):
        self.train_total = {}
        self.test_total = {}
        self.cd_total = {}
        self.label_total = {}
        self.label_num_total = {}
        cd_idx = 0
        path = self.path + "images/"
        path_list = os.listdir(path)
        path_list.sort()
        for filename in path_list:
            class_path = os.path.join(path, filename)
            class_list = os.listdir(class_path)
            for img_name in class_list:
                idx = img_name[:-5]
                if idx in self.test_labels:
                    label = self.test_labels[idx]
                    for l_ in label:
                        if l_ in self.test_total:
                            self.test_total[l_] += 1
                else:
                    pass
                cd_idx += 1
