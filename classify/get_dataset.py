import os
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import torch


class DataSet(data.Dataset):
    def __init__(self, train):
        self.train = train
        self.path = "../corel_5k/"
        self._load_labels()
        self._load_images()
        self.transforms_train = transforms.Compose([
            # transforms.RandomChoice([
                transforms.Resize((64, 64)),
                # transforms.RandomResizedCrop(64),
            # ]),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([transforms.ColorJitter()]),
            transforms.ToTensor(),
            transforms.Normalize((0.3861, 0.4012, 0.3479), (0.2014, 0.1901, 0.1827)),
        ])
        self.transforms_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.3861, 0.4012, 0.3479), (0.2014, 0.1901, 0.1827)),
        ])

    def _load_labels(self):
        self.info_labels = {}
        if self.train:
            path = self.path + "labels/training_label"
        else:
            path = self.path + "labels/test_label"
        with open(path, "r") as f:
            label = f.readline()
            while label:
                label = label.split()
                self.info_labels[label[0]] = []
                for i in range(1, len(label)):
                    self.info_labels[label[0]].append(int(label[i]))
                label = f.readline()

    def _load_images(self):
        self.data = []
        self.labels = []
        class_idx = 0
        path = self.path + "images/"
        path_list = os.listdir(path)
        path_list.sort()
        for filename in path_list:
            class_path = os.path.join(path, filename)
            class_list = os.listdir(class_path)
            for img_name in class_list:
                img = os.path.join(class_path, img_name)
                if img_name[:-5] in self.info_labels:
                    idx = img_name[:-5]
                    label = self.info_labels[idx]
                    mask = [1] * len(label) + [0] * (5 - len(label))
                    label += [0 for _ in range(5 - len(label))]
                    class_array = np.zeros(50)
                    class_array[class_idx] = 1
                    self.data.append((img, label, idx, mask, class_array, label))
            class_idx += 1
        random.shuffle(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.open(d[0])
        if self.train:
            img = self.transforms_train(img)
        else:
            img = self.transforms_test(img)
        return img, d[1], d[2], d[3], d[4], d[5]

    def __len__(self):
        return len(self.data) - 1
