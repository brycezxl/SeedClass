import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class Corel(data.Dataset):
    def __init__(self, train):
        self.train = train
        self.path = '../corel_5k/'
        self._load_labels()
        self._load_images()

        self.transforms_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transforms_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
                    label = []
                    for i in self.info_labels[idx]:
                        label.append(i - 1)
                    label += [374 for _ in range(5 - len(label))]
                    self.data.append((img, class_idx, label))
            class_idx += 1
        random.shuffle(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        img = Image.open(d[0])
        if self.train:
            img = self.transforms_train(img)
        else:
            img = self.transforms_test(img)
        return (img, d[1]), d[2]

    def __len__(self):
        return len(self.data)
