import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image
from torchvision import transforms
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return float(self.sum) / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class AnalysisMeter(object):

    def __init__(self):
        self.num_classes = 50
        self.difficult_images = {}
        self.acc = []
        self.total = []
        self.reset()

    def reset(self):
        self.total = [0] * self.num_classes
        self.acc = [0] * self.num_classes

    def update(self, predict, label):
        for i in range(predict.size(0)):
            self.total[int(label[i])] += 1
            if int(label[i]) == int(predict[i]):
                self.acc[int(predict[i])] += 1

    def result(self, i):
        return float(self.acc[i]) / self.total[i]

    def difficult_image(self, predict, label, idx):
        predict = f.softmax(predict, dim=1)
        x = predict * label
        x = torch.sum(x, dim=1)
        x = torch.where(x < 0.1)[0]
        for i in range(len(x)):
            k = idx[x[i]]
            if k in self.difficult_images:
                self.difficult_images[k] += 1
            else:
                self.difficult_images[k] = 1

    def show_difficult_image(self):
        a = sorted(self.difficult_images.items(), key=lambda x: x[1], reverse=True)
        print(a[:20])


def image_analysis():
    means = [0, 0, 0]
    std = [0, 0, 0]
    count = 0

    path = "../corel_5k/images/"
    path_list = os.listdir(path)
    path_list.sort()
    for filename in path_list:
        class_path = os.path.join(path, filename)
        class_list = os.listdir(class_path)
        for img_name in class_list:
            img = os.path.join(class_path, img_name)
            img = Image.open(img)
            transform = transforms.Compose([
                # transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            img = transform(img)
            for i in range(3):
                means[i] += img[i, :, :].mean()
                std[i] += img[i, :, :].std()
            count += 1
    for i in range(3):
        means[i] = means[i] / count
        std[i] = std[i] / count
    print('means:', means)
    print('std:', std)


class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'\
            .format(size=self.size, interpolation=self.interpolation)


class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__


def gen_a(num_classes, t, adj_file):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    # _adj[_adj < t] = 0
    # _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    _adj = torch.from_numpy(_adj).double()
    return _adj


def gen_adj(a):
    D = torch.pow(a.sum(1).float(), -0.5)
    D = torch.diag(D).double()
    adj = torch.matmul(torch.matmul(a, D).t(), D)
    return adj.detach()


class F1Score(object):
    def __init__(self):
        self.tp = torch.zeros(374).cuda()
        self.fp = torch.zeros(374).cuda()
        self.fn = torch.zeros(374).cuda()
        self.best_f1 = 0

    def update(self, predict, label):
        label = label.cuda()
        x = torch.zeros_like(predict).cuda()
        threshold = 0.2
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                if predict[i][j] > threshold:
                    x[i][j] = 1
        predict = x

        self.tp += torch.sum(label * predict, dim=0)
        self.fp += torch.sum((1 - label) * predict, dim=0)
        self.fn += torch.sum(label * (torch.tensor(1) - predict), dim=0)

    def get_f1(self):
        p = self.tp / (self.tp + self.fp + 1e-5)
        r = self.tp / (self.tp + self.fn + 1e-5)
        f1 = (2 * p * r) / (p + r + 1e-5)

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


def load_label_mask(path):
    result = pickle.load(open(path, 'rb'))
    result = torch.from_numpy(result).cuda()
    return result


def load_emb(path):
    emb = pickle.load(open(path, 'rb'))
    emb = torch.from_numpy(emb).cuda()
    return emb
