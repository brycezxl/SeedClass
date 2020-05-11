import time
import torch
import torch.nn.functional as f
import os
from PIL import Image
from torchvision import transforms


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


if __name__ == '__main__':
    image_analysis()
