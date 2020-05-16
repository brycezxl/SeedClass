import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gensim.models import KeyedVectors


def gen_adj():
    adj = np.zeros((374, 374))
    path_train = "../corel_5k/labels/training_label"
    path_test = "../corel_5k/labels/test_label"
    with open(path_train, "r") as f:
        label = f.readline()
        while label:
            label = label.split()[1:]
            if len(label) > 1:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        x1 = int(label[i]) - 1
                        x2 = int(label[j]) - 1
                        adj[x1][x2] += 1
                        adj[x2][x1] += 1
            label = f.readline()
    with open(path_test, "r") as f:
        label = f.readline()
        while label:
            label = label.split()[1:]
            if len(label) > 1:
                for i in range(len(label) - 1):
                    for j in range(i + 1, len(label)):
                        x1 = int(label[i]) - 1
                        x2 = int(label[j]) - 1
                        adj[x1][x2] += 1
                        adj[x2][x1] += 1
            label = f.readline()
    nums = np.sum(adj, axis=-1)
    result = {'adj': adj, 'nums': nums}
    with open('../corel_5k/adj.pkl', 'wb+') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gen_cd_adj():
    info_labels = {}
    train_path = "../corel_5k/labels/training_label"
    test_path = "../corel_5k/labels/test_label"
    with open(train_path, "r") as f:
        label = f.readline()
        while label:
            label = label.split()
            info_labels[label[0]] = []
            for i in range(1, len(label)):
                info_labels[label[0]].append(int(label[i]))
            label = f.readline()
    with open(test_path, "r") as f:
        label = f.readline()
        while label:
            label = label.split()
            info_labels[label[0]] = []
            for i in range(1, len(label)):
                info_labels[label[0]].append(int(label[i]))
            label = f.readline()

    adj = np.zeros((50, 374, 374))
    class_idx = 0
    path = "../corel_5k/images/"
    path_list = os.listdir(path)
    path_list.sort()
    for filename in path_list:
        class_path = os.path.join(path, filename)
        class_list = os.listdir(class_path)
        for img_name in class_list:
            idx = img_name[:-5]
            if idx in info_labels:
                info = info_labels[idx]
                for i in range(len(info) - 1):
                    for j in range(i + 1, len(info)):
                        x1 = int(info[i]) - 1
                        x2 = int(info[j]) - 1
                        adj[class_idx, x1, x2] += 1
                        adj[class_idx, x2, x1] += 1
        class_idx += 1

    with open("../corel_5k/cd_adj.pkl", "wb+") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_emb():
    emb_path = "/home/zxl/Documents/glove.840B.300d.bin"
    word_path = "../corel_5k/labels/words"
    path = "../corel_5k/word2vec.pkl"
    emb = np.zeros((374, 300))
    word2vec = KeyedVectors.load_word2vec_format(emb_path, binary=True)
    with open(word_path, "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        word = words[i][:-1]
        emb[i, :] = np.asarray(word2vec[word])
    with open(path, "wb+") as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)


def heat_map():
    sns.set()
    result = pickle.load(open("../corel_5k/adj.pkl", 'rb'))
    adj = result['adj']
    sns.heatmap(adj)
    plt.show()

    sns.set()
    num = result['nums']
    adj = adj / num
    sns.heatmap(adj)
    plt.show()


def analysis_occur():
    result = pickle.load(open("../corel_5k/adj.pkl", 'rb'))
    total_occur = [0] * 40
    num = result['nums']
    count = 0
    total1 = np.sum(num)
    print("mean: %4.2f, median: %4.2f" % (float(np.mean(num)), float(np.median(num))))
    for n in range(len(num)):
        idx = int(num[n] / 10)
        total_occur[min(idx, 39)] += 1
        if num[n] < 50:
            num[n] = 0
            count += 1
    total2 = np.sum(num)
    print(total1, total2, 374 - count)
    plt.plot([i * 10 for i in range(40)], total_occur)
    plt.xlabel("occur times")
    plt.ylabel("edge num")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()


def gen_label_mask():
    info_labels = {}
    train_path = "../corel_5k/labels/training_label"
    test_path = "../corel_5k/labels/test_label"
    with open(train_path, "r") as f:
        label = f.readline()
        while label:
            label = label.split()
            info_labels[label[0]] = []
            for i in range(1, len(label)):
                info_labels[label[0]].append(int(label[i]))
            label = f.readline()
    with open(test_path, "r") as f:
        label = f.readline()
        while label:
            label = label.split()
            info_labels[label[0]] = []
            for i in range(1, len(label)):
                info_labels[label[0]].append(int(label[i]))
            label = f.readline()

    label_of_cd = {}
    for i in range(50):
        label_of_cd[i] = {}
    cd_of_label = {}
    for i in range(374):
        cd_of_label[i] = {}

    class_idx = 0
    path = "../corel_5k/images/"
    path_list = os.listdir(path)
    path_list.sort()
    for filename in path_list:
        class_path = os.path.join(path, filename)
        class_list = os.listdir(class_path)
        for img_name in class_list:
            idx = img_name[:-5]
            if idx in info_labels:
                for i in info_labels[idx]:
                    i_ = i - 1
                    if i_ in label_of_cd[class_idx]:
                        label_of_cd[class_idx][i_] += 1
                    else:
                        label_of_cd[class_idx][i_] = 1
                    if class_idx in cd_of_label[i_]:
                        cd_of_label[i_][class_idx] += 1
                    else:
                        cd_of_label[i_][class_idx] = 1
        class_idx += 1

    label_mask_path = "../corel_5k/label_mask.pkl"
    label_mask = np.zeros((50, 374))
    for cd, v in label_of_cd.items():
        total = 0
        n = 0
        n_list = []
        n_ = 0
        t_ = 0
        for label, count in v.items():
            total += count
            n += 1
        for label, count in v.items():
            if count < 5:
                continue
            n_ += 1
            t_ += count
            label_mask[cd][label] = count
            n_list.append(count)

        # plt.xlabel("occur times")
        # plt.ylabel("edge num")
        # plt.hist(sorted(n_list, reverse=True), bins=50)

        # plt.show()
        print(t_, total, n, n_)
        label_mask[cd] = label_mask[cd] / total
    with open(label_mask_path, "wb+") as f:
        pickle.dump(label_mask, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # gen_adj()
    # gen_emb()
    # heat_map()
    # analysis_occur()
    gen_label_mask()
    # gen_cd_adj()
