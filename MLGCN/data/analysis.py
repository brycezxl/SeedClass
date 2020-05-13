import numpy as np
import pickle
from gensim.models import KeyedVectors


def gen_adj():
    adj = np.zeros((374, 374))
    path_train = "corel_5k/labels/training_label"
    path_test = "corel_5k/labels/test_label"
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
    with open('corel_5k/adj.pkl', 'wb+') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gen_emb():
    emb_path = "/home/zxl/Documents/glove.840B.300d.bin"
    word_path = "./corel_5k/labels/words"
    path = "./corel_5k/word2vec.pkl"
    emb = np.zeros((374, 300))
    word2vec = KeyedVectors.load_word2vec_format(emb_path, binary=True)
    with open(word_path, "r") as f:
        words = f.readlines()
    for i in range(len(words)):
        word = words[i][:-1]
        emb[i, :] = np.asarray(word2vec[word])
    with open(path, "wb+") as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # gen_adj()
    # gen_emb()
    print(1)
