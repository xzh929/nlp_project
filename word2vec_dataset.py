import sys
from torch.utils.data import Dataset
import os
import utils
import jieba
import torch
from tqdm import tqdm


class Word2VecDataset(Dataset):
    def __init__(self, word_path):
        data = []
        self.ori_data = []
        with open(word_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split("|")
                line0 = jieba.lcut(line[0])
                line1 = jieba.lcut(line[1])
                line0 = utils.filtersentence(line0)
                data.append(line0+line1)
        self.id_dict = utils.word2id(data)
        # print(self.id_dict)
        self.word2id = utils.list_word2id(data, pad=False)
        # print(self.word2id)
        print("--load data--")
        for sentence in tqdm(self.word2id, file=sys.stdout):
            for i, word in enumerate(sentence):
                if i == 0:
                    self.ori_data.append([word, sentence[i + 1]])
                elif i == len(sentence) - 1:
                    self.ori_data.append([word, sentence[i - 1]])
                else:
                    self.ori_data.append([word, sentence[i - 1]])
                    self.ori_data.append([word, sentence[i + 1]])
        # print(self.ori_data)

    def __len__(self):
        return len(self.ori_data)

    def __getitem__(self, item):
        return torch.tensor(self.ori_data[item][0]), torch.tensor(self.ori_data[item][1])


if __name__ == '__main__':
    dataset = Word2VecDataset(r"data/name.txt")
    data, label = dataset[0]
    print(data,label)
    # id_list = list(dataset.id_dict.keys())
    # with open(r"data/vocab_dict.txt", "w", encoding="utf-8") as f:
    #     for k in id_list:
    #         f.write("{} ".format(k))
    # print(len(dataset))
    # print(data, label)
