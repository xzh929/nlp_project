import random
import numpy as np
import torch
from torch import nn


# 带批次的word2id
def word2id(batch_list):
    maps = {'<start>': 0, '<pad>': 1, '<end>': 2}
    for list in batch_list:
        for item in list:
            if item not in maps:
                maps[item] = len(maps)
    return maps


def list_word2id(batch_list, pad=True):
    maps = word2id(batch_list)
    max_length = len(max(batch_list, key=len))
    for list in batch_list:
        if pad:
            list_length = len(list)
            if list_length < max_length:
                for i in range(max_length - list_length):
                    list.append('<pad>')
        for i, item in enumerate(list):
            list[i] = maps[item]
        list.append(maps['<end>'])
    return batch_list


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def padswapstart(list):
    pad = list[len(list) - 2]
    list.pop()
    list.insert(0, pad)
    return list


def getvaluedim2(tensor, tag):
    idx = torch.nonzero(tag)
    return tensor[idx[:, 0], idx[:, 1]]


def removelist(origin_list, removed_list):
    for remove_ele in origin_list:
        if remove_ele in removed_list:
            origin_list.remove(remove_ele)
    return origin_list


def getlabeldict(label_path):
    file = open(label_path, "r", encoding="utf-8")
    lines = file.readlines()
    label_list = []
    for data in lines:
        data = data.strip()
        data = data.split("|")
        label_list.append(data[1])
    file.close()
    label_dict = {}
    for item in label_list:
        if item not in label_dict:
            label_dict[item] = len(label_dict)
    return label_dict


def getsentence(pre_data):
    label_dict = getlabeldict(r"data/name.txt")
    idx = [list(label_dict.keys())[i] for i in pre_data]
    return idx


def word2vec(word_list, vec_path):
    with open(vec_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            exit()


def filtersentence(sentence):
    sentence_copy = sentence.copy()
    for item in sentence:
        if item == '\u3000':
            sentence_copy.remove(item)
        if item == '\t':
            sentence_copy.remove(item)
        if len(item) > 9:
            sentence_copy.remove(item)
    return sentence_copy


def cosinesimilarity(matrix1, matrix2):
    max_ele = torch.max(matrix2)
    matrix1, matrix2 = matrix1 / max_ele, matrix2 / max_ele
    return torch.matmul(matrix1, matrix2.T) / (torch.norm(matrix1, dim=1, keepdim=True) * torch.norm(matrix2, dim=1))


if __name__ == '__main__':
    # feature = torch.randn(5, 2)
    # tag = torch.tensor([1, 1, 0, 1, 0])
    # label = getlabeldict(r"data/name.txt")
    # a = [0, 1, 15, 54, 54, 31]
    # out = getsentence(a)
    # print(out)
    word_list = [['20%', '脂肪乳', '注射液', '<pad>']]
    word2vec(word_list, r".vector_cache/medicinevec.txt")
