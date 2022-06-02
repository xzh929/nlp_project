import pandas as pd
import torchtext
import numpy as np
import random
from utils import removelist


def readxlsx2txt(excel_path, save_path):
    file = pd.read_csv(excel_path, usecols=[0, 2])
    data = file.values
    with open(save_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write("{}|{}\n".format(line[0], line[1]))


def sample(file_path, train_path, test_path):
    data = []
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip()
            line = line.split("|")
            line[0] = line[0].strip()
            data.append(line)

    test_data = random.sample(data, 30000)
    train_data = removelist(data, test_data)
    with open(train_path, "w", encoding="utf-8") as train:
        for train_ele in train_data:
            train.write("{}|{}\n".format(train_ele[0], train_ele[1]))
    with open(test_path, "w", encoding="utf-8") as test:
        for test_ele in test_data:
            test.write("{}|{}\n".format(test_ele[0], test_ele[1]))

    return train_data, test_data


if __name__ == '__main__':
    file_path = r"data/name.txt"
    train_path = r"data/train.txt"
    test_path = r"data/test.txt"
    # readxlsx2txt(r"D:\data\ATC.csv", file_path)
    train_data, test_data = sample(file_path, train_path, test_path)