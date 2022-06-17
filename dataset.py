import sys
from torch.utils.data import Dataset
import utils
import jieba
import torch
from tqdm import tqdm


class MedicineDataset(Dataset):
    def __init__(self, data_path):
        file = open(data_path, "r", encoding="utf-8")
        vocab_file = open(r"data/vocab_dict.txt", "r", encoding="utf-8")
        vocab_list = vocab_file.readline().split()
        lines = file.readlines()
        self.sentence_list = []
        self.label_list = []
        print("--loading data--")
        for data in tqdm(lines, file=sys.stdout):
            data = data.strip()
            data = data.split("|")
            new_sentence = jieba.lcut(data[0])
            new_sentence = utils.filtersentence(new_sentence)
            if len(new_sentence) < 31:
                new_sentence.extend('<pad>' for i in range(31 - len(new_sentence)))
            new_sentence = [vocab_list.index(vocab) for vocab in new_sentence]
            self.sentence_list.append(new_sentence)
            self.label_list.append(data[1])
        file.close()
        vocab_file.close()
        self.sentence_list = list(set(tuple(sentence) for sentence in self.sentence_list))
        self.label_dict = utils.getlabeldict(r"data/name.txt")
        self.label_list_id = [int(self.label_dict[i]) for i in self.label_list]

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, item):
        return torch.tensor(self.sentence_list[item]), torch.tensor(self.label_list_id[item])


if __name__ == '__main__':
    dataset = MedicineDataset(r"data/test.txt")
    print(len(dataset))
    data, label = dataset[125]
    print(data, label)
    # label_dict = utils.getlabeldict(r"data/name.txt")
    # print(label_dict)
    # vocab_file = open(r"data/vocab_dict.txt", "r", encoding="utf-8")
    # vocab_list = vocab_file.readline().split()
    # print(vocab_list[1395],vocab_list[1396],vocab_list[110],vocab_list[2591])
