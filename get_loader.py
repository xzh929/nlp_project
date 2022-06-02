from torchtext.legacy.data import Dataset, BucketIterator, Iterator, Example, Field
from torchtext.vocab import Vectors
import os
from tqdm import trange, tqdm
from sample import sample
import jieba
import argparse


class TorchtextSentenceDataset(Dataset):

    def __init__(self, data_path, sentence_field, label_field):
        self.sentence_field = sentence_field
        self.label_field = label_field

        fields = [('sentence', self.sentence_field), ('label', self.label_field)]
        examples = []

        file = open(data_path, "r", encoding="utf-8")
        lines = file.readlines()
        self.sentence_list = []
        self.label_list = []
        for data in lines:
            data = data.strip()
            data = data.split("|")
            self.sentence_list.append(data[0])
            self.label_list.append(data[1])
        file.close()

        for index in trange(len(self.sentence_list)):
            examples.append(Example.fromlist([self.sentence_list[index], self.label_list[index]], fields))

        for index in range(len(examples)):
            if len(examples[index].sentence) < 31:
                examples[index].sentence.extend('<pad>' for i in range(31 - len(examples[index].sentence)))

        for index in range(len(examples)):
            if len(examples[index].label) < 16:
                examples[index].label.extend('<pad>' for i in range(16 - len(examples[index].label)))

        super().__init__(examples, fields)


def get_iterator(opt, train_data_path, test_data_path, sentence_field, label_field, vectors_path):
    train_dataset = TorchtextSentenceDataset(train_data_path, sentence_field, label_field)
    test_dataset = TorchtextSentenceDataset(test_data_path, sentence_field, label_field)
    cache = './vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=vectors_path, cache=cache)
    sentence_field.build_vocab(train_dataset, min_freq=5, vectors=vectors)
    label_field.build_vocab(test_dataset, min_freq=5, vectors=vectors)

    train_iterator = BucketIterator(train_dataset, batch_size=opt.batch_size,
                                    device='cuda' if opt.cuda else 'cpu', sort_key=lambda x: len(x.sentence),
                                    sort_within_batch=True, shuffle=True)
    # train_iterator = Iterator(train_dataset, batch_size=opt.batch_size,
    #                                 device='cuda' if opt.cuda else 'cpu', sort_key=lambda x: len(
    #                                     x.sentence),
    #                                 sort_within_batch=True, shuffle=True)
    test_iterator = Iterator(test_dataset, batch_size=opt.batch_size,
                             train=False, sort=False,
                             sort_within_batch=False, shuffle=False,
                             device='cuda' if opt.cuda else 'cpu')

    # for index, data in enumerate(test_iterator):
    #     if index == 65:
    #         a = 0

    return train_iterator, test_iterator, sentence_field.vocab.vectors


# get_iterator('data/train.json', 8, '0', is_test=False)

