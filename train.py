from torchtext.legacy.data import Dataset, BucketIterator, Iterator, Example, Field
from torchtext.vocab import Vectors
import os
from tqdm import trange, tqdm
import jieba
import argparse
from get_loader import get_iterator
from net import NLPnet
from torch import optim
from torch import nn

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-no_train', action='store_true', default=False, help='test or train')
    parse.add_argument('-cuda', action='store_true', default=False, help='if True: use cuda. if False: use cpu')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('-seed_num', type=int, default=13, help='random seed')
    parse.add_argument('-batch_size', type=int, default=512, help='batch size number')
    parse.add_argument('-iterator_type', type=str, default='bucket', help='bucket, iterator')
    parse.add_argument('-min_freq', type=int, default=5, help='build vocab\'s min word freq')
    opt = parse.parse_args()
    print(opt)


    def tokenize(x):
        return jieba.lcut(x)


    sentence_field = Field(sequential=True, tokenize=tokenize, lower=False, batch_first=True)
    label_field = Field(sequential=True, tokenize=tokenize, lower=False, batch_first=True)
    train_iterator, test_iterator, embedding_vector = get_iterator(opt, 'data/train.txt', 'data/test.txt',
                                                                   sentence_field, label_field,
                                                                   './.vector_cache/sgns.wiki.bigram-char')

    net = NLPnet(vocab_size=embedding_vector.shape[0], embedding_dim=embedding_vector.shape[1],
                 embedding_vector=embedding_vector, hide_num=1024).cuda()

    optimizer = optim.Adam(net.parameters())
    loss_fun = nn.MSELoss()

    for epoch in range(100000):

        for i, data in enumerate(tqdm(train_iterator)):
            out = net(data.sentence)
            loss = loss_fun(out, data.label)
            print(loss)
            exit()