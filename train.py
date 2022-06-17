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
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import getlabeldict
import sys


def train(train_iterator, net, optimizer, loss_fun, summary, module_path, opt_path):
    init_acc = 0.
    for epoch in range(100000):
        sum_loss = 0.
        label_nums = 0.
        sum_acc = 0.
        for i, data in enumerate(tqdm(train_iterator,file=sys.stdout)):
            out = net(data.sentence)
            loss = loss_fun(out, data.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            label_nums += data.label.shape[0]

            sum_acc += torch.mean(torch.eq(torch.argmax(out, dim=1), data.label).float())

        avg_loss = sum_loss / (label_nums // opt.batch_size)
        avg_acc = sum_acc / (label_nums // opt.batch_size)
        summary.add_scalar("loss", avg_loss, epoch)
        summary.add_scalar("acc", avg_acc, epoch)
        print("epoch:{} loss:{} acc:{}".format(epoch + 1, avg_loss, avg_acc))

        if avg_acc > init_acc:
            init_acc = avg_acc
            torch.save(net.state_dict(), module_path)
            torch.save(optimizer.state_dict(), opt_path)
            print("save success")


def tester(test_iterator, net, loss_fun):
    label_dict = getlabeldict(r"data/name.txt")
    file = open(r"data/test.txt", "r", encoding="utf-8")
    lines = file.readlines()
    sentence_list = []
    for data in lines:
        data = data.strip()
        data = data.split("|")
        sentence_list.append(data[0])
    file.close()
    with torch.no_grad():
        net.eval()
        sum_loss = 0.
        label_nums = 0.
        sum_acc = 0.
        pre_data = []
        for i, data in enumerate(tqdm(test_iterator)):
            out = net(data.sentence)
            loss = loss_fun(out, data.label)

            sum_loss += loss.item()
            label_nums += data.label.shape[0]
            pre = torch.argmax(out, dim=1)
            sum_acc += torch.mean(torch.eq(pre, data.label).float())

            pre_data.append(pre)

        avg_loss = sum_loss / (label_nums // opt.batch_size)
        avg_acc = sum_acc / (label_nums // opt.batch_size)
        print("loss:{} acc:{}".format(avg_loss, avg_acc))
        pre_data = torch.cat(pre_data)
        pre_label = [list(label_dict.keys())[i] for i in pre_data]
        for i, sentence in enumerate(sentence_list):
            print(sentence, pre_label[i])


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-no_train', action='store_true', default=False, help='test or train')
    parse.add_argument('-cuda', action='store_true', default=True, help='if True: use cuda. if False: use cpu')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parse.add_argument('-seed_num', type=int, default=13, help='random seed')
    parse.add_argument('-batch_size', type=int, default=256, help='batch size number')
    parse.add_argument('-iterator_type', type=str, default='bucket', help='bucket, iterator')
    parse.add_argument('-min_freq', type=int, default=5, help='build vocab\'s min word freq')
    opt = parse.parse_args()
    print(opt)


    def tokenize(x):
        return jieba.lcut(x)


    sentence_field = Field(sequential=True, tokenize=tokenize, lower=False, batch_first=True)
    label_field = Field(sequential=False, use_vocab=False)
    train_iterator, test_iterator, embedding_vector = get_iterator(opt, 'data/train.txt', 'data/test.txt',
                                                                   sentence_field, label_field,
                                                                   './.vector_cache/medicinevec.txt')

    net = NLPnet(vocab_size=embedding_vector.shape[0], embedding_dim=embedding_vector.shape[1],
                 embedding_vector=embedding_vector, hide_num=1024, sentence_len=31, cls_num=3846).cuda()

    optimizer = optim.Adam(net.parameters())
    loss_fun = nn.NLLLoss()
    summary = SummaryWriter("logs")
    module_path = r"module/medicine.pth"
    opt_path = r"module/opt.pth"

    if opt.no_train is False:
        train(train_iterator, net, optimizer, loss_fun, summary, module_path, opt_path)
    else:
        if os.path.exists(module_path):
            net.load_state_dict(torch.load(module_path))
            optimizer.load_state_dict(torch.load(opt_path))
            print("load module")
        else:
            print("no module")
        tester(test_iterator, net, loss_fun)
