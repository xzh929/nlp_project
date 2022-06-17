from word2vec_dataset import Word2VecDataset
from net import SkipNet
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import SmoothL1Loss
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
import os
import utils
import numpy as np

train_path = r"data/name.txt"


class Vectrain():
    def __init__(self):
        self.summary = SummaryWriter("veclogs")
        self.dataset = Word2VecDataset(train_path)
        self.train_loader = DataLoader(self.dataset, batch_size=512, shuffle=True)
        self.net = SkipNet(vocab_size=len(self.dataset.id_dict), embedding_dim=128).cuda()
        self.opt = optim.Adam(self.net.parameters())

    def __call__(self):
        print("--start training--")
        for epoch in range(1000000):
            sum_loss = 0.
            for i, (data, label) in enumerate(tqdm(self.train_loader, file=sys.stdout)):
                data, label = data.cuda(), label.cuda()
                loss = self.net(data, label)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()

            avg_loss = sum_loss / len(self.train_loader)
            self.summary.add_scalar("loss", avg_loss, epoch)
            print("epoch:{} loss:{}".format(epoch + 1, avg_loss))
            torch.save(self.net.state_dict(), r"module/vec.pth")
            torch.save(self.opt.state_dict(), r"module/vec_opt.pth")
            print("save success")

            self.net.eval()
            embed_weight = self.net.center_weight.weight.data.cpu()
            embed_weight = embed_weight / torch.max(embed_weight)
            test_file = open(r"data/test_words.txt", "r", encoding="utf-8")
            test_words = test_file.readline().strip().split()
            word_dict = self.dataset.id_dict
            for word in test_words:
                test_words_id = word_dict[word]
                test_center_emb = embed_weight[test_words_id][None]
                # 比较余弦相似度
                sim = utils.cosinesimilarity(test_center_emb, embed_weight)
                emb_idx_top = torch.argsort(sim, dim=1, descending=True).squeeze()[:6]
                similar_words = [list(word_dict.keys())[idx] for idx in emb_idx_top[1:]]
                print("{}:{}".format(word, ','.join(similar_words)))


if __name__ == '__main__':
    trainer = Vectrain()
    trainer()
    # net = trainer.net
    # opt = trainer.opt
    # if os.path.exists(r"module/vec.pth"):
    #     net.load_state_dict(torch.load(r"module/vec.pth"))
    #     opt.load_state_dict(torch.load(r"module/vec_opt.pth"))
    #
    # weight = net.sentences_emb_weight
    # weight.requires_grad = False
    # weight[1] = 0.
    # weight = weight.tolist()
    # id_dict = utils.padswapstart(trainer.dataset.id_dict)
    # with open(r".vector_cache/medicinevec.txt", "w", encoding="utf-8") as f:
    #     for k, v in tqdm(id_dict.items()):
    #         id_dict[k] = weight[v]
    #         vec_str = [str(x) for x in id_dict[k]]
    #         f.write("{} {}\n".format(k, " ".join(vec_str)))
    # print(id_dict)
