from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import sys
import utils

class Trainer:
    def __init__(self, train_loader, net, opt, loss_fun, module_path, opt_path):
        self.summary = SummaryWriter("logs")
        self.loader = train_loader
        self.net = net
        self.opt = opt
        self.loss_fun = loss_fun
        self.module_path = module_path
        self.opt_path = opt_path

    def __call__(self):
        init_acc = 0.
        print("--start training--")
        for epoch in range(152, 100000):
            sum_loss = 0.
            sum_acc = 0.
            for i, (data, label) in enumerate(tqdm(self.loader, file=sys.stdout)):
                data, label = data.cuda(), label.cuda()
                out = self.net(data, label)
                loss = self.loss_fun(out, label)
                # print("data:", data, "out:", out, "label:", label)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                sum_loss += loss.item()
                sum_acc += torch.mean(torch.eq(torch.argmax(out, dim=1), label).float())

            avg_loss = sum_loss / len(self.loader)
            avg_acc = sum_acc / len(self.loader)
            self.summary.add_scalar("loss", avg_loss, epoch)
            self.summary.add_scalar("acc", avg_acc, epoch)
            print("epoch:{} loss:{} acc:{}".format(epoch + 1, avg_loss, avg_acc))

            if avg_acc > init_acc:
                init_acc = avg_acc
                torch.save(self.net.state_dict(), self.module_path)
                torch.save(self.opt.state_dict(), self.opt_path)
                print("save success")


class Tester:
    def __init__(self, train_loader, net, opt, loss_fun, module_path, opt_path):
        self.loader = train_loader
        self.net = net
        self.opt = opt
        self.loss_fun = loss_fun
        self.module_path = module_path
        self.opt_path = opt_path
        self.label_dict = utils.getlabeldict(r"data/name.txt")
        file = open(r"data/test.txt", "r", encoding="utf-8")
        lines = file.readlines()
        self.sentence_list = []
        for data in lines:
            data = data.strip()
            data = data.split("|")
            self.sentence_list.append(data[0])
        file.close()

    def __call__(self):
        print("--start testing--")
        sum_loss = 0.
        sum_acc = 0.
        pre_data = []
        self.net.eval()
        for i, (data, label) in enumerate(tqdm(self.loader, file=sys.stdout)):
            data, label = data.cuda(), label.cuda()
            out = self.net(data, label)
            loss = self.loss_fun(out, label)
            # print("data:", data, "out:", out, "label:", label)

            sum_loss += loss.item()
            sum_acc += torch.mean(torch.eq(torch.argmax(out, dim=1), label).float())
            pre = torch.argmax(out, dim=1)
            pre_data.append(pre)

        avg_acc = sum_acc / len(self.loader)
        avg_loss = sum_loss / len(self.loader)
        print("loss:{} acc:{}".format(avg_loss, avg_acc))
        pre_data = torch.cat(pre_data)
        pre_label = [list(self.label_dict.keys())[i] for i in pre_data]
        for i, sentence in enumerate(self.sentence_list):
            print(sentence, pre_label[i])
