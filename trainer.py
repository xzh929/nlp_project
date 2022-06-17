from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import sys


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
        for epoch in range(100000):
            sum_loss = 0.
            sum_acc = 0.
            for i, (data, label) in enumerate(tqdm(self.loader, file=sys.stdout)):
                data, label = data.cuda(), label.cuda()
                out = self.net(data)
                loss = self.loss_fun(out, label)

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
