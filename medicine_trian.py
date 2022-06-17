from dataset import MedicineDataset
from net import NLPnet, SkipNet
from net2 import My_net
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.nn import NLLLoss
import torch
import os
from trainer import Trainer
from torch import nn

if __name__ == '__main__':
    train_dataset = MedicineDataset(r"data/train.txt")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    vec_net = SkipNet(vocab_size=11639, embedding_dim=128)
    vec_opt = Adam(vec_net.parameters())
    if os.path.exists(r"module/vec.pth"):
        vec_net.load_state_dict(torch.load(r"module/vec.pth"))
        vec_opt.load_state_dict(torch.load(r"module/vec_opt.pth"))
    weight = vec_net.center_weight.weight
    weight.requires_grad = False
    net = NLPnet(embedding_dim=weight.shape[1], embedding_vector=weight, hide_num=1024,
                 sentence_len=31, cls_num=3846).cuda()
    # net = My_net().cuda()
    # opt = Adam(net.parameters())
    opt = SGD(net.parameters())
    loss_fun = NLLLoss()
    module_path = r"module/medicine.pt"
    opt_path = r"module/medicine_opt.pt"
    trainer = Trainer(train_loader, net, opt, loss_fun, module_path, opt_path)
    trainer()
