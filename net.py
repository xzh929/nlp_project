import torch
from torch import nn
import math
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Arcloss(nn.Module):
    def __init__(self, input_dim, output_dim, m):
        super().__init__()
        self._w = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self._m = m
        self.cls_num = output_dim

    def forward(self, f, s=10):
        f = F.normalize(f, dim=1)
        w = F.normalize(self._w, dim=0)
        # s = torch.sqrt(torch.sum(torch.pow(f, 2))) * torch.sqrt(torch.sum(torch.pow(w, 2)))
        cosa = torch.matmul(f, w) / s
        angle = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(angle + self._m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(angle + self._m)))

        return arcsoftmax


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 2, 1, bias=False),
            nn.BatchNorm1d(in_channel // 2),
            nn.Mish(),
            nn.Conv1d(in_channel // 2, in_channel // 2, 3, 1, 1, bias=False),
            nn.BatchNorm1d(in_channel // 2),
            nn.Mish(),
            nn.Conv1d(in_channel // 2, in_channel, 1, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.Mish()
        )

    def forward(self, x):
        return self.layer(x) + x


class NLPnet(nn.Module):
    def __init__(self, embedding_dim, embedding_vector, hide_num, sentence_len, cls_num):
        super().__init__()
        self.hide_num = hide_num
        self.sentence_len = sentence_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_vector, padding_idx=1)
        # self.embedding = nn.Embedding(embedding_vector.shape[0], embedding_vector.shape[1], padding_idx=1)

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 128, 3, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Conv1d(128, 256, 3, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.Hardswish()
        )

        # self.positionemb = PositionalEncoding(256)
        transformerlayer = nn.TransformerEncoderLayer(d_model=256,
                                                      nhead=8,
                                                      dim_feedforward=self.hide_num,
                                                      batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(transformerlayer, num_layers=6)

        self.feature = nn.Sequential(
            nn.Linear(256 * 14, 256 * 6, bias=False),
            nn.BatchNorm1d(256 * 6),
            nn.Hardswish(),
            nn.Linear(256 * 6, 256)
        )
        self.arcsoftmax = Arcloss(256, cls_num, 0.001)

    def forward(self, x):
        emb = self.embedding(x)
        cnn_out = self.input_layer(emb.permute(0, 2, 1))
        # pos_emb = self.positionemb(cnn_out.permute(0, 2, 1))
        encoder_out = self.encoder(cnn_out.permute(0, 2, 1))
        feature = self.feature(encoder_out.reshape(-1, 256 * 14))
        cls = torch.log(self.arcsoftmax(feature))
        return cls


class SkipNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_weight = nn.Embedding(vocab_size, embedding_dim)
        self.out_weight = nn.Embedding(vocab_size, embedding_dim)
        self.center_weight.weight.data.uniform_(-1, 1)
        self.out_weight.weight.data.uniform_(-1, 1)

    def forward(self, x, y):
        embed_center = self.center_weight(x)
        embed_out = self.out_weight(y)
        out_loss = F.logsigmoid((torch.sum(torch.mul(embed_center, embed_out), dim=1)))
        return -out_loss.mean()


if __name__ == '__main__':
    weight = torch.randn(11639, 128)
    a = torch.randint(10, (2, 31))
    # x = torch.randint(100, (2,))
    # y = torch.randint(100, (2,))
    net = NLPnet(128, weight, 1024, 31, 3846)
    # skipgram = SkipNet(11639, 128)
    out = net(a)
    # loss = skipgram(x, y)
    print(out.shape)
    # print(loss)
