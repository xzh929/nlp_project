import torch
from torch import nn
import math


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


class NLPnet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_vector, hide_num):
        super().__init__()
        self.hide_num = hide_num
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding.weight.data.copy_(embedding_vector)
        # self.embedding.weight.requires_grad = False

        transformerlayer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                      nhead=6,
                                                      dim_feedforward=self.hide_num,
                                                      batch_first=True)
        self.encoder = nn.TransformerEncoder(transformerlayer, num_layers=4)
        self.positionemb = PositionalEncoding(embedding_dim)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.positionemb(emb)
        out = self.encoder(emb)
        return out


if __name__ == '__main__':
    a = torch.randint(10, (2, 22))
    net = NLPnet(6938, 300, None, 1024)
    out = net(a)
    print(out.shape)
