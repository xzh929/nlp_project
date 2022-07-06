from utils import getlabeldict, cosinesimilarity
import torch
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict,Counter
from transformers import pipeline

# with open(r"data/vocab_dict.txt", "r", encoding="utf-8") as f:
#     line = f.readline()
#     line = line.strip().split()
#     test_idx = np.random.randint(11639, size=15)
#     test_idx = np.unique(test_idx)
#     test_words = np.array(line)[test_idx]
#     with open(r"data/test_words.txt", "w", encoding="utf-8") as t:
#         for word in test_words:
#             t.write("{} ".format(str(word)))
# a = torch.tensor([[1., 2, 3]]) / 9.0
# b = torch.tensor([[4., 5, 6], [7, 8, 9], [1, 2, 3], [2, 2, 3]]) / 9.0
# norm_a = torch.norm(a, dim=1, keepdim=True)
# norm_b = torch.norm(b, dim=1)
# # print(torch.matmul(a, b.T) / (norm_a * norm_b))
# sim = cosinesimilarity(a, b)
# print(sim)
# sim_top = torch.argsort(sim, dim=1, descending=True).squeeze()
# print(sim_top)
translate = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
translate('Ce cours est produit par Hugging Face.')
