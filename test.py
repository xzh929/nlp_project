import os
import torchtext
from torchtext.legacy.data import Field, Example, Dataset
from torchtext import vocab

vector_path = r".vector_cache/sgns.wiki.bigram-char"
p = os.path.expanduser(vector_path)
TEXT = Field(sequential=True, lower=True, fix_length=10, tokenize=str.split, batch_first=True)
TEXT

print(TEXT.vocab.vectors.shape)
