import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import re
from torch.utils.data import Dataset, DataLoader
import torch
import math
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data = requests.get(url)
with open("sheakespeare.txt", "w") as f:
    f.write(data.text)
with open("sheakespeare.txt", "r") as f:
    text = f.read()
len(text)
def tokenize(text):
   text= text.replace("?","")
   ext= text.lower()
   text = re.sub(r"[^a-z\s]", "", text)
   return text.split()

tokenize(text)
vocab = {"<PAD>": 0,"<UNK>" : 1}
for word in tokenize(text):
    if word not in vocab:
        vocab[word] = len(vocab)
vocab_size=len(vocab)
vocab.keys()
def word_to_idx(text):
  return[vocab.get(word,0) for word in tokenize(text)]
idx=word_to_idx(text)

inputs = []
labels = []
seq_len = 50
for i in range(len(idx)-seq_len):
  inputs.append(idx[i:i+seq_len])
  labels.append(idx[i+seq_len])
class sheakespeare_dataset(Dataset):
  def __init__(self,inputs,labels):
    self.input = inputs
    self.label = labels

  def __len__(self):
     return len(self.input)

  def __getitem__(self,index):
    x = torch.tensor(self.input[index] , dtype=torch.long)
    y= torch.tensor(self.label[index] , dtype= torch.long)
    return x, y
datast=sheakespeare_dataset(inputs,labels)
print(f"Total samples : {len(datast)}")

x, y = datast[20]
print(f"Input tensor  : {x}")
print(f"Target tensor : {y}")
dataload = DataLoader(dataset=datast , batch_size=32, shuffle=True)
len(dataload)