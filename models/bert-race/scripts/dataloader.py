from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
import utils
from transformers import BertTokenizer

class TokenDataset(Dataset):
    def __init__(self, df,  x_col, y_col):
        self.df = df.copy()
        self.text =  list(map(utils.clean, self.df[x_col].to_list()))
        self.label = list(map(int,self.df[y_col].to_list()))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokens = self.tokenizer(self.text, padding=True, return_tensors='pt', truncation=True, max_length=512)
        self.input_ids= self.tokens["input_ids"]
        self.attention_masks= self.tokens["attention_mask"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        input_ids=self.input_ids[idx]
        attention_mask=self.attention_masks[idx]
        label = self.label[idx]
        return input_ids,attention_mask, label
    
    def getLabels(self):
        return self.label