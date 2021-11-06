from __future__ import print_function, division
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import csv
from utils import read_files
import matplotlib.pyplot as plt

class CNN_Data(Dataset):
    '''
    img_dir (str): path to the actual image files
    csv_file (str): path to the labeled csv file
    json_file (str): path to the User demon profiles json file
    '''
    def __init__(self,img_dir, csv_file, json_file, seed = 1000):
        random.seed(seed)
        self.img_dir = img_dir
        self.data_list, self.label_list = read_files(json_file, csv_file, self.img_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        label = self.label_list[idx]
        print(self.data_list[idx])
        img = np.array(Image.open(self.data_list[idx])) # img now converts to numpy data, shape (1,224,224,3)

        #TODO: extract only faces among the data_list


        
        return self.data_list[idx], img, label # return tuple of (filename, actual image data, label)
    # TODO: get weighted sampling to balance classes
    def get_sample_weights(self):
        pass

