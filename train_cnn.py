import csv
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch.nn.functional as F
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Training script for 3D CNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    parser.add_argument('--dataset', type=str, default='./demographicPrediction/profile_pics/profile pics/', help='Dataset to train on')
    parser.add_argument('--needbalanced', action='store_true', help='Balanced sampling')
    
    parser.add_argument('--epochs', metavar='N', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--num_workers', type=float, default=2, help='Number of workers for data loader')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')


def save_checkpoint(ckpt, is_best, fname):
    ckpt_path = 'dummy_path' # TODO: add checkpoint path here
    print(f'=> Saving checkpoint to {ckpt_path}.pth...')
    torch.save(ckpt, f'{ckpt_path}.pth')
    if is_best:
        print(f'=> Saving checkpoint to {ckpt_path}_best.pth...')
        torch.save(ckpt, f'{ckpt_path}_best.pth')

    
def train(model, train_loader, criterion, optimizer, epoch):
    # TODO: train model
    pass


def validate(model, loader, criterion, epoch=0):
    # TODO: validate model
    pass


def confusion_matrix(test_loader, model):
    # TODO: use test data to generate confusion matrix and report f1, recall, precision
    pass

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()
