import sys
import os
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
from tqdm import tqdm, trange
from configparser import ConfigParser
import librosa
import random

def getFoldIndices(n):
    val = [i+n*400 for i in range(400)]
    train = [i for i in range(2000) if i not in val]
    return val, train

class ESCDataset(Dataset):

    def __init__(self,df,mask, random_crop=False):
        self.df = df.iloc[mask, :]
        self.transform = random_crop

    def __len__(self):    
        return len(self.df)

    def __getitem__(self,idx):

        sample = {}
        file_name = self.df.iloc[idx]['filename']

        x, Fs = librosa.load("ESC-50-master/audio/"+file_name, sr=None)
        x = librosa.resample(x, Fs, 16000)
        x_norm = 2.*(x-np.min(x))/(np.max(x)-np.min(x)) - 1

        if self.transform:
            start_int = random.randint(0, len(x)-24000)
            x_norm = x_norm[start_int:start_int+24000]

        x_norm = np.expand_dims(x_norm, axis=0)
        x_norm = torch.FloatTensor(x_norm)
        target = self.df.iloc[idx]['target']

        sample['data'] = x_norm
        sample['target'] = target
            
        return sample