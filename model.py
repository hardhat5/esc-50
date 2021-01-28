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

class EnvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=40, kernel_size=8),
                                    nn.BatchNorm1d(40), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=40, out_channels=40, kernel_size=8),
                                nn.BatchNorm1d(40),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=160, ceil_mode=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=50, kernel_size=[8, 13]),
                                nn.BatchNorm2d(50),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=50, out_channels=50, kernel_size=[1,5]),
                                nn.BatchNorm2d(50),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=[1,3]))

        self.linear = nn.Sequential(nn.Linear(7700, 4096), nn.Dropout(p=0.5), nn.ReLU(),
                                    nn.Linear(4096, 4096), nn.Dropout(p=0.5), nn.ReLU(),
                                    nn.Linear(4096, 50))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.unsqueeze(1)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out