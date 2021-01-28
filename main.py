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
from model import EnvNet
from utils import getFoldIndices, ESCDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('ESC-50-master/meta/esc50.csv')

batch_size = 16

val_idx, train_idx = getFoldIndices(1)

train_dataset = ESCDataset(df, train_idx, random_crop=True)
train_load = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

val_dataset = ESCDataset(df, val_idx)
val_load = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

lr = 0.01
epochs = 150

model = EnvNet()
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = lr, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,100,120], gamma=0.1)

print('Starting training')
for i in range(epochs):
    
    training_loss = 0.0
    
    model.train()
    print("Training epoch {}".format(i))
    for j, sample in enumerate(tqdm(train_load)):
        data = sample['data']
        data = data[:,:,:24000]
        target = sample['target']
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        training_loss += loss.item()

    
    model.eval()
    correct = 0
    with torch.no_grad():
        for j, sample in enumerate(tqdm(val_load)):
            
            data = sample['data']
            target = sample['target']
            data = data.to(device)

            for k in range(data.shape[0]):
                p = 0
                pred = []
                while(p+24000<=80000):
                    section = data[k,:,p:p+24000]
                    if(torch.max(torch.abs(section))>=0.2):
                        section = torch.unsqueeze(section, dim=0)
                        output = model(section)
                        output = F.softmax(output)
                        pred.append(output.detach().cpu().numpy())
                    p += 3200

                pred = np.array(pred)
                pred = np.sum(pred, axis=0)
                pred = np.argmax(pred)
                if(pred==target[k]): correct+=1

    acc = correct/len(val_idx)
    print("Accuracy: ", acc)
    print("Training loss: ", training_loss)
    torch.save(model.state_dict(), 'weights/epoch_{}'.format(i))
                
