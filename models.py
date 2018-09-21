import os
import argparse
import logging as log
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from torchvision.models import resnet34, resnet50
from loader import get_train_loader, get_val_loader
import settings
from metrics import accuracy

N_CLASSES = 100

def create_res50():
    resnet = resnet50(pretrained=True)

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, N_CLASSES)
    #resnet = resnet.cuda()
    resnet.name = 'res50'
    return resnet

def create_res50_2():
    resnet = resnet50(pretrained=True)

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, N_CLASSES)) 
    resnet.name = 'res50_2'
    return resnet

