import os
import argparse
import logging as log
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import resnet34, resnet50
from loader import get_train_loader, get_val_loader
import settings
from metrics import accuracy

N_CLASSES = 100
batch_size = 64
epochs = 10

def create_res50(load_weights=False):
    resnet = resnet50(pretrained=True)

    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, N_CLASSES)
    #resnet = resnet.cuda()
    resnet.name = 'res50'
    return resnet

def train(args):
    model = create_res50()
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=args.lr)

    train_loader = get_train_loader(batch_size=batch_size)
    val_loader = get_val_loader(batch_size=batch_size)
    model.train()

    train_loss = 0
    iteration = 0
    best_val_loss = validate(model, criterion, val_loader)
    model.train()

    for epoch in range(epochs):
        current_lr = get_lrs(optimizer) 
        print('lr:', current_lr)
        bg = time.time()

        for batch_idx, data in enumerate(train_loader):
            iteration += 1
            x, target = data
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('epoch {}: {}/{} batch loss: {:.4f}, avg loss: {:.4f} lr: {}'
                    .format(epoch, batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1), current_lr), end='\r')

            if iteration % 500 == 0:
                val_loss = validate(model, criterion, val_loader)
                model.train()
                print('\nval loss: {:.4f}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_file)

def validate(model, criterion, val_loader):
    print('validating...')
    model.eval()
    val_loss = 0
    targets = None
    outputs = None
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.cuda(), target.cuda()
            output = model(x)
            loss = criterion(output, target)
            val_loss += loss.item()
            #acc = accuracy(output, target)
            #print('val acc:', acc)
    val_loss = val_loss / (val_loader.num/batch_size)
    print('\nval loss: {:.4f}'.format(val_loss))
    return val_loss

       
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    args = parser.parse_args()

    log.basicConfig(
        filename = 'trainlog.txt', 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)

    train(args)