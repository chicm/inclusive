import os
import argparse
import logging as log
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from loader import get_train_loader, get_val_loader, get_val2_loader
import settings
from metrics import accuracy
from models import create_res50, create_res50_2

N_CLASSES = 100

def train(args):
    model = create_res50_2()
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

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=3, min_lr=5e-6)

    train_loader = get_train_loader(batch_size=args.batch_size)
    val_loader = get_val_loader(batch_size=args.batch_size)
    val2_loader = get_val2_loader(batch_size=args.batch_size)
    model.train()

    train_loss = 0
    iteration = 0
    best_val_loss = validate(model, criterion, val_loader, args.batch_size)
    best_val_loss = validate(model, criterion, val2_loader, args.batch_size)
    lr_scheduler.step(best_val_loss)
    model.train()

    bg = time.time()
    current_lr = get_lrs(optimizer) 
    for epoch in range(args.epochs):
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
            print('epoch {}: {}/{} batch loss: {:.4f}, avg loss: {:.4f} lr: {}, {:.1f} min'
                    .format(epoch, args.batch_size*(batch_idx+1), train_loader.num,
                    loss.item(), train_loss/(batch_idx+1), current_lr, (time.time() - bg) / 60), end='\r')

            if iteration % 200 == 0:
                #val_loss = validate(model, criterion, val_loader, args.batch_size)
                val_loss = validate(model, criterion, val2_loader, args.batch_size)
                model.train()
                #print('\nval loss: {:.4f}'.format(val_loss))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print('saveing... {}'.format(model_file))
                    torch.save(model.state_dict(), model_file)
                lr_scheduler.step(val_loss)
                current_lr = get_lrs(optimizer) 
                print('lr:', current_lr)

def validate(model, criterion, val_loader, batch_size):
    print('\nvalidating...')
    model.eval()
    val_loss = 0
    targets = None
    outputs = None
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.cuda(), target.cuda()
            output = model(x)
            if targets is None:
                targets = target
            else:
                targets = torch.cat([targets, target])
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat([outputs, output])    
            loss = criterion(output, target)
            val_loss += loss.item()
            #acc = accuracy(output, target)
            
    val_loss = val_loss / (val_loader.num/batch_size)
    acc = accuracy(outputs, targets)
    print('\nval acc:', acc)
    print('\nval loss: {:.4f}'.format(val_loss))
    log.info(str(acc))
    return val_loss

       
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    args = parser.parse_args()

    log.basicConfig(
        filename = 'trainlog.txt', 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)

    train(args)
