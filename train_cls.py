import os
import argparse
import numpy as np
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
from torchvision.models import resnet34
import pdb
import settings
from single_class_loader import get_train_val_loaders
import cv2
from net.senet import se_resnext50_32x4d, se_resnet50

MODEL_DIR = settings.MODEL_DIR

def focal_loss(x, y):
    '''Focal loss.

    Args:
    x: (tensor) sized [N,D].
    y: (tensor) sized [N,].

    Return:
    (tensor) focal loss.
        '''
    alpha = 0.25
    gamma = 2

    #t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
    #t = t[:,1:]  # exclude background
    #t = Variable(t).cuda()  # [N,20]

    t = torch.eye(7272).cuda()
    t = t.index_select(0, y)

    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    #return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
    return F.binary_cross_entropy_with_logits(x, t, w)

def criterion(outputs, targets):
    #c = nn.CrossEntropyLoss()
    loss = focal_loss(outputs, targets)
    return loss

def accuracy(output, label, topk=(1,5)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res


def create_model():
    basenet = se_resnext50_32x4d(num_classes=7272, pretrained=None)
    basenet.name = 'se_resnext50_32x4d'
    return basenet

def test_model():
    x = torch.randn(2, 3, 256, 256).cuda()
    model = create_model().cuda()
    y = model(x)
    print(y.size())

def train(args):
    print('start training...')

    model = create_model()
    model_file = os.path.join(MODEL_DIR, 'classifier', model.name, 'best.pth')

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.init_ckp is not None:
        CKP = args.init_ckp
    else:
        CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_val_loaders(args, batch_size=args.batch_size, dev_mode=args.dev_mode)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    print('epoch |   lr    |   %        |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  acc   |')

    best_iout, _iou, _f, _l, _b, _ship, best_cls_acc = validate(args, model, val_loader, args.start_epoch)
    print('val   |         |            |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      | {:.4f} |'.format(
        _f, _l, _b, _ship, _iou, best_iout, best_cls_acc, best_cls_acc))
    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_iout)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target = data
            img, target = img.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(img)
            
            loss = criterion(output, target)
            loss.backward()
 
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                iout, iou, focal_loss, lovaz_loss, bce_loss, cls_loss, cls_acc = validate(args, model, val_loader, epoch=epoch)
                
                _save_ckp = ''
                if cls_acc > best_cls_acc:
                    best_cls_acc = cls_acc
                    torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                # print('epoch |   lr    |   %       |  loss  |  avg   | f loss | lovaz  |  bce   |  cls   |  iou   | iout   |  best  | time | save |  ship  |')
                print(' {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} | {:.4f} |'.format(
                    focal_loss, lovaz_loss, bce_loss, cls_loss, iou, iout, best_cls_acc, (time.time() - bg) / 60, _save_ckp, cls_acc))

                #log.info('epoch {}: train loss: {:.4f} focal loss: {:.4f} lovaz loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {} {}'
                #    .format(epoch, train_loss, focal_loss, lovaz_loss, iout, best_iout, iou, current_lr, _save_ckp))

                model.train()
                
                if args.lrs == 'plateau':
                    lr_scheduler.step(cls_acc)
                else:
                    lr_scheduler.step()
                current_lr = get_lrs(optimizer)

    del model, train_loader, val_loader, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader, epoch=0, threshold=0.5, cls_threshold=0.5):
    #return [0]*7
    model.eval()
    #print('validating...')

    total_num = 0
    corrects = 0
    ship_loss = 0
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.cuda(), target.cuda()
            output = model(img)
            loss  = criterion(output, target)
            ship_loss += loss.item()

            #preds = output.max(1, keepdim=True)[1]
            #corrects += preds.eq(target.view_as(preds)).sum().item()
            _, top5 = accuracy(output, target)
            corrects += top5
            total_num += len(img)
            
    cls_acc = corrects / total_num
    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    return 0, 0, 0, 0, ship_loss / n_batches, ship_loss/ n_batches, cls_acc



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0001, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--iter_val', default=100, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='plateau', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=15, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--cls_type', choices=['trainable', 'tuning'], type=str, default='trainable', help='train class type')
    parser.add_argument('--start_index', default=0, type=int, help='start index of classes')
    parser.add_argument('--end_index', default=7272, type=int, help='end index of classes')
    parser.add_argument('--max_labels', default=1, type=int, help='filter max labels')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)

    train(args)
    #pred_class(args)
    #test_model()
