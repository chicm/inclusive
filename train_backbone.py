import os
import argparse
import numpy as np
import pandas as pd
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
from backbone_loader import get_train_val_loaders, get_test_loader
import cv2
from models import create_backbone_model, InclusiveNet
from utils import get_classes, get_cls_counts, get_weights_by_counts


MODEL_DIR = settings.MODEL_DIR
cls_weights = None

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

def criterion(args, outputs, targets, num_output, num_target):
    global cls_weights

    if cls_weights is None:
        classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)
        cnts = get_cls_counts(classes, args.cls_type)
        cls_weights = get_weights_by_counts(cnts, max_weight=10)
        cls_weights = torch.Tensor(cls_weights).cuda()

    #c = nn.CrossEntropyLoss(cls_weights)
    c = nn.CrossEntropyLoss()
    cls_loss = c(outputs, targets)
    #num_preds = torch.sigmoid(num_output)*5
    num_loss = F.mse_loss(num_output.squeeze(), num_target.float())
    return cls_loss + num_loss*0.01, cls_loss.item(), num_loss.item()
    '''
    if args.focal_loss:
        return focal_loss(outputs, targets)
    else:
        return c(outputs, targets)
    '''

def accuracy(output, label, topk=(1,10)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum().item()
        res.append(correct_k)
    return res

def create_single_class_model(args, num_classes=7172, prediction=False):
    #model = create_backbone_model(args.pretrained)
    model = InclusiveNet(backbone_name=args.backbone, pretrained=args.pretrained, num_classes=num_classes)
    if args.pretrained:
        model_file = os.path.join(MODEL_DIR, 'backbone', model.name, 'pretrained', 'best.pth')
    else:
        model_file = os.path.join(MODEL_DIR, 'backbone', model.name, 'scratch', 'best.pth')

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.init_ckp is not None:
        CKP = args.init_ckp
    else:
        CKP = model_file
    print('{}, exist: {}'.format(CKP, os.path.exists(CKP)))
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    elif prediction:
        raise ValueError('model file not exist')

    return model, model_file

def train(args):
    print('start training...')
    model, model_file = create_single_class_model(args)
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_val_loaders(args, batch_size=args.batch_size, dev_mode=args.dev_mode)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 
    best_cls_acc = 0.

    print('epoch |   lr    |   %        |  loss  |  avg   |  loss  |  cls   |  num   |  top1   | top5   |  best  |  time |  save  |')

    if not args.no_first_val:
        best_cls_acc, top1_acc, total_loss, cls_loss, num_loss = validate(args, model, val_loader, args.start_epoch)
        print('val   |         |            |        |        | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(
            total_loss, cls_loss, num_loss, top1_acc, best_cls_acc, best_cls_acc))

    if args.val:
        return

    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_cls_acc)
    else:
        lr_scheduler.step()
    train_iter = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            train_iter += 1
            img, target, num_target = data
            img, target, num_target = img.cuda(), target.cuda(), num_target.cuda()
            optimizer.zero_grad()
            output, num_output = model(img)
            
            loss, _, _ = criterion(args, output, target, num_output, num_target)
            loss.backward()
 
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:.5f} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1)), end='')

            if train_iter > 0 and train_iter % args.iter_val == 0:
                cls_acc, top1_acc, total_loss, cls_loss, num_loss = validate(args, model, val_loader, epoch=epoch)
                
                _save_ckp = ''
                if args.always_save or cls_acc > best_cls_acc:
                    best_cls_acc = cls_acc
                    torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                print('  {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} |  {:4s} |'.format(
            total_loss, cls_loss, num_loss, top1_acc, cls_acc, best_cls_acc, (time.time() - bg) / 60, _save_ckp))


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
    top1_corrects, corrects = 0, 0
    total_loss, cls_loss, num_loss = 0, 0, 0
    with torch.no_grad():
        for img, target, num_target in val_loader:
            img, target, num_target = img.cuda(), target.cuda(), num_target.cuda()
            output, num_output = model(img)
            loss, _cls_loss, _num_loss  = criterion(args, output, target, num_output, num_target)
            total_loss += loss.item()
            cls_loss += _cls_loss
            num_loss += _num_loss

            #preds = output.max(1, keepdim=True)[1]
            #corrects += preds.eq(target.view_as(preds)).sum().item()
            top1, top5 = accuracy(output, target)
            top1_corrects += top1
            corrects += top5
            total_num += len(img)
            
    top5_acc = corrects / total_num
    top1_acc = top1_corrects / total_num
    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    return top5_acc, top1_acc, total_loss/ n_batches, cls_loss / n_batches, num_loss / n_batches

def create_submission(args, predictions, outfile):
    meta = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    if args.dev_mode:
        meta = meta.iloc[:len(predictions)]  # for dev mode
    meta['labels'] = predictions
    meta.to_csv(outfile, index=False)

def predict(args):
    model, _ = create_model(args, prediction=True)
    model.eval()
    test_loader = get_test_loader(args, batch_size=args.batch_size, dev_mode=args.dev_mode)

    preds = None
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.cuda()
            #output = torch.sigmoid(model(x))
            output, _ = model(x)
            output = F.softmax(output, dim=1)
            _, pred = output.topk(3, 1, True, True)

            if preds is None:
                preds = pred.cpu()
            else:
                preds = torch.cat([preds, pred.cpu()], 0)
            print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')

    classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)
    label_names = []
    preds = preds.numpy()
    print(preds.shape)
    for row in preds:
        label_names.append(' '.join([classes[i] for i in row]))
    if args.dev_mode:
        print(len(label_names))
        print(label_names)

    create_submission(args, label_names, args.sub_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
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
    parser.add_argument('--t_max', default=12, type=int, help='lr scheduler patience')
    parser.add_argument('--init_ckp', default=None, type=str, help='resume from checkpoint path')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--cls_type', choices=['trainable', 'tuning'], type=str, default='trainable', help='train class type')
    parser.add_argument('--start_index', default=0, type=int, help='start index of classes')
    parser.add_argument('--end_index', default=7172, type=int, help='end index of classes')
    parser.add_argument('--max_labels', default=3, type=int, help='filter max labels')
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--sub_file', default='sub_backbone_4.csv', help='optimizer')
    parser.add_argument('--no_first_val', action='store_true')
    parser.add_argument('--always_save',action='store_true', help='alway save')
    #parser.add_argument('--img_sz', default=256, type=int, help='image size')
    
    args = parser.parse_args()
    print(args)

    if args.predict:
        predict(args)
    else:
        train(args)
