import os
import argparse
import logging as log
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from loader import get_train_val_loaders, get_tuning_loader
import settings
from metrics import accuracy, f2_scores, f2_score, accuracy_th, find_fix_threshold
from models import create_model, AttentionResNet

def weighted_bce(args, x, y):

    w = y*args.pos_weight + 1

    return F.binary_cross_entropy_with_logits(x, y, w)

def train(args):
    model = create_model(args.backbone, pretrained=args.pretrained, num_classes=args.end_index-args.start_index, load_backbone_weights=True)
    sub_dir = '{}_{}_{}'.format(args.cls_type, args.start_index, args.end_index)

    if args.pretrained:
        model_file = os.path.join(settings.MODEL_DIR, model.name, sub_dir, 'best_pretrained.pth')
    else:
        model_file = os.path.join(settings.MODEL_DIR, model.name, sub_dir, 'best_scratch.pth')

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()
    #criterion = nn.BCEWithLogitsLoss()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=0.0001, lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0001, lr=args.lr)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, min_lr=args.min_lr)

    train_loader, val_loader = get_train_val_loaders(args, batch_size=args.batch_size)
    #val_loader = get_tuning_loader(args, batch_size=args.batch_size)
    model.train()

    iteration = 0

    print('epoch | itr |   lr    |   %             |  loss  |  avg   |  loss  | optim f2 |  15 f2  | 10 f2  |  best f2  |  thresh  |  time | save |')

    best_val_loss, best_val_score, th = validate(argsm model, criterion, val_loader, args.batch_size, args.no_score)

    print('val   |     |         |                 |        |        | {:.4f} | {:.4f}   |         |        |  {:.4f}   |   {:.3f} |       |'.format(
        best_val_loss, best_val_score, best_val_score, th))

    if args.val:
        return

    lr_scheduler.step(best_val_score - best_val_loss)
    model.train()

    bg = time.time()
    current_lr = get_lrs(optimizer) 
    for epoch in range(args.epochs):
        train_loss = 0
        if epoch > 0 and epoch % 20 == 0:
            train_loader, _ = get_train_val_loaders(args, batch_size=args.batch_size)
        for batch_idx, data in enumerate(train_loader):
            iteration += 1
            x, target = data
            x, target = x.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(x)
            #loss = focal_loss(output, target)
            #loss = criterion(output, target)
            loss = weighted_bce(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} |{:4d} | {:.5f} | {:7d}/{:7d} | {:.4f} | {:.4f} |'
                    .format(epoch, iteration, float(current_lr[0]), args.batch_size*(batch_idx+1), train_loader.num,
                    loss.item(), train_loss/(batch_idx+1)), end='')

            if iteration % args.iter_save == 0:
                val_loss, val_score, th = validate(args, model, criterion, val_loader, args.batch_size, args.no_score)
                model.train()
                _save_ckp = ''

                if val_score - val_loss > best_val_score - best_val_loss:
                    best_val_score = val_score
                    best_val_loss = val_loss

                    torch.save(model.state_dict(), model_file)
                    _save_ckp = '*'
                
                lr_scheduler.step(val_score-val_loss)
                current_lr = get_lrs(optimizer) 

                print(' {:.4f} | {:.4f}   |         |        |  {:.4f}   |  {:.3f} | {:.1f}  | {:4s} |'.format(
                    val_loss, val_score, best_val_score, th, (time.time() - bg) / 60, _save_ckp))
                bg = time.time()

def validate(args, model, criterion, val_loader, batch_size, no_score=False):
    #print('\nvalidating...')
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
            #loss = criterion(output, target)
            loss = weighted_bce(output, target)
            #loss = focal_loss(output, target)
            val_loss += loss.item()
            
    val_loss = val_loss / (val_loader.num/batch_size)
    optimized_score = 0.

    if not no_score:
        best_th = find_fix_threshold(outputs, targets)
        optimized_score = f2_score(targets, torch.sigmoid(outputs), threshold=best_th)
        #optimized_score = f2_score(targets, torch.sigmoid(outputs), best_th)

    #print(best_th)
    #print('optimized score:', optimized_score)
    #print('optimized acc:', accuracy_th(outputs, targets, torch.Tensor(best_th).cuda()))

    #acc = accuracy(outputs, targets)
    #print('acc:', acc)
    #acc_sum = sum([acc[i][2] for i in range(1,7)])

    #score = f2_scores(outputs, targets)
    #print('f2 scores:', score)
    #score_sum = sum([score[i][1] for i in range(1,7)])
    #print('val loss: {:.4f}, threshold f2 score: {:.4f}, threshold acc: {:.4f}'
    #    .format(val_loss, score_sum, acc_sum))
    #log.info(str(optimized_score))
    return val_loss, optimized_score, best_th

       
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--optim', choices=['Adam', 'SGD'], type=str, default='SGD', help='optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
    parser.add_argument('--batch_size', default=96, type=int, help='batch size')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='backbone')
    parser.add_argument('--layers', default=34, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='epochs')
    parser.add_argument('--iter_save', default=200, type=int, help='epochs')
    parser.add_argument('--scratch',action='store_true', help='pretrained')
    parser.add_argument('--val',action='store_true', help='val only')
    parser.add_argument('--cls_type', choices=['trainable', 'tuning'], type=str, default='trainable', help='train class type')
    parser.add_argument('--start_index', default=0, type=int, help='start index of classes')
    parser.add_argument('--end_index', default=100, type=int, help='end index of classes')
    parser.add_argument('--no_score',action='store_true', help='do not calculate f2 score')
    parser.add_argument('--pretrained',action='store_true',help='backbone use pretrained model')
    parser.add_argument('--pos_weight', default=20, type=int, help='end index of classes')
    args = parser.parse_args()

    print(args)

    #log.basicConfig(
    #    filename = 'trainlog.txt', 
    #    format   = '%(asctime)s : %(message)s',
    #    datefmt  = '%Y-%m-%d %H:%M:%S', 
    #    level = log.INFO)

    train(args)
