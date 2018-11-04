import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import settings
from models import create_model, InclusiveNet
from loader import get_test_loader, get_tuning_loader, get_train_val_loaders
from metrics import find_threshold, f2_score, find_fix_threshold, f2_scores
from utils import get_classes

def create_prediction_model(args):
    model = InclusiveNet(backbone_name=args.backbone, num_classes=args.end_index - args.start_index)
    sub_dir = '{}_{}_{}'.format(args.cls_type, args.start_index, args.end_index)

    model_file = os.path.join(settings.MODEL_DIR, model.name, sub_dir, args.ckp_name)

    if not os.path.exists(model_file):
        raise ValueError('model file {} does not exist'.format(model_file))
    print('loading {}...'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()

    return model, model_file

def find_best_thresholds(args, model, val_loader):
    print('finding thresholds with validation data')
    model.eval()
    targets = None
    outputs = None
    with torch.no_grad():
        for x, target, _ in val_loader:
            x, target = x.cuda(), target.cuda()
            output,_ = model(x)
            if targets is None:
                targets = target
            else:
                targets = torch.cat([targets, target])
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat([outputs, output])    

    #best_th = find_threshold(outputs, targets)
    
    #best_th[0] = 0.5
    #optimized_score = f2_score(targets, torch.sigmoid(outputs), threshold=torch.Tensor(best_th).cuda())

    if args.activation == 'sigmoid':
        preds = torch.sigmoid(outputs)
    elif args.activation == 'softmax':
        preds = F.softmax(outputs, dim=1)
    else:
        raise ValueError('error activate function')
    
    fix_th = find_fix_threshold(preds, targets)

    fix_score = f2_score(targets, (preds>fix_th).float())

    #print(f2_scores(outputs, targets))
    return fix_th, fix_score

def get_label_names(row, classes):
    assert len(row) == len(classes)
    label_names = [classes[i] for i in range(len(row)) if row[i] == 1]
    tmp = [classes[i] for i in range(len(row)) if row[i] != 0]
    assert len(tmp) == len(label_names)
    return ' '.join(label_names)

def model_predict(args, model, model_file, check, tta_num=2):
    model.eval()

    preds = []
    for flip_index in range(tta_num):
        test_loader = get_test_loader(args, batch_size=args.batch_size, dev_mode=args.dev_mode, tta_index=flip_index)

        outputs = None
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.cuda()
                output, _ = model(x)
                if args.activation == 'sigmoid':
                    output = torch.sigmoid(output)
                else:
                    output = F.softmax(output, dim=1)
                if outputs is None:
                    outputs = output
                else:
                    outputs = torch.cat([outputs, output])
                print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')
                if check and i == 0:
                    break
        
        preds.append(outputs.cpu().numpy())
        #return outputs
    results = np.mean(preds, 0)

    parent_dir = model_file+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred.npy')
    np.save(np_file, results)

    return results

def predict(args):
    model, model_file = create_prediction_model(args)
    
    if args.th > 0:
        fix_th = args.th
    else:
        if args.tuning_val:
            val_loader = get_tuning_loader(args, batch_size=args.batch_size)
        else:
            _, val_loader = get_train_val_loaders(args, batch_size=args.batch_size, val_num=20000)

        fix_th, fix_score = find_best_thresholds(args, model, val_loader)
        print('fixed th:', fix_th)
        print('fixed score:', fix_score)
    
    print('using threshold: {}'.format(fix_th))

    if args.val:
        return
    
    outputs = model_predict(args, model, model_file, args.check, tta_num=args.tta_num)

    classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)

    label_names = []
    pred = (outputs > fix_th).astype(np.uint8)
    for row in pred:
        label_names.append(get_label_names(row, classes))

    if args.check:
        print(label_names)
        return

    create_submission(args, label_names, args.sub_file)

def create_submission(args, predictions, outfile):
    meta = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    if args.dev_mode:
        meta = meta.iloc[:len(predictions)]  # for dev mode
    meta['labels'] = predictions
    meta.to_csv(outfile, index=False)


def ensemble_np(args):
    if args.th < 0:
        raise AssertionError('Please specify threshold')
    
    np_files = args.ensemble_np.split(',')
    if len(np_files) < 1:
        raise AssertionError('no np files')
    outputs = []
    for np_file in np_files:
        if not os.path.exists(np_file):
            raise AssertionError('np file does not exist')
        output = np.load(np_file)
        print(np_file, output.shape)
        outputs.append(output)
    ensemble_outputs = np.mean(outputs, 0)
    preds = (ensemble_outputs > args.th).astype(np.uint8)

    classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)
    label_names = []
    for row in preds:
        label_names.append(get_label_names(row, classes))

    if args.check:
        print(label_names[:10])
        return

    create_submission(args, label_names, args.sub_file)
    
def merge_dfs():
    df1 = pd.read_csv('sub_tuning_0_100.csv', index_col='image_id', na_filter=False)
    df1.columns = ['label1']
    df2 = pd.read_csv('sub_res34_100_200_th_020.csv', index_col='image_id', na_filter=False)
    df2.columns = ['label2']
    df3 = df1.join(df2)
    df3['labels'] = df3.label1.astype(str).str.cat(df3.label2.astype(str), sep=' ')

    print(df3.head())
    df3.to_csv('merged_tuning_0_200_2.csv', columns=['labels'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--backbone', default='se_resnext50_32x4d', type=str, help='model name')
    parser.add_argument('--ckp_name', type=str, default='best_pretrained.pth',help='check point file name')
    parser.add_argument('--check',action='store_true', help='check only')
    parser.add_argument('--tuning_val',action='store_true', help='check only')
    parser.add_argument('--val',action='store_true', help='check only')
    parser.add_argument('--th', type=float, default=-1, help='threshold')
    parser.add_argument('--cls_type',  default='trainable', choices=['trainable', 'tuning'], type=str, help='class type')
    parser.add_argument('--start_index', type=int, default=0, help='start index of classes')
    parser.add_argument('--end_index', type=int, default=7172, help='end index of classes')
    parser.add_argument('--sub_file', required=True, type=str, help='submission file name')
    parser.add_argument('--activation', choices=['softmax', 'sigmoid'], type=str, default='softmax', help='activation')
    parser.add_argument('--dev_mode', action='store_true')
    parser.add_argument('--tta_num', type=int, default=4, help='tta number')
    parser.add_argument('--ensemble_np', default=None, type=str, help='ensemble np files')
    args = parser.parse_args()

    print(args)

    if args.ensemble_np is not None:
        ensemble_np(args)
    else:
        predict(args)

    #merge_dfs()
