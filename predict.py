import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import settings
from models import create_model
from loader import get_test_loader, get_tuning_loader, get_train_val_loaders
from metrics import find_threshold, f2_score
from utils import get_classes

def create_prediction_model(args):
    model = create_model('resnet', args.layers, pretrained=not args.scratch, num_classes=args.end_index-args.start_index)
    sub_dir = '{}_{}_{}'.format(args.cls_type, args.start_index, args.end_index)
    model_file = os.path.join(settings.MODEL_DIR, model.name, sub_dir, 'best.pth')
    if not os.path.exists(model_file):
        raise ValueError('model file not exist: {}'.format(model_file))
    print('model file: {}'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()

    return model

def find_best_thresholds(model, val_loader):
    print('finding thresholds with validation data')
    model.eval()
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

    best_th = find_threshold(outputs, targets)
    #best_th[0] = 0.5
    optimized_score = f2_score(targets, torch.sigmoid(outputs), threshold=torch.Tensor(best_th).cuda())
    return best_th, optimized_score

def get_label_names(row, classes):
    assert len(row) == len(classes)
    label_names = [classes[i] for i in range(len(row)) if row[i] == 1]
    tmp = [classes[i] for i in range(len(row)) if row[i] != 0]
    assert len(tmp) == len(label_names)
    return ' '.join(label_names)

def model_predict(args, model):
    model.eval()
    test_loader = get_test_loader(args, batch_size=args.batch_size, dev_mode=False)

    outputs = None
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.cuda()
            output = torch.sigmoid(model(x))

            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat([outputs, output])
            print('{}/{}'.format(args.batch_size*(i+1), test_loader.num), end='\r')

            #if i == 0:
            #    break
 
    return outputs

def predict(args):
    model = create_prediction_model(args)
    _, val_loader = get_train_val_loaders(args, batch_size=args.batch_size)
    #val_loader = get_tuning_loader(args, batch_size=args.batch_size)

    thresholds, tuning_f2 = find_best_thresholds(model, val_loader)
    print(thresholds)
    print('optimized tuning f2:', tuning_f2)

    if args.val:
        return

    outputs = model_predict(args, model)

    classes, _ = get_classes(args.cls_type, args.start_index, args.end_index)

    label_names = []
    pred = (outputs > torch.Tensor(thresholds).cuda()).byte().cpu().numpy()
    for row in pred:
        label_names.append(get_label_names(row, classes))

    #print(label_names)

    create_submission(label_names, args.out_file)

def create_submission(predictions, outfile):
    meta = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    meta['labels'] = predictions
    meta.to_csv(outfile, index=False)

'''
def ensemble():
    outputs = []
    for layer in [18, 34, 50]:
        model = create_model('resnet', layer, pretrained=True)
        output = model_predict(model)
        outputs.append(output)
    mean_output = torch.mean(torch.stack(outputs), 0)

    label_names = []
    pred = (mean_output > threshold).byte().cpu().numpy()
    for row in pred:
        label_names.append(get_label_names(row))

    #print(label_names)
    create_submission(label_names)
'''

def merge_dfs():
    df1 = pd.read_csv('res34_optim_threshold.csv', index_col='image_id')
    df1.columns = ['label1']
    df2 = pd.read_csv('sub_res34_50_200.csv', index_col='image_id')
    df2.columns = ['label2']
    df3 = df1.join(df2)
    df3['labels'] = df3.label1.astype(str).str.cat(df3.label2.astype(str), sep=' ')

    print(df3.head())
    df3.to_csv('merged_0_200_2.csv', columns=['labels'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--model_name', default='resnet', type=str, help='model name')
    parser.add_argument('--layers', default=34, type=int, help='batch size')
    parser.add_argument('--scratch',action='store_true', help='train from scratch')
    parser.add_argument('--val',action='store_true', help='val only')
    parser.add_argument('--cls_type', choices=['trainable', 'tuning'], type=str, required=True, help='class type')
    parser.add_argument('--start_index', type=int, required=True, help='start index of classes')
    parser.add_argument('--end_index', type=int, required=True, help='end index of classes')
    parser.add_argument('--out_file', default='sub_res34_50_200.csv', type=str, help='submission file name')
    args = parser.parse_args()

    #predict(args)
    merge_dfs()