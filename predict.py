import os
import torch
import torch.nn as nn
import pandas as pd
import settings
from models import create_model
from loader import get_test_loader, get_val2_loader
from utils import get_classes
from train import validate

threshold = 0.05
batch_size = 100 #128

classes, _ = get_classes(settings.CLASSES_FILE)

def get_label_names(row):
    assert len(row) == settings.N_CLASSES
    label_names = [classes[i] for i in range(len(row)) if row[i] == 1]
    tmp = [classes[i] for i in range(len(row)) if row[i] != 0]
    assert len(tmp) == len(label_names)
    return ' '.join(label_names)

def create_submission(predictions, outfile):
    meta = pd.read_csv(settings.STAGE1_SAMPLE_SUB)
    meta['labels'] = predictions
    meta.to_csv(outfile, index=False)

def model_predict(model):
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    print('predicting...')
    print('loading {}...'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()

    #val2_loader = get_val2_loader(batch_size=batch_size)
    #validate(model, nn.BCEWithLogitsLoss(), val2_loader, batch_size)

    test_loader = get_test_loader(batch_size=batch_size, dev_mode=False)

    outputs = None
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.cuda()
            output = torch.sigmoid(model(x))

            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat([outputs, output])
            print('{}/{}'.format(batch_size*(i+1), test_loader.num), end='\r')
            if i == 0:
                break

    return outputs

def predict():
    model = create_model('resnet', 34, pretrained=True)
    outputs = model_predict(model)

    label_names = []
    pred = (outputs > threshold).byte().cpu().numpy()
    for row in pred:
        label_names.append(get_label_names(row))

    print(label_names)

    #create_submission(label_names, 'sub_res34_th015_1.csv')

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


if __name__ == '__main__':
    predict()
    #ensemble()
