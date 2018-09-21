import os
import torch
import pandas as pd
import settings
from models import create_res50
from loader import get_test_loader
from utils import get_classes

threshold = 0.4
batch_size = 128

classes = get_classes(settings.CLASSES_FILE)

def get_label_names(row):
    #label_names = ['' if row[i] == 0 else classes[i] for i in len(row)]
    assert len(row) == 100
    label_names = [classes[i] for i in range(len(row)) if row[i] == 1]
    tmp = [classes[i] for i in range(len(row)) if row[i] != 0]
    assert len(tmp) == len(label_names)
    #label_names = [ x if len(x) > 0 for x in label_names]
    return ' '.join(label_names)


def create_submission(predictions):
    meta = pd.read_csv(settings.STAGE1_SAMPLE_SUB)
    meta['labels'] = predictions
    meta.to_csv('sub1.csv', index=False)

def predict():
    model = create_res50()
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    print('predicting...')
    print('loading {}...'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()

    test_loader = get_test_loader(batch_size=batch_size)

    label_names = []
    #preds = None
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.cuda()
            output = model(x)

            pred = (torch.sigmoid(output) > threshold).byte().cpu().numpy()
            for row in pred:
                label_names.append(get_label_names(row))

            #if preds is None:
            #    preds = pred
            #else:
            #    preds = torch.cat([preds, pred])
            print('{}/{}'.format(batch_size*(i+1), test_loader.num), end='\r')
    #print(preds.size())
    #preds = preds.cpu().numpy().tolist()

    #label_names = []
    #for row in preds:
    #    label_names.append(get_label_names(row))
    create_submission(label_names)

if __name__ == '__main__':
    predict()