import os
import torch
import pandas as pd
import settings
from models import create_res50, create_resnet_model
from loader import get_test_loader
from utils import get_classes

threshold = 0.15
batch_size = 128

classes = get_classes(settings.CLASSES_FILE)

def get_label_names(row):
    assert len(row) == 100
    label_names = [classes[i] for i in range(len(row)) if row[i] == 1]
    tmp = [classes[i] for i in range(len(row)) if row[i] != 0]
    assert len(tmp) == len(label_names)
    return ' '.join(label_names)

def create_submission(predictions):
    meta = pd.read_csv(settings.STAGE1_SAMPLE_SUB)
    meta['labels'] = predictions
    meta.to_csv('sub1.csv', index=False)

def model_predict(model):
    model_file = os.path.join(settings.MODEL_DIR, model.name, 'best.pth')
    print('predicting...')
    print('loading {}...'.format(model_file))
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    model.eval()

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

    return outputs

def predict():
    model = create_resnet_model(50)
    outputs = model_predict(model)

    label_names = []
    pred = (outputs > threshold).byte().cpu().numpy()
    for row in pred:
        label_names.append(get_label_names(row))

    #print(label_names)

    create_submission(label_names)

def ensemble():
    outputs = []
    for layer in [34, 50]:
        model = create_resnet_model(layer)
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
    #predict()
    ensemble()