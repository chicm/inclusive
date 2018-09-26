import os
import torch
import torch.nn as nn
import settings
import pandas as pd
from models import create_resnet_model

def concat_sub(sub_file1, sub_file2, out_file):
    df1 = pd.read_csv(sub_file1)
    df2 = pd.read_csv(sub_file2)

    img_ids = df1['image_id'].values
    print(img_ids.shape)

    new_labels = []
    for v1, v2 in zip(df1['labels'].astype('str').values, df2['labels'].astype('str').values):
        new_labels.append(' '.join([v1, v2]))

    df3 = df1.copy()
    df3['labels'] = new_labels
    df3.to_csv(out_file, index=False)

def convert_model():
    res50 = create_resnet_model(50)
    res50.load_state_dict(torch.load(os.path.join(settings.MODEL_DIR, 'res50', 'best_lb029.pth')))
    num_ftrs = res50.fc.in_features
    res50.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 100)) 

    torch.save(res50.state_dict(), os.path.join(settings.MODEL_DIR, 'res50_2', 'best_lb029.pth'))


if __name__ == '__main__':
    #concat_sub('sub1.csv', 'sub1_naive.csv', 'merged1.csv')
    convert_model()