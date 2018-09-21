import settings
import pandas as pd

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

if __name__ == '__main__':
    concat_sub('sub1.csv', 'sub1_naive.csv', 'merged1.csv')