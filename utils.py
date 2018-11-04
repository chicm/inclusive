import os
import numpy as np
import pandas as pd
import settings

def get_cls_counts(classes, cls_type):
    if cls_type == 'trainable':
        df_counts = pd.read_csv(settings.SORTED_CLASSES_TRAINABLE).set_index('label_code')
    elif cls_type == 'tuning':
        df_counts = pd.read_csv(settings.SORTED_TUNING_CLASSES).set_index('label_code')
    else:
        raise ValueError('class type error')

    cls_counts = df_counts.to_dict()['counts']
    counts = [cls_counts[x] for x in classes]
    return counts

def get_classes(cls_type, start_index, end_index):
    if cls_type == 'trainable':
        file_name = settings.SORTED_CLASSES_TRAINABLE
    elif cls_type == 'tuning':
        file_name = settings.SORTED_TUNING_CLASSES
    classes = pd.read_csv(file_name)['label_code'].values.tolist()[start_index: end_index]
    #print(len(classes))
    stoi = {classes[i]: i for i in range(len(classes))}
    return classes, stoi

def get_train_val_meta(cls_type, start_index, end_index):
    assert cls_type in ['trainable', 'tuning']

    #meta_file_name = os.path.join(settings.  f, 'train_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    meta_file_name = settings.TRAIN_LABEL_FILE_COUNTS
    
    df_labels = pd.read_csv(meta_file_name, na_filter=False)
    split_index = int(df_labels.shape[0] * 0.9)

    df_train = df_labels.iloc[:split_index]

    #val_end_index = split_index
    #if df_labels.shape[0] < val_end_index:
    #    val_end_index = df_labels.shape[0]

    #print(split_index, val_end_index)
    df_val = df_labels.iloc[split_index:]
    #print(df_train.shape, df_val.shape)

    return df_train, df_val

def get_tuning_meta(cls_type, start_index, end_index):
    '''
    Stage 1 validation meta
    '''
    assert cls_type in ['trainable', 'tuning']

    #meta_file_name = os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    meta_file_name = settings.TUNING_LABELS_COUNTS
    df_labels = pd.read_csv(meta_file_name, na_filter=False)
    print(df_labels.shape)
    return df_labels

def get_test_ids():
    test_df = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    ids = test_df['image_id'].values.tolist()
    print(len(ids))
    return ids

def get_trainable_classes():
    return pd.read_csv(settings.CLASSES_TRAINABLE)['label_code'].values.tolist()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_weights_by_counts_old(counts, max_weight=100):
    counts = np.sqrt(counts)
    counts = np.array(counts)
    #counts = sigmoid(counts)
    #print(counts)
    v_min = np.min(counts)
    v_max = np.max(counts)
    counts = (counts ) / (v_max - v_min)
    #print(counts)
    counts = 1 / counts
    counts = np.clip(counts, 0, max_weight)
    
    #print(counts.tolist()[:100])
    #counts = ((counts * max_scale) + 1) / max_scale
    #counts = np.sqrt(counts)
    #print(counts)
    return counts

def get_weights_by_counts(counts, max_weight=100):
    counts = np.array(counts)
    
    counts = np.clip(counts, 50, 10000)
    counts = 10000 / counts
    
    #print(counts.tolist()[:100])
    #counts = ((counts * max_scale) + 1) / max_scale
    #counts = np.sqrt(counts)
    #print(counts)
    return counts



def test_bbox():
    df = pd.read_csv(settings.TRAIN_MACHINE_LABELS)
    print(df['LabelName'].nunique())
    print(df['ImageID'].nunique())

def test_cls_weights():
    classes, _ = get_classes('trainable', 0, 7172)
    cnts = get_cls_counts(classes, 'trainable')
    cls_weights = get_weights_by_counts(cnts, max_weight=20)
    print(cls_weights[-500:])

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
    test_cls_weights()

    c, s = get_classes('trainable', 0, 1000)
    cnt = get_cls_counts(c, 'trainable')
    #print(cnt[:100])
    #w = get_weights_by_counts(cnt)
    #print(w[:100])

    #get_train_ids()
    #get_val_ids()
    #get_class_converter()
    #get_test_ids()
    #pass
    #test_bbox()

    #train_meta, val_meta = get_train_val_meta('trainable', 0, 250)
    #print(train_meta.shape)
    #print(train_meta.head())
    #print(val_meta.shape)
    #print(val_meta.head())
