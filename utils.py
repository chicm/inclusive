import os
import pandas as pd
import settings


def get_classes(cls_type, start_index, end_index):
    if cls_type == 'trainable':
        file_name = settings.SORTED_CLASSES_TRAINABLE
    elif cls_type == 'tuning':
        file_name = settings.SORTED_TUNING_CLASSES
    classes = pd.read_csv(file_name)['label_code'].values.tolist()[start_index: end_index]
    print(len(classes))
    stoi = {classes[i]: i for i in range(len(classes))}
    return classes, stoi

def get_train_val_meta(cls_type, start_index, end_index):
    assert cls_type in ['trainable', 'tuning']

    meta_file_name = os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    df_labels = pd.read_csv(meta_file_name, na_filter=False)
    split_index = int(df_labels.shape[0] * 0.9)

    df_train = df_labels.iloc[:split_index]

    val_end_index = split_index + 25000
    if df_labels.shape[0] < val_end_index:
        val_end_index = df_labels.shape[0]

    #print(split_index, val_end_index)
    df_val = df_labels.iloc[split_index: val_end_index]
    print(df_train.shape, df_val.shape)

    return df_train, df_val

def get_tuning_meta(cls_type, start_index, end_index):
    '''
    Stage 1 validation meta
    '''
    assert cls_type in ['trainable', 'tuning']

    meta_file_name = os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    df_labels = pd.read_csv(meta_file_name, names=['ImageID', 'LabelName'], na_filter=False)
    print(df_labels.shape)
    return df_labels

def get_test_ids():
    test_df = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    ids = test_df['image_id'].values.tolist()
    print(len(ids))
    return ids

'''
def get_val2_ids():
    val_df = pd.read_csv(settings.VAL2_LABEL_FILE, names=['ImageID', 'LabelName'])
    print(val_df.shape)
    #print(val_df.values[:5])
    ids = val_df['ImageID'].values.tolist()
    #print(len(ids))
    return ids

def get_val2_meta():
    meta = pd.read_csv(settings.VAL2_LABEL_FILE, names=['ImageID', 'LabelName'])
    print(meta.shape)
    return meta

def get_classes(class_file_name=settings.CLASSES_FILE):
    classes = pd.read_csv(class_file_name)['label_code'].values.tolist()
    print(len(classes))
    #print(classes[:10])
    stoi = {classes[i]: i for i in range(len(classes))}
    return classes, stoi
'''


if __name__ == '__main__':
    #get_classes()
    #get_train_ids()
    #get_val_ids()
    #get_class_converter()
    #get_test_ids()
    #pass
    train_meta, val_meta = get_train_val_meta('trainable', 0,50)
    print(train_meta.shape)
    print(train_meta.head())
    print(val_meta.shape)
    print(val_meta.head())
