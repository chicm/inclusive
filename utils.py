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

    #meta_file_name = os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    meta_file_name = settings.TRAIN_LABEL_FILE
    
    df_labels = pd.read_csv(meta_file_name, na_filter=False)
    split_index = int(df_labels.shape[0] * 0.9)

    df_train = df_labels.iloc[:split_index]

    #val_end_index = split_index
    #if df_labels.shape[0] < val_end_index:
    #    val_end_index = df_labels.shape[0]

    #print(split_index, val_end_index)
    df_val = df_labels.iloc[split_index:]
    print(df_train.shape, df_val.shape)

    return df_train, df_val

def get_tuning_meta(cls_type, start_index, end_index):
    '''
    Stage 1 validation meta
    '''
    assert cls_type in ['trainable', 'tuning']

    #meta_file_name = os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_{}_{}_{}.csv'.format(cls_type, start_index, end_index))
    meta_file_name = settings.TUNING_LABELS
    df_labels = pd.read_csv(meta_file_name, names=['ImageID', 'LabelName'], na_filter=False)
    print(df_labels.shape)
    return df_labels

def get_test_ids():
    test_df = pd.read_csv(settings.STAGE_1_SAMPLE_SUBMISSION)
    ids = test_df['image_id'].values.tolist()
    print(len(ids))
    return ids

def get_trainable_classes():
    return pd.read_csv(settings.CLASSES_TRAINABLE)['label_code'].values.tolist()

def test_bbox():
    df = pd.read_csv(settings.TRAIN_MACHINE_LABELS)
    print(df['LabelName'].nunique())
    print(df['ImageID'].nunique())

if __name__ == '__main__':
    #get_classes()
    #get_train_ids()
    #get_val_ids()
    #get_class_converter()
    #get_test_ids()
    #pass
    test_bbox()

    train_meta, val_meta = get_train_val_meta('trainable', 0, 250)
    print(train_meta.shape)
    print(train_meta.head())
    print(val_meta.shape)
    print(val_meta.head())
