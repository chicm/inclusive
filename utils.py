import pandas as pd
import settings


def get_train_meta():
    return pd.read_csv(settings.TRAIN_LABEL_FILE).iloc[:1350000]

def get_train_ids():
    train_df = get_train_meta()
    ids = train_df['ImageID'].values.tolist()
    print(len(ids))
    return ids

def get_val_meta():
    return pd.read_csv(settings.TRAIN_LABEL_FILE).iloc[1350000:1355000]

def get_val_ids():
    return get_val_meta()['ImageID'].values.tolist()

def get_test_ids():
    test_df = pd.read_csv(settings.STAGE1_SAMPLE_SUB)
    ids = test_df['image_id'].values.tolist()
    print(len(ids))
    return ids

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

if __name__ == '__main__':
    #get_classes()
    #get_train_ids()
    #get_val_ids()
    #get_class_converter()
    #get_test_ids()
    #pass
    print(get_val_meta().head(5))
