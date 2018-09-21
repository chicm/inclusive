import pandas as pd
import settings

def get_train_ids():
    train_df = pd.read_csv(settings.TRAIN_LABEL_FILE)
    ids = train_df['ImageID'].values.tolist()[:1350000]
    print(len(ids))
    return ids

def get_train_meta():
    return pd.read_csv(settings.TRAIN_LABEL_FILE)

def get_val_ids():
    return get_train_ids()[1350000:1355000]

def get_val_meta():
    return get_train_meta().iloc[1350000:1355000]

def get_test_ids():
    test_df = pd.read_csv(settings.STAGE1_SAMPLE_SUB)
    ids = test_df['image_id'].values.tolist()
    print(len(ids))
    return ids

def get_val2_ids():
    val_df = pd.read_csv(settings.VAL2_LABEL_FILE, names=['ImageID', 'LabelName'])
    print(val_df.shape)
    print(val_df.values[:5])
    ids = val_df['ImageID'].values.tolist()
    print(len(ids))
    return ids

def get_val2_meta():
    meta = pd.read_csv(settings.VAL2_LABEL_FILE, names=['ImageID', 'LabelName'])
    print(meta.shape)
    print(meta.head(5))
    return meta

def get_classes(class_file_name):
    classes = pd.read_csv(class_file_name)['label_code'].values.tolist()
    print(len(classes))
    print(classes[:10])
    return classes

def get_class_stoi(classes):
    stoi = {classes[i]: i for i in range(len(classes))}
    print(classes[:5])
    #print(stoi['/m/010jjr'])
    return stoi

if __name__ == '__main__':
    #get_classes()
    #get_train_ids()
    #get_val_ids()
    #get_class_converter()
    #get_test_ids()
    #pass
    print(get_val_meta().head(5))
