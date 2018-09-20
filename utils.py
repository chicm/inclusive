import pandas as pd
import settings

def get_train_ids():
    train_df = pd.read_csv(settings.TRAIN_LABEL_FILE)
    ids = train_df['ImageID'].values.tolist()
    print(len(ids))
    return ids

def get_train_meta():
    return pd.read_csv(settings.TRAIN_LABEL_FILE)

def get_val_ids():
    val_df = pd.read_csv(settings.VAL_LABEL_FILE, names=['ImageID', 'LabelName'])
    print(val_df.shape)
    print(val_df.values[:5])
    ids = val_df['ImageID'].values.tolist()
    print(len(ids))
    return ids

def get_val_meta():
    return pd.read_csv(settings.VAL_LABEL_FILE, names=['ImageID', 'LabelName'])

def get_classes():
    classes = pd.read_csv(settings.CLASSES_FILE)['label_code'].values.tolist()
    print(len(classes))
    print(classes[:10])
    return classes

def get_class_converter():
    classes = get_classes()
    stoi = {classes[i]: i for i in range(len(classes))}
    print(classes[:5])
    print(stoi['/m/010jjr'])
    return classes, stoi

if __name__ == '__main__':
    #get_classes()
    #get_train_ids()
    #get_val_ids()
    get_class_converter()