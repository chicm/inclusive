import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import get_classes
import settings

bad_ids = ['a251467db63ddc0c', 'a254fdb8377c32ac', 'a256e3fc24eb2692']

def _generate_train_label(classes, output_file):
    print('creating train labels:', len(classes), output_file)
    df_human_labels = pd.read_csv(settings.TRAIN_HUMAN_LABELS) #, index_col='ImageID')
    print('total human labels:', df_human_labels.shape)
    print('total classes:', df_human_labels.LabelName.unique().shape)
    df_human_labels = df_human_labels[df_human_labels['LabelName'].isin(classes)]

    print('trainable labels for specified classes:', df_human_labels.shape)
    print('trainable classes:', df_human_labels.LabelName.unique().shape)
    df_human_labels = df_human_labels[df_human_labels['Confidence'] == 1]

    print('conf1:', df_human_labels.shape)
    print(df_human_labels.LabelName.unique().shape)

    df_human_labels = df_human_labels[~df_human_labels.ImageID.isin(bad_ids)]
    print('trainable labels:', df_human_labels.shape)
    print('trainable images:', df_human_labels.ImageID.unique().shape)
    print('trainable classes:', df_human_labels.LabelName.unique().shape)

    df = df_human_labels.groupby('ImageID')['LabelName'].apply(' '.join).reset_index()
    print('trainable images:', df.shape)
    df = shuffle(df)
    print('shuffled trainable images:', df.shape)
    print('saving:', output_file)
    df.to_csv(output_file, index=False)

def generate_train_labels():
    if not os.path.exists(settings.TRAIN_LABEL_DIR):
        os.makedirs(settings.TRAIN_LABEL_DIR)
    
    top_classes, _ = get_classes('trainable', 0, 50)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_trainable_0_50.csv'))
    top_classes, _ = get_classes('trainable', 50, 200)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_trainable_50_200.csv'))
    top_classes, _ = get_classes('trainable', 200, 500)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_trainable_200_500.csv'))
    top_classes, _ = get_classes('trainable', 500, 2000)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_trainable_500_2000.csv'))

    top_classes, _ = get_classes('tuning', 0, 100)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_tuning_0_100.csv'))
    top_classes, _ = get_classes('tuning', 100, 484)
    _generate_train_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'train_labels_tuning_100_484.csv'))
    

    #print(df.head())
    #print(df.shape)
    #df = df[~df['ImageID'].isin(bad_ids)]
    #print(df.shape)
    #df.to_csv(output_file, index=False)
    #print(df2[df2['E'].isin(['two','four'])])

def find_no_exist_train_files():
    bad_ids = []
    df = pd.read_csv(settings.TRAIN_LABEL_FILE)
    ids = df['ImageID'].values.tolist()
    for i, img_id in enumerate(ids):
        fn = os.path.join(settings.TRAIN_IMG_DIR, '{}.jpg'.format(img_id))
        if not os.path.isfile(fn):
            print(img_id)
            bad_ids.append(img_id)
        if i % 100000 == 0:
            print(i)
    print(bad_ids)
    
def test():
    from io import StringIO

    data = StringIO("""
    name1,"hej","2014-11-01"
    name1,"du","2014-11-02"
    name1,"aj","2014-12-01"
    name1,"oj","2014-12-02"
    name2,"fin","2014-11-01"
    name2,"katt","2014-11-02"
    name2,"mycket","2014-12-01"
    name2,"lite","2014-12-01"
    """)

    # load string as stream into dataframe
    df = pd.read_csv(data,header=0, names=["name","text","date"],parse_dates=[2])
    print(df)

    # add column with month
    #df["month"] = df["date"].apply(lambda x: x.month)
    #print(df.groupby('name'))
    #df2 = df.groupby('name')['text'].agg(lambda x: ','.join(x))
    df2 = df.groupby('name')['text'].apply(','.join).reset_index()
    print(type(df2))
    print(df2)

def check_val_count():
    tuning_labels = pd.read_csv(settings.VAL_LABEL_FILE, names=['id', 'labels'], index_col=['id'])
    df_counts = tuning_labels['labels'].str.split().apply(pd.Series).stack().value_counts()
    total = df_counts.values.sum()
    topn = df_counts.head(100).values.sum()
    print(df_counts.shape, total, topn)

    classes = tuning_labels['labels'].str.split().apply(pd.Series).stack().value_counts().head(100).index.tolist()
    print(classes[:10])
    df2 = pd.DataFrame(classes, columns=['label_code'])
    df2.to_csv(settings.TOP100_VAL_CLASS_FILE, index=False)

def check_train_count():
    labels = pd.read_csv(settings.TRAIN_LABEL_FILE)
    classes = labels['LabelName'].str.split().apply(pd.Series).stack().value_counts().head(200).index.tolist()
    df2 = pd.DataFrame(classes, columns=['label_code'])
    df2.to_csv(settings.TOP200_TRAIN_CLASS_FILE, index=False)
    #total = df_counts.values.sum()
    #topn = df_counts.head(200).values.sum()
    #print(df_counts.shape, total, topn)

def check_class_intersection():
    df1 = pd.read_csv(settings.TOP100_VAL_CLASS_FILE)
    df2 = pd.read_csv(settings.TOP200_TRAIN_CLASS_FILE)
    common = set(df1['label_code'].values.tolist()) & set(df2['label_code'].values.tolist())
    print(list(common)[:10])
    print(len(common))

def generate_topk_class():
    df1 = pd.read_csv(settings.TOP100_VAL_CLASS_FILE)
    df2 = pd.read_csv(settings.TOP200_TRAIN_CLASS_FILE)
    u = set(df1['label_code'].values.tolist()) | set(df2['label_code'].values.tolist())
    df3 = pd.DataFrame(list(u), columns=['label_code'])
    df3.to_csv(settings.TOP272_CLASS_FILE, index=False)

# create stage 1 validation labels for specified classes
def _generate_tuning_label(classes, out_file):
    df_turning_labels = pd.read_csv(settings.TUNING_LABELS, names=['ImageID', 'LabelName'])
    img_ids = []
    labels = []
    for row in df_turning_labels.values:
        filtered_label = [x for x in row[1].split() if x in classes]
        #if len(filtered_label) > 0:
        img_ids.append(row[0])
        labels.append(' '.join(filtered_label))
    filtered_df = pd.DataFrame({'image_id': img_ids, 'labels': labels})
    filtered_df.to_csv(out_file, header=None, index=False)

def create_tuning_labels():
    top100_classes, _ = get_classes('tuning', 0, 100)
    _generate_tuning_label(top100_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_tuning_0_100.csv'))
    classes_384, _ = get_classes('tuning', 100, 484)
    _generate_tuning_label(classes_384, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_tuning_100_484.csv'))

    top_classes, _ = get_classes('trainable', 0, 50)
    _generate_tuning_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_trainable_0_50.csv'))
    top_classes, _ = get_classes('trainable', 50, 200)
    _generate_tuning_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_trainable_50_200.csv'))
    top_classes, _ = get_classes('trainable', 200, 500)
    _generate_tuning_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_trainable_200_500.csv'))
    top_classes, _ = get_classes('trainable', 500, 2000)
    _generate_tuning_label(top_classes, os.path.join(settings.TRAIN_LABEL_DIR, 'tuning_labels_trainable_500_2000.csv'))

def create_class_counts_df():
    classes = pd.read_csv(settings.CLASSES_TRAINABLE)['label_code'].values.tolist()
    df_labels = pd.read_csv(settings.TRAIN_HUMAN_LABELS, index_col=['LabelName'])
    df_labels = df_labels[df_labels.index.isin(classes)]
    df_counts = df_labels.index.value_counts().to_frame(name='counts')
    df_counts.index.name = 'label_code'
    df_counts['index_copy'] = df_counts.index
    df_counts = df_counts.sort_values(['counts', 'index_copy'], ascending=False)
    df_counts.to_csv(settings.SORTED_CLASSES_TRAINABLE, columns=['counts'])

def create_tuning_class_counts_df():
    classes = pd.read_csv(settings.CLASSES_TRAINABLE)['label_code'].values.tolist()
    tuning_labels = pd.read_csv(settings.TUNING_LABELS, names=['id', 'labels'])

    df_counts = tuning_labels['labels'].str.split().apply(pd.Series).stack().value_counts().to_frame(name='counts')
    df_counts.index.name = 'label_code'
    df_counts['index_copy'] = df_counts.index
    df_counts = df_counts.sort_values(['counts', 'index_copy'], ascending=False)
    print(df_counts.head(), df_counts.shape)
    df_counts = df_counts[df_counts.index.isin(classes)]
    print(df_counts.shape)
    df_counts.to_csv(settings.SORTED_TUNING_CLASSES, columns=['counts'])

'''
    print(df_labels.shape)
    df_labels = df_labels[df_labels.LabelName.isin(classes)]
    print(df_labels.shape)
    df_counts = df_labels.LabelName.value_counts().to_frame(name='counts')
    df_counts.index.name = 'label_code'
    df_counts.to_csv(os.path.join(settings.DATA_DIR, 'sorted-tuning-classes.csv'))
'''

def test_stratify():
    X = [1, 1, 1, 2,2,2,3,3,3,4,4,4]
    Y = ['a'] * 6 + ['b']*6
    print([i for i in zip(X,Y)])

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y)
    print('x_train:', x_train)
    print('y_train:', y_train)
    print('x_val:', x_val)
    print('y_val:', y_val)


if __name__ == '__main__':
    #test_stratify()
    #check_val_count()
    #generate_val2_label()
    #generate_train_split_from_human(get_classes(settings.CLASSES_FILE), settings.TRAIN_LABEL_FILE)
    #find_no_exist_train_files()
    
    #check_train_count()
    #check_class_intersection()
    #generate_topk_class()
    #create_class_counts_df()
    #create_tuning_class_counts_df()
    create_tuning_labels()
    generate_train_labels()