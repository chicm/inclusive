import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import get_classes, get_trainable_classes
import settings

bad_ids = ['a251467db63ddc0c', 'a254fdb8377c32ac', 'a256e3fc24eb2692']

# Not found trainalbe classes in human labels:
# '/m/07r2x', '/m/06f8q', '/m/0brl6', '/m/01dgzv', '/m/04q347y', '/m/01wr8'

def _generate_train_label(classes, output_file):
    #print('creating train labels:', len(classes), output_file)
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
    df = df.sort_values(['ImageID'])
    df = shuffle(df, random_state=1234)
    print('shuffled trainable images:', df.shape)
    print('saving:', output_file)
    df.to_csv(output_file, index=False)

def generate_full_train_labels():
    _generate_train_label(get_trainable_classes(), os.path.join(settings.DATA_DIR, 'generated_train_labels_7172.csv'))


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

def add_counts_tuning():
    tuning_labels = pd.read_csv(settings.TUNING_LABELS, names=['ImageID', 'LabelName'])
    print(tuning_labels.head())
    tuning_labels['obj_num'] = tuning_labels['LabelName'].map(lambda x: len(x.split()))
    print(tuning_labels.head())

    df_counts = pd.read_csv(settings.SORTED_TUNING_CLASSES).set_index('label_code')
    print(df_counts.head())
    cls_counts = df_counts.to_dict()['counts']
    print(cls_counts)

    tuning_labels['total_counts'] = tuning_labels['LabelName'].map(lambda x: sum([cls_counts[c] for c in x.split()]))
    tuning_labels['avg_counts'] = tuning_labels['total_counts'] // tuning_labels['obj_num']
    tuning_labels['rare_counts'] = tuning_labels['LabelName'].map(lambda x: min([cls_counts[c] for c in x.split()]))
    print(tuning_labels.head(10))

    tuning_labels.to_csv(settings.TUNING_LABELS_COUNTS, index=False, columns=['ImageID', 'obj_num', 'total_counts', 'avg_counts', 'rare_counts', 'LabelName'])

def add_counts_trainable():
    df = pd.read_csv(settings.TRAIN_LABEL_FILE)
    print(df.head())
    df['obj_num'] = df['LabelName'].map(lambda x: len(x.split()))
    print(df.head())

    df_counts = pd.read_csv(settings.SORTED_CLASSES_TRAINABLE).set_index('label_code')
    print(df_counts.head())
    cls_counts = df_counts.to_dict()['counts']
    #print(cls_counts)

    df['total_counts'] = df['LabelName'].map(lambda x: sum([cls_counts[c] for c in x.split()]))
    df['avg_counts'] = df['total_counts'] // df['obj_num']
    df['rare_counts'] = df['LabelName'].map(lambda x: min([cls_counts[c] for c in x.split()]))
    print(df.head(10))

    df.to_csv(settings.TRAIN_LABEL_FILE_COUNTS, index=False, columns=['ImageID', 'obj_num', 'total_counts', 'avg_counts', 'rare_counts', 'LabelName'])



def test_stratify():
    X = [1, 1, 1, 2,2,2,3,3,3,4,4,4]
    Y = ['a'] * 6 + ['b']*6
    print([i for i in zip(X,Y)])

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, stratify=Y)
    print('x_train:', x_train)
    print('y_train:', y_train)
    print('x_val:', x_val)
    print('y_val:', y_val)

def preprocess():
    create_class_counts_df()
    create_tuning_class_counts_df()
    generate_full_train_labels()

if __name__ == '__main__':
    preprocess()
    add_counts_trainable()
    add_counts_tuning()
