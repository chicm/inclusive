import os

IMG_SZ = 224

DATA_DIR = r'G:\inclusive'

MODEL_DIR = os.path.join(DATA_DIR, 'models')

TRAIN_IMG_DIR = r'G:\detect\train\224'
TEST_IMG_DIR = os.path.join(DATA_DIR, 'stage_1_test_images')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'stage_1_test_images')

CLASSES_FILE = os.path.join(DATA_DIR, 'classes-trainable.csv')
HUMAN_TRAIN_LABEL_FILE = os.path.join(DATA_DIR, 'train_human_labels.csv')
TRAIN_LABEL_FILE = os.path.join(DATA_DIR, 'generated_train_labels.csv')

VAL_LABEL_FILE = os.path.join(DATA_DIR, 'tuning_labels.csv')
STAGE1_SAMPLE_SUB = os.path.join(DATA_DIR, 'stage_1_sample_submission.csv')