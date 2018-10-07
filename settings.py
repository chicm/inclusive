import os
from local_settings import DATA_DIR, TRAIN_IMG_DIR

IMG_SZ = 224
N_CLASSES = 7178

MODEL_DIR = os.path.join(DATA_DIR, 'models')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'stage_1_test_images')

CLASS_DESCRIPTIONS = os.path.join(DATA_DIR, 'class-descriptions.csv')
CLASSES_TRAINABLE = os.path.join(DATA_DIR, 'classes-trainable.csv')
TRAIN_HUMAN_LABELS = os.path.join(DATA_DIR, 'train_human_labels.csv')
TRAIN_MACHINE_LABELS = os.path.join(DATA_DIR, 'train_machine_labels.csv')
TRAIN_BOUNDING_BOXES = os.path.join(DATA_DIR, 'train_bounding_boxes.csv')
TUNING_LABELS = os.path.join(DATA_DIR, 'tuning_labels.csv')
STAGE_1_SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'stage_1_sample_submission.csv')

CLASSES_FILE = os.path.join(DATA_DIR, 'classes-trainable.csv')
TRAIN_LABEL_FILE = os.path.join(DATA_DIR, 'generated_train_labels_7178.csv')
VAL2_LABEL_FILE = os.path.join(DATA_DIR, 'tuning_labels.csv')

