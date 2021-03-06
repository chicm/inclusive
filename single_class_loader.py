import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from attrdict import AttrDict
from torchvision import datasets, models, transforms
from utils import get_classes, get_test_ids, get_train_val_meta, get_tuning_meta
from balanced_sampler import BalancedSammpler
from weighted_sampler import get_weighted_sample
from PIL import Image
from sklearn.utils import shuffle
import random
import settings

IMG_SZ = settings.IMG_SZ

train_transforms = transforms.Compose([
            transforms.Resize((320,320)),
            transforms.RandomResizedCrop(IMG_SZ, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) imagenet mean and std
            transforms.Normalize([0.4557, 0.4310, 0.3968], [0.2833, 0.2771, 0.2890]) # open images mean and std
        ])
def get_tta_transform(index=0):
    if index == 0:
        return transforms.Compose([
            transforms.Resize((IMG_SZ,IMG_SZ)),
            transforms.ToTensor(),
            transforms.Normalize([0.4557, 0.4310, 0.3968], [0.2833, 0.2771, 0.2890])
        ])
    elif index == 1:
        return transforms.Compose([
            transforms.Resize((IMG_SZ,IMG_SZ)),
            transforms.RandomHorizontalFlip(p=2.),
            transforms.ToTensor(),
            transforms.Normalize([0.4557, 0.4310, 0.3968], [0.2833, 0.2771, 0.2890])
        ])
    else:
        return train_transforms

class ImageDataset(data.Dataset):
    def __init__(self, train_mode, img_ids, img_dir, classes, stoi, val_index=None, label_names=None, tta_index=0):
        self.input_size = settings.IMG_SZ
        self.train_mode = train_mode
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num = len(img_ids)
        self.classes = classes 
        self.stoi = stoi
        self.label_names = label_names
        if val_index is not None:
            #self.df_class_counts = df_class_counts.set_index('label_code')
            self.val_index = val_index
        self.tta_index = tta_index

    def __getitem__(self, index):
        fn = os.path.join(self.img_dir, '{}.jpg'.format(self.img_ids[index]))
        #img = cv2.imread(fn)
        img = Image.open(fn, 'r')
        img = img.convert('RGB')
        
        if self.train_mode:
            img = train_transforms(img)
        else:
            tta_transform = get_tta_transform(self.tta_index)
            img = tta_transform(img)

        if self.label_names is None:
            return img
        else:
            labels = self.get_label(index)
            return img, labels[0], labels[1]

    def __len__(self):
        return self.num
    '''
    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        for i in range(num_imgs):
            inputs[i] = imgs[i]

        if self.label_names is None:
            return inputs
        else:
            labels = [x[1] for x in batch]
            return inputs, torch.tensor(labels)
    '''

    def get_label(self, index):
        label_codes = [x for x in self.label_names[index].strip().split() if x in self.classes]
        #label_counts = [self.df_class_counts.loc[x]['counts'] for x in label_codes]
        if self.train_mode:
            #random.seed(hash(self.img_ids[index]))
            return self.stoi[label_codes[int(random.random()*len(label_codes))]], len(label_codes) # random select for train
        else:
            return self.stoi[label_codes[self.val_index % len(label_codes)]], len(label_codes)

def get_train_loader(args, batch_size=32, dev_mode=False, train_shuffle=True):
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)
    train_meta, _ = get_train_val_meta(args.cls_type, args.start_index, args.end_index)

    # filter, keep label counts <= args.max_labels
    train_meta = train_meta[train_meta['obj_num'] <= args.max_labels]
    
    print(train_meta.shape)
    if len(classes) < 7172:
        classes_set = set(classes)
        train_meta['tmp_label_count'] = train_meta['LabelName'].map(lambda x: len(set(x.split()) & classes_set))
        train_meta = train_meta[train_meta['tmp_label_count'] > 0]
        #tuning_labels['LabelName'].map(lambda x: sum([cls_counts[c] for c in x.split()]))
        #print('>>', train_meta.shape)

    # resample training data
    train_img_ids = get_weighted_sample(train_meta, 1024*100)
    df_sampled = train_meta.set_index('ImageID').loc[train_img_ids]

    if dev_mode:
        train_meta = train_meta.iloc[:10]
        train_shuffle = False
    img_dir = settings.TRAIN_IMG_DIR
    
    train_set = ImageDataset(True, train_img_ids, img_dir, classes, stoi, None, df_sampled['LabelName'].values.tolist())
    
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=4, drop_last=True)#, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = train_set.num

    return train_loader

def get_val_loader(args, val_index, batch_size=32, dev_mode=False, val_num=3000):
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)
    _, val_meta = get_train_val_meta(args.cls_type, args.start_index, args.end_index)

    # filter, keep label counts <= args.max_labels
    val_meta = val_meta[val_meta['obj_num'] <= args.max_labels]

    if len(classes) < 7172:
        classes_set = set(classes)
        val_meta['tmp_label_count'] = val_meta['LabelName'].map(lambda x: len(set(x.split()) & classes_set))
        val_meta = val_meta[val_meta['tmp_label_count'] > 0]

    #print(val_meta.shape)
    #print(val_meta['LabelName'].str.split().apply(pd.Series).stack().nunique())
    val_meta = shuffle(val_meta, random_state=1234).iloc[:val_num]

    #print(val_meta.shape)

    if dev_mode:
        val_meta = val_meta.iloc[:10]
    img_dir = settings.TRAIN_IMG_DIR
    
    val_set = ImageDataset(False, val_meta['ImageID'].values.tolist(), img_dir, classes, stoi, val_index, val_meta['LabelName'].values.tolist())

    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    val_loader.num = val_set.num

    return val_loader

def get_test_loader(args, batch_size=8, dev_mode=False):
    img_ids = get_test_ids()
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)

    img_dir = settings.TEST_IMG_DIR
    if dev_mode:
        img_ids = img_ids[:10]
        batch_size = 4
    
    dset = ImageDataset(False, img_ids, img_dir, classes, stoi)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    dloader.num = dset.num
    dloader.img_ids = img_ids
    return dloader

def test_train_loader():
    args = AttrDict({'cls_type': 'trainable', 'start_index': 0, 'end_index': 1000, 'max_labels': 5})
    loader = get_train_loader(args, dev_mode=True, batch_size=10)
    for i, data in enumerate(loader):
        imgs, targets, num_targets = data
        print(targets, type(targets), num_targets)
        print(imgs.size(), targets.size(), num_targets.size())
        if i > 0:
            break

def test_val_loader():
    args = AttrDict({'cls_type': 'trainable', 'start_index': 0, 'end_index': 7172, 'max_labels': 3})
    loader = get_val_loader(args, 0, dev_mode=True, batch_size=10)
    for i, data in enumerate(loader):
        imgs, targets, num_targets = data
        print(targets, type(targets), num_targets)
        print(imgs.size(), targets.size(), num_targets.size())
        if i > 0:
            break


def test_test_loader():
    args = AttrDict({'cls_type': 'trainable', 'start_index': 0, 'end_index': 7172, 'max_labels': 3})
    loader = get_test_loader(args, dev_mode=True, batch_size=10)
    for i, data in enumerate(loader):
        imgs = data
        print(imgs.size())
        if i > 0:
            break


if __name__ == '__main__':
    #test_test_loader()
    test_train_loader()
    #test_val_loader()
