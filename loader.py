import os, glob
import numpy as np
import torch
import torch.utils.data as data
from attrdict import AttrDict
from torchvision import datasets, models, transforms
from utils import get_classes, get_test_ids, get_train_val_meta, get_tuning_meta
from balanced_sampler import BalancedSammpler
from weighted_sampler import get_weighted_sample
from PIL import Image
import settings

IMG_SZ = settings.IMG_SZ

train_transforms = transforms.Compose([
            transforms.Resize((320,320)),
            transforms.RandomResizedCrop(IMG_SZ, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
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
    def __init__(self, train_mode, img_ids, img_dir, classes, stoi, label_names=None, tta_index=0):
        self.input_size = settings.IMG_SZ
        self.train_mode = train_mode
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num = len(img_ids)
        self.label_names = label_names
        self.classes = classes 
        self.stoi = stoi
        self.tta_index = tta_index

    def __getitem__(self, index):
        fn = os.path.join(self.img_dir, '{}.jpg'.format(self.img_ids[index]))
        img = Image.open(fn, 'r')
        img = img.convert('RGB')
        
        if self.train_mode:
            img = train_transforms(img)
        else:
            tta_transform = get_tta_transform(self.tta_index)
            img = tta_transform(img)

        if self.label_names is not None:
            return img, self.label_names[index]
        else:
            return [img]

    def __len__(self):
        return self.num

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]

        if self.label_names is not None:
            labels = [x[1] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        targets = []
        obj_nums = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            if self.label_names is not None:
                target, obj_num = self.get_label_tensor(labels[i])
                targets.append(target)
                obj_nums.append(obj_num)

        if self.label_names is not None:
            return inputs, torch.stack(targets), torch.Tensor(obj_nums)
        else:
            return inputs
    def get_label_tensor(self, label_names):
        label_idx = set([self.stoi[x] for x in label_names.strip().split() if x in self.classes])
        target = [ (1 if i in label_idx else 0) for i in range(len(self.classes))]
        return torch.FloatTensor(target), len(label_idx)

def get_train_val_loaders(args, batch_size=32, dev_mode=False, train_shuffle=True, val_num=4000):
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)
    train_meta, val_meta = get_train_val_meta(args.cls_type, args.start_index, args.end_index)

    #sampler = BalancedSammpler(train_meta, classes, stoi, balanced=args.balanced, min_label_num=500, max_label_num=700)
    #df1 = train_meta.set_index('ImageID')
    #sampled_train_meta = df1.loc[sampler.img_ids]

    train_meta = train_meta[train_meta['obj_num'] <= 10]
    val_meta = val_meta[val_meta['obj_num'] <= 10]

    # resample training data
    train_img_ids = get_weighted_sample(train_meta, 1024*100)
    df_sampled = train_meta.set_index('ImageID').loc[train_img_ids]

    #print(df_sampled.shape)
    if val_num is not None:
        val_meta = val_meta.iloc[:val_num]

    #if dev_mode:
    #    train_meta = train_meta.iloc[:10]
    #    val_meta = val_meta.iloc[:10]
    img_dir = settings.TRAIN_IMG_DIR
    #train_set = ImageDataset(True, sampled_train_meta.index.values.tolist(), img_dir, classes, stoi, sampled_train_meta['LabelName'].values.tolist())
    train_set = ImageDataset(True, train_img_ids, img_dir, classes, stoi, df_sampled['LabelName'].values.tolist())
    
    val_set = ImageDataset(False, val_meta['ImageID'].values.tolist(), img_dir, classes, stoi, val_meta['LabelName'].values.tolist())

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=4, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = train_set.num

    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn, drop_last=False)
    val_loader.num = val_set.num

    return train_loader, val_loader

def get_tuning_loader(args, batch_size=32, dev_mode=False, shuffle=False):
    '''
    Stage 1 tuning validating loader
    '''
    img_dir = settings.TEST_IMG_DIR
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)
    meta = get_tuning_meta(args.cls_type, args.start_index, args.end_index)
    if dev_mode:
        meta = meta.iloc[:10]

    img_ids = meta['ImageID'].values.tolist()
    labels = meta['LabelName'].values.tolist()
    print(len(img_ids))

    dset = ImageDataset(False, img_ids, img_dir, classes, stoi, labels)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn, drop_last=False)
    dloader.num = dset.num
    return dloader

def get_test_loader(args, batch_size=8, dev_mode=False, tta_index=0):
    img_ids = get_test_ids()
    classes, stoi = get_classes(args.cls_type, args.start_index, args.end_index)

    img_dir = settings.TEST_IMG_DIR
    if dev_mode:
        img_ids = img_ids[:10]
    
    dset = ImageDataset(False, img_ids, img_dir, classes, stoi, tta_index=tta_index)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dset.collate_fn, drop_last=False)
    dloader.num = dset.num
    return dloader

def test_train_loader():
    args = AttrDict({'cls_type': 'trainable', 'start_index': 0, 'end_index': 50})
    loader, _ = get_train_val_loaders(args, dev_mode=False, batch_size=10)
    for i, data in enumerate(loader):
        imgs, targets, obj_num = data
        print(targets, obj_num)
        print(imgs.size(), targets.size())
        if i > 0:
            break

def test_test_loader():
    args = AttrDict({'cls_type': 'trainable', 'start_index': 0, 'end_index': 50})
    loader = get_test_loader(args, dev_mode=True)
    for i, data in enumerate(loader):
        imgs = data
        print(imgs.size())
        if i > 10:
            print(imgs)
            break

def test_tuning_loader():
    args = AttrDict({'cls_type': 'tuning', 'start_index': 0, 'end_index': 100})
    loader = get_tuning_loader(args, dev_mode=True)
    for i, data in enumerate(loader):
        imgs, targets = data
        print(targets)
        print(imgs.size(), targets.size())


if __name__ == '__main__':
    test_test_loader()
    #test_tuning_loader()
    #test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])
