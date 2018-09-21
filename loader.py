import os, cv2, glob
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from utils import get_class_stoi, get_train_meta, get_val_meta, get_classes, get_test_ids
from PIL import Image
import settings

class ImageDataset(data.Dataset):
    def __init__(self, img_ids, img_dir, label_names=None):
        self.input_size = settings.IMG_SZ
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num = len(img_ids)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.label_names = label_names
        self.classes = get_classes(settings.CLASSES_FILE)
        self.stoi = get_class_stoi(self.classes)

    def __getitem__(self, index):
        fn = os.path.join(self.img_dir, '{}.jpg'.format(self.img_ids[index]))
        #img = cv2.imread(fn)
        img = Image.open(fn, 'r')
        img = img.convert('RGB')
        img = self.transform(img)

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
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            if self.label_names is not None:
                target = self.get_label_tensor(labels[i])
                targets.append(target)

        if self.label_names is not None:
            return inputs, torch.stack(targets)
        else:
            return inputs
    def get_label_tensor(self, label_names):
        label_idx = set([self.stoi[x] for x in label_names.strip().split()])
        target = [ (1 if i in label_idx else 0) for i in range(len(self.classes))]
        return torch.FloatTensor(target)

def get_train_loader(img_dir=settings.TRAIN_IMG_DIR, batch_size=8, dev_mode=False, shuffle=True):
    meta = get_train_meta()
    if dev_mode:
        meta = meta.iloc[:10]

    img_ids = meta['ImageID'].values.tolist()
    labels = meta['LabelName'].values.tolist()
    
    dset = ImageDataset(img_ids, img_dir, labels)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn, drop_last=True)
    dloader.num = dset.num
    return dloader

def get_val_loader(img_dir=settings.VAL_IMG_DIR, batch_size=8, dev_mode=False, shuffle=False):
    meta = get_val_meta()
    if dev_mode:
        meta = meta.iloc[:10]

    img_ids = meta['ImageID'].values.tolist()
    labels = meta['LabelName'].values.tolist()
    
    dset = ImageDataset(img_ids, img_dir, labels)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=dset.collate_fn)
    dloader.num = dset.num
    return dloader

def get_test_loader(img_dir=settings.TEST_IMG_DIR, batch_size=8):
    img_ids = get_test_ids()
    
    dset = ImageDataset(img_ids, img_dir)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dset.collate_fn, drop_last=False)
    dloader.num = dset.num
    return dloader

def test_train_loader():
    loader = get_train_loader(dev_mode=True)
    for i, data in enumerate(loader):
        imgs, targets = data
        print(targets)
        print(imgs.size(), targets.size())

def test_test_loader():
    loader = get_test_loader()
    for i, data in enumerate(loader):
        imgs = data
        print(imgs.size())
        if i > 10:
            print(imgs)
            break

if __name__ == '__main__':
    test_test_loader()
    #test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])
