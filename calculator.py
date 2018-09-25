import os
import glob
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import time

import settings

img_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
    ])

class TrainDataset(data.Dataset):
    def __init__(self, img_dir):
        self.file_names = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.num = len(self.file_names)

    def __getitem__(self, index):
        img = Image.open(self.file_names[index], 'r')
        img = img.convert('RGB')
        img = img_transforms(img)
        return img
    def __len__(self):
        return self.num

def calculate_mean():
    dset = TrainDataset(settings.TRAIN_IMG_DIR)
    print(len(dset))
    dloader = data.DataLoader(dset, batch_size=32, shuffle=False, num_workers=4, drop_last=True)
    bg = time.time()
    img_sum = None
    for i, img in enumerate(dloader):
        img = img.cuda()
        if img_sum is None:
            img_sum = img
        else:
            img_sum += img

        if i % 100 == 0:
            print(i, time.time() - bg)
        if i == 999:
            break
        #print(img.size())
        #break
    print(img_sum.size())
    img_sum = torch.sum(img_sum, 0)
    print(img_sum.size())
    img_sum = torch.sum(torch.sum(img_sum, 1), 1)
    print(img_sum / (1000*32*224*224))
    


if __name__ == '__main__':
    calculate_mean()
