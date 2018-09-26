import os
import argparse
import glob
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

import settings


class TrainDataset(data.Dataset):
    def __init__(self, img_dir, img_sz):
        self.file_names = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.num = len(self.file_names)
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_sz),
            transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img = Image.open(self.file_names[index], 'r')
        img = img.convert('RGB')
        img = self.img_transforms(img)
        return img
    def __len__(self):
        return self.num

def calculate_mean(args):
    dset = TrainDataset(settings.TRAIN_IMG_DIR, args.img_sz)
    print(len(dset))
    dloader = data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    img_sum = None
    for i, img in enumerate(dloader):
        if args.n_batches > 0 and i >= args.n_batches:
            break
        img = img.cuda()
        if img_sum is None:
            img_sum = img
        else:
            img_sum += img

        if i % 100 == 0:
            print('.', end='')
    print('')
    print(img_sum.size())
    img_sum = torch.sum(img_sum, 0)
    print(img_sum.size())
    img_sum = torch.sum(torch.sum(img_sum, 1), 1)

    if args.n_batches == 0:
        num_batches = len(dset) // args.batch_size
    elif args.n_batches > 0:
        num_batches = args.n_batches
    else:
        raise ValueError('n_batches error')
    
    mean = img_sum / (num_batches*args.batch_size*args.img_sz*args.img_sz)
    print('mean:', mean)
    return mean
    

def calculate_std(args, rgb_mean):
    rgb_mean = torch.unsqueeze(rgb_mean, -1)
    rgb_mean = torch.unsqueeze(rgb_mean, -1)
    rgb_mean = torch.unsqueeze(rgb_mean, 0)

    dset = TrainDataset(settings.TRAIN_IMG_DIR, args.img_sz)
    print(len(dset))
    dloader = data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    std_sum = None
    for i, img in enumerate(dloader):
        if args.n_batches > 0 and i >= args.n_batches:
            break
        img = img.cuda()
        if i % 100 == 0:
            print('.', end='')
        if std_sum is None:
            std_sum = (img - rgb_mean) * (img - rgb_mean)
        else:
            std_sum += (img - rgb_mean) * (img - rgb_mean)
    print(std_sum.size()) 
    std_sum = torch.sum(std_sum, 0)
    print(std_sum.size()) 
    std_sum = torch.sum(torch.sum(std_sum, 1), 1)

    if args.n_batches == 0:
        num_batches = len(dset) // args.batch_size
    elif args.n_batches > 0:
        num_batches = args.n_batches
    else:
        raise ValueError('n_batches error')

    std_dev = torch.sqrt(std_sum / (num_batches*args.batch_size*args.img_sz*args.img_sz - 1))
    print('std:', std_dev)
    return std_dev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inclusive')
    parser.add_argument('--img_sz', default=256, type=int, help='image size')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--n_batches', default=0, type=int, help='batch number')

    args = parser.parse_args()

    rgb_mean = calculate_mean(args)
    calculate_std(args, rgb_mean)
