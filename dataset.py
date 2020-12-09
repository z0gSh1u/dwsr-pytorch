'''
    Dataset preparation for DWSR.
    LR images are genereated on-the-fly.
'''

import numpy as np
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from util import *


class DWSRDataset(Dataset):

    def __init__(self, hr_dir, crop_size, phase):
        super(DWSRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.hr_files = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir)]
        self.phase = phase

        # crop (crop_size, crop_size) patch, and random flip for augmentation
        self.train_public_transform = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=0, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        # during validation phase, keep every batch output the same
        self.val_public_transform = transforms.Compose([
            transforms.CenterCrop(crop_size)
        ])
        # downsample by 4x using BICUBIC
        self.lr_transoform = transforms.Compose([
            transforms.Resize(crop_size // 4, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        # do nothing
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # provide only Y channel for training
        image = Image.open(self.hr_files[idx]).convert('RGB').convert('YCbCr')
        image = np.array(image)  # HWC, 0-255, YCbCr
        image = Image.fromarray(image[:, :, 0])  # HW, 0-255, Y only

        mid_image = self.train_public_transform(
            image) if self.phase == 'train' else self.val_public_transform(image)

        lr_image = self.lr_transoform(mid_image) # CHW, Y-only, 0-1
        hr_image = self.hr_transform(mid_image) # CHW, Y-only, 0-1

        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_files)
