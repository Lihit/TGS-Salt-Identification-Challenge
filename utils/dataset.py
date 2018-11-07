# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import random
from .transform import *

def add_depth_channels(image_tensor):
    _, h, w = image_tensor.shape
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor

class TrainDataSet(data.Dataset):
    def __init__(self, train_ids, opt):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.opt = opt
        self.imgs = train_ids
        self.train_transforms = T.Compose([
            T.Lambda(lambda x:x.permute(2,0,1)),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = T.Compose([
            T.Resize(self.opt.image_h),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_name = self.imgs[index]
        data = train_df.loc[img_name, 'images']
        mask = train_df.loc[img_name, 'masks']
        label = train_df.loc[img_name, 'coverage_class']
        label_empty = 1 if label else 0
        #data, mask = np.asarray(data), np.asarray(mask)
        data, mask = self.train_augment(data, mask)
        data = add_depth_channels(np.stack([data]*3,0))
        data, mask = torch.from_numpy(data), torch.from_numpy(mask)
        mask = torch.unsqueeze(mask,0)
        #data = self.train_transforms(data)
        return data, mask, label, label_empty

    def __len__(self):
        return len(self.imgs)

    def train_augment(self, image, mask):
        image = image.copy()
        mask = mask.copy()
    
        if np.random.rand() < 0.5:
            image, mask = do_horizontal_flip2(image, mask)
#         if np.random.rand() < 0.5:
#             image, mask = do_vertical_flip2(image, mask)
        if np.random.rand() < 0.5:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.2)
#         if np.random.rand() < 0.3:
#              c = np.random.choice(2)
#              if c==0:
#                   image, mask = do_horizontal_shear2(image, mask)
#              else:
#                   image, mask = do_elastic_transform2(image, mask)
        if np.random.rand() < 0.5:
            c = np.random.choice(3)
            if c == 0:
                image, mask = do_horizontal_shear2(
                    image, mask, dx=np.random.uniform(-0.07, 0.07))
                pass
    
            if c == 1:
                image, mask = do_shift_scale_rotate2(
                    image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))  # 10
    
            if c == 2:
                image, mask = do_elastic_transform2(
                    image, mask, grid=10, distort=np.random.uniform(0, 0.15))  # 0.10
    
        if np.random.rand() < 0.5:
            c = np.random.choice(3)
            if c == 0:
                image = do_brightness_shift(image, np.random.uniform(-0.1, +0.1))
            if c == 1:
                image = do_brightness_multiply(
                    image, np.random.uniform(1-0.08, 1+0.08))
            if c == 2:
                image = do_gamma(image, np.random.uniform(1-0.08, 1+0.08))
    
        #image, mask = do_resize2(image, mask, self.opt.SIZE, self.opt.SIZE)
        #image, mask = do_center_pad2(image, mask, self.opt.PAD)
        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
        return image, mask


class ValDataSet(data.Dataset):
    def __init__(self, val_ids, opt):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.opt = opt
        self.imgs = val_ids
        self.train_transforms = T.Compose([
            T.Lambda(lambda x:x.permute(2,0,1)),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = T.Compose([
            T.Resize(self.opt.image_h),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_name = self.imgs[index]
        data = train_df.loc[img_name, 'images']
        mask = train_df.loc[img_name, 'masks']
        label = train_df.loc[img_name, 'coverage_class']
        label_empty = 1 if label else 0
        #data, mask = np.asarray(data), np.asarray(mask)
        data, mask = self.val_augment(data, mask)
        data = add_depth_channels(np.stack([data]*3,0))
        data, mask = torch.from_numpy(data), torch.from_numpy(mask)
        #data = self.train_transforms(data)
        mask = torch.unsqueeze(mask,0)
        return data, mask, label, label_empty

    def __len__(self):
        return len(self.imgs)

    def val_augment(self, image, mask):
        image = image.copy()
        mask = mask.copy()
        #image, mask = do_resize2(image, mask, self.opt.SIZE, self.opt.SIZE)
        #image, mask = do_center_pad2(image, mask, self.opt.PAD)
        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
        return image,mask

class TestDataSet(data.Dataset):
    def __init__(self, test_ids, opt):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.opt = opt
        self.imgs = test_ids
        self.train_transforms = T.Compose([
            T.Lambda(lambda x:x.permute(2,0,1)),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_name = self.imgs[index]
        data = test_df.loc[img_name, 'images']
        data = self.test_augment(data)
        data = add_depth_channels(np.stack([data]*3,0))
        #data = np.asarray(data)
        data = torch.from_numpy(data)
        #data = self.train_transforms(data)
        return data, img_name

    def __len__(self):
        return len(self.imgs)

    def test_augment(self, image):
        #image = do_resize(image, self.opt.SIZE, self.opt.SIZE)
        #image = do_center_pad(image, self.opt.PAD)
        image = do_center_pad_to_factor(image, factor=32)
        if self.opt.use_tta:
            image = do_horizontal_flip(image)
        return image