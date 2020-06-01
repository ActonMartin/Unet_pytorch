from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch
import numpy as np
import json
import cv2
import glob
import os
import random

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ISBIDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data_dir = data_dir
        self.dataset = ISBIDataSet(self.data_dir, trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ISBIDataSet(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.transform = transform

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = Image.open(image_path).convert('1')
        label = Image.open(label_path).convert('1')
        image = self.transform(image)
        label = self.transform(label)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    data_dir = "../data/own/train/"
    isbi_dataset = ISBIDataSet(data_dir)
    print("数据个数：", len(isbi_dataset))
    train_loader = ISBIDataLoader(data_dir, batch_size=2, shuffle=True, validation_split=0, num_workers=0)
    for image, label in train_loader:
        print(image.shape)


