import pdb

import torch
from torch.utils.data import Dataset

import cv2


class ImagePaths(Dataset):
    def __init__(self, image_paths, image_size=(256, 256)):
        """
        :param image_paths: 图像文件位置的list
        :param image_size: 图像统一的大小
        """
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        """
        获取单个图像item
        :param index: 图像index
        :return: 返回LAB图像，规模为(3, h, w)
        """
        # 如果index大于最大的图像数量，则说明需要翻转图片扩充数据集
        flip = False
        if index >= self._length:
            index = index - self._length
            flip = True
        
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        while img is None:
            # print(f'{img_path}, {index}')
            img = cv2.imread(img_path)
        if flip:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1).contiguous()
        img = (img - 127.5) / 127.5
        return img
    