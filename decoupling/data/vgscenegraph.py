import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import numpy as np
import os, h5py, cv2, pdb


class VgSceneGraphDataset(Dataset):
    def __init__(self, h5_path, image_dir, image_size=(256, 256), max_samples=None):
        """
        :param h5_path: h5文件的位置
        :param image_dir: 图像文件的位置
        :param image_size: 图像resize的大小
        :param normalize_images: 是否归一化图像数据集
        """
        super(VgSceneGraphDataset, self).__init__()
        
        self.image_dir = image_dir
        self.image_size = image_size
        self.max_samples = max_samples
        
        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
    
    def __len__(self):
        """
        获取数据集图像数量
        :return: 数据集中图像的数量
        """
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num
    
    def __getitem__(self, index):
        """
        获取单个图像item
        :param index: 要获取的图像的index
        
        :return: 返回LAB的图像，规模为(3, h, w)
        """
        # 如果index大于最大的图像数量，则说明需要翻转图片扩充数据集
        flip = False
        if index >= self.data['object_names'].size(0):
            index = index - self.data['object_names'].size(0)
            flip = True
        
        img_path = os.path.join(self.image_dir, self.image_paths[index].decode('utf-8'))
        
        # 利用cv2读取图像，切换到LAB颜色空间，并转换成Tensor，得到的图像为(c, h, w)
        img = cv2.imread(img_path)
        if flip:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = torch.Tensor(img)
        img = img.permute(2, 0, 1).contiguous()
        img = (img - 127.5) / 127.5
        return img


def get_VgSceneGraph_dataset(dataset_type="train"):
    """
    :param dataset_type: 要获取的数据集 train, val, test
    
    :return: dataset: 数据集
    :return: vocab: 词典
    :return: num_classes: 物体的类别总数
    """
    
    if dataset_type == 'test':
        h5_path = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/transLayout/data/test.h5'
    elif dataset_type == 'val':
        h5_path = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/transLayout/data/val.h5'
    else:
        h5_path = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/transLayout/data/train.h5'
    image_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/transLayout/data/data/images/'
    dataset = VgSceneGraphDataset(h5_path, image_dir, image_size=(256, 256))
    return dataset


if __name__ == '__main__':
    #pdb.set_trace()
    dataset = get_VgSceneGraph_dataset("train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for batch in loader:
        cv2.imshow("L", batch[0][0].numpy())
        cv2.waitKey()
        break
    