import os
import random
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from datetime import datetime
import pdb

from decoupling.data.ILSVRC2012 import get_ILSVRC2012_dataset
from decoupling.data.vgscenegraph import get_VgSceneGraph_dataset


def mkdir(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


def make_dirs(args, stage_type="train", now=None):
    if now is None:
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if args['load_model']:
        # 如果加载已有的模型，则继续保存在指定的文件夹下
        log_path = os.path.join(args['root_path'], "log")
        root_path = os.path.join(args['root_path'], stage_type)
    else:
        log_path = os.path.join(args['save_path'], now, "log")
        root_path = os.path.join(args['save_path'], now, stage_type)
    mkdir(log_path)
    
    # 只有训练阶段创建model文件夹
    model_path = None
    if stage_type == "train":
        model_path = os.path.join(root_path, "model")
        mkdir(model_path)
    image_path = os.path.join(root_path, "images")
    mkdir(image_path)
    return image_path, model_path, log_path, now


def get_data_loader(dataset_type, stage_type="train", batch_size=8):
    """
    加载数据集
    :param dataset_type: 加载的数据集类型
    :param stage_type: train, val, test
    :param batch_size: batch size
    :return:
    """
    if dataset_type == 'ILSVRC2012':
        train_data = get_ILSVRC2012_dataset(stage_type)
    else:
        train_data = get_VgSceneGraph_dataset(stage_type)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    return train_loader


def save_image(img, im_name, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image = make_grid(img.cpu().data, nrow=4).permute(1, 2, 0).contiguous().numpy()
    image = np.asarray((image * 127.5 + 127.5).astype(np.uint8))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(os.path.join(image_path, f'{im_name}.png'), image)


def set_random_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deter
