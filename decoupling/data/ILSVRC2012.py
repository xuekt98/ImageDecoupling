from torch.utils.data import Dataset, DataLoader
from decoupling.data.base import ImagePaths
from skimage import io
import os
import shutil
import pdb
from tqdm import tqdm
from PIL import Image


def mkdir(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


def check_single_channel_images(type="train"):
    print(f'check {type} images start')
    root_dir = None
    if type == 'train':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_train'
    elif type == 'val':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_val_'
    elif type == 'test':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_test'
    black_white = os.path.join('/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/black_white', type)

    dir_list = os.listdir(root_dir)
    for i in tqdm(range(0, len(dir_list))):
        path = os.path.join(root_dir, dir_list[i])
        if os.path.isfile(path):
            im = Image.open(path)
            if len(im.split()) == 1:
                mkdir(black_white)
                new_path = os.path.join(black_white, dir_list[i])
                # print(f'move {path} to {new_path}')
                shutil.move(path, new_path)
        elif os.path.isdir(path):
            image_list = os.listdir(path)
            for j in range(0, len(image_list)):
                image_path = os.path.join(path, image_list[j])
                im = Image.open(image_path)
                if len(im.split()) == 1:
                    mkdir(os.path.join(black_white, dir_list[i]))
                    new_path = os.path.join(black_white, dir_list[i], image_list[j])
                    # print(f'move {image_path} to {new_path}')
                    shutil.move(image_path, new_path)
    print(f'check {type} images complete')


def check_jpeg_images(type="train"):
    print(f'check {type} images start')
    root_dir=None
    if type == 'train':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_train'
    elif type == 'val':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_val_'
    elif type == 'test':
        root_dir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_test'
    broken_dir = os.path.join('/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/broken_image', type)
    
    dir_list = os.listdir(root_dir)
    for i in tqdm(range(0, len(dir_list))):
        path = os.path.join(root_dir, dir_list[i])
        # pdb.set_trace()
        if os.path.isfile(path):
            try:
                io.imread(path)
            except Exception as e:
                mkdir(broken_dir)
                new_path = os.path.join(broken_dir, dir_list[i])
                print(f'move {path} to {new_path}')
                shutil.move(path, new_path)
        elif os.path.isdir(path):
            print(f'{path} start')
            image_list = os.listdir(path)
            for j in range(0, len(image_list)):
                image_path = os.path.join(path, image_list[j])
                try:
                    io.imread(image_path)
                except Exception as e:
                    # pdb.set_trace()
                    mkdir(os.path.join(broken_dir, dir_list[i]))
                    new_path = os.path.join(broken_dir, dir_list[i], image_list[j])
                    print(f'move {image_path} to {new_path}')
                    shutil.move(image_path, new_path)
            print(f'{path} complete')
    print(f'check {type} images complete')


def get_train_image_paths():
    """
    这个是脚本函数，用来读取ILSVRC2012的训练集里边所有图像的名字，没有其他作用
    :return:
    """
    with open('../../dataset/ILSVRC2012/train.txt', 'w') as f:
        rootdir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_train'
        list = os.listdir(rootdir)
        print(len(list))
        image_paths = []
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                image_list = os.listdir(path)
                for j in range(0, len(image_list)):
                    image_path = os.path.join(path, image_list[j])
                    f.write(image_path)
                    f.write('\n')
                    image_paths.append(image_path)
    return image_paths


def get_val_test_image_paths(type="val"):
    """
        这个是脚本函数，用来读取ILSVRC2012的验证集和测试集里边所有图像的名字，没有其他作用
        :return:
    """
    image_paths = None
    if type == "test":
        with open('../../dataset/ILSVRC2012/test.txt', 'w') as f:
            rootdir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_test'
            list = os.listdir(rootdir)
            print(len(list))
            image_paths = []
            for i in range(0, len(list)):
                path = os.path.join(rootdir, list[i])
                if os.path.isfile(path):
                    f.write(path)
                    f.write('\n')
                    image_paths.append(path)
    elif type == "val":
        with open('../../dataset/ILSVRC2012/val.txt', 'w') as f:
            rootdir = '/mnt/3af6c1ab-a773-4f52-9de5-8a24b46de932/dataset/ILSVRC2012/ILSVRC2012_img_val'
            list = os.listdir(rootdir)
            print(len(list))
            image_paths = []
            for i in range(0, len(list)):
                path = os.path.join(rootdir, list[i])
                if os.path.isfile(path):
                    f.write(path)
                    f.write('\n')
                    image_paths.append(path)
    return image_paths


class ILSVRC2012Dataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256)):
        super().__init__()
        self.data = ImagePaths(image_paths=image_paths, image_size=image_size)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        img = self.data[i]
        return img


def get_ILSVRC2012_dataset(dataset_type='train', image_size=(256, 256)):
    if dataset_type == 'test':
        root = './dataset/ILSVRC2012/test.txt'
    elif dataset_type == 'val':
        root = './dataset/ILSVRC2012/val.txt'
    else:
        root = './dataset/ILSVRC2012/train.txt'
    
    with open(root, "r") as f:
        image_paths = f.read().splitlines()
        return ILSVRC2012Dataset(image_paths, image_size)
        
        
if __name__ == '__main__':
    check_single_channel_images("train")
    check_single_channel_images("val")
    check_single_channel_images("test")
    # check_jpeg_images("train")
    # check_jpeg_images("val")
    # check_jpeg_images("test")
    # get_train_image_paths()
    # get_val_test_image_paths("val")
    # get_val_test_image_paths("test")
    
    # root = '../../dataset/ILSVRC2012/train.txt'
    # from utils import set_random_seed
    # from tqdm import tqdm
    # set_random_seed(1234)
    # with open(root, "r") as f:
    #     image_paths = f.read().splitlines()
    #     dataset = ILSVRC2012Dataset(image_paths, image_size=(256, 256))
    # train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True)
    # d = dataset[996679] #996679
    # for d in tqdm(train_loader):
    #     continue
