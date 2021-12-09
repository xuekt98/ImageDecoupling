import os
import torch
from skimage.metrics import peak_signal_noise_ratio
from tqdm.autonotebook import tqdm
from decoupling.decoupling_model.vqmodel import VQModel
from torch.utils.tensorboard import SummaryWriter
from utils import make_dirs, save_image, set_random_seed, get_data_loader


def train_vqmodel(args=None):
    set_random_seed(args['random_seed'])
    image_path, model_path, log_path, now = make_dirs(args, "train")
    writer = SummaryWriter(log_path)
    device = torch.device('cuda:0')
    
    # 加载数据集
    train_loader = get_data_loader(args['dataset'], "train", args['batch_size'])
    
    # 获取一个batch的固定测试图像数据
    test_image_data = None
    for data in train_loader:
        test_image_data = data.to(device)
        save_image(test_image_data, f'test_origin', os.path.join(image_path, 'test'))
        break
    
    # decoupling model
    vqmodel = VQModel(**args['vqmodel_config']).to(device)
    
    # initialize optimizer
    lr = args['lr']
    vqmodel_params = []
    for key, value in dict(vqmodel.named_parameters()).items():
        if value.requires_grad:
            vqmodel_params += [{'params': [value], 'lr': lr}]
    vqmodel_optimizer = torch.optim.Adam(vqmodel_params, betas=(0.5, 0.9))
    # 从现有模型加载优化器
    if args['load_model']:
        vqmodel_optimizer.load_state_dict(torch.load(args['state_dict_path'], map_location="cpu")['vqmodel_optim'])
    
    pbar = range(args['total_epoch'])
    pbar = tqdm(pbar, initial=args['start_epoch'], dynamic_ncols=True, smoothing=0.01)
    count = args['start_count']
    for idx in pbar:
        for i, data in enumerate(train_loader):
            try:
                images = data.to(device)
                
                vqmodel.train()
                vqmodel.zero_grad()
                
                images_rec_ab, qloss = vqmodel(images)
                images_rec = torch.cat((images[:, 0:1, :, :], images_rec_ab), dim=1)
                images_ab = images[:, 1:3, :, :]
                
                rec_loss = torch.abs(images_ab.contiguous() - images_rec_ab.contiguous())
                model_loss = (rec_loss.mean() + qloss.mean()) * 20.0
                model_loss.backward()
                vqmodel_optimizer.step()
                
                writer.add_scalar('loss', model_loss, count)
                
                pbar.set_description(
                    (
                        f'count: {count} model_loss: {model_loss:.4f}'
                    )
                )
                
                # 存储中间实验结果信息
                if (count <= 5000 and count % 200 == 0) \
                        or (5000 < count <= 10000 and count % 500 == 0) \
                        or (count > 10000 and count % 1000 == 0):
                    # 保存中间的图像结果
                    save_image(images, f"ori_{count}", os.path.join(image_path, 'origin'))
                    save_image(images_rec, f'rec_{count}', os.path.join(image_path, 'reconstruct'))
                    
                    # 获取固定的测试数据的结果
                    with torch.no_grad():
                        vqmodel.eval()
                        images_test_ab, _ = vqmodel(test_image_data)
                        images_test = torch.cat((test_image_data[:, 0:1, :, :], images_test_ab), dim=1)
                        save_image(images_test, f'test_{count}', os.path.join(image_path, 'test'))
                if count > 0 and count % 30000 == 0:
                    # 在验证集上计算PSNR
                    vqmodel.eval()
                    PSNR = val_test_vqmodel(args, vqmodel, "val", now)
                    writer.add_scalar("VAL_PSNR", PSNR, count)
                    writer.add_scalar('loss', model_loss, count)
                    
                    # 存储模型断点
                    state_dict = {
                        'vqmodel': vqmodel.state_dict(),
                        'vqmodel_optim': vqmodel_optimizer.state_dict()
                    }
                    start_epoch = args['start_epoch']
                    torch.save(state_dict, os.path.join(model_path, f'VQModel_{idx + start_epoch}_{count}.pth'))
                count = count + 1
            except BaseException as e:
                print('Exception save model start!!!')
                state_dict = {
                    'vqmodel': vqmodel.state_dict(),
                    'vqmodel_optim': vqmodel_optimizer.state_dict()
                }
                start_epoch = args['start_epoch']
                torch.save(state_dict, os.path.join(model_path, f'VQModel_{idx + start_epoch}_{count}.pth'))
                print('Exception save model success!!!')
                print(e)


@torch.no_grad()
def val_test_vqmodel(args, vqmodel=None, stage_type="val", now=None):
    set_random_seed(args['random_seed'])
    image_path, _, _, _ = make_dirs(args, stage_type, now)
    device = torch.device('cuda:0')
    
    if vqmodel is None:
        vqmodel = VQModel(**args['vqmodel_config']).to(device)
    
    # 加载数据集
    batch_size = args['batch_size']
    image_count = 0
    PSNR_SUM = 0
    MAXI = 255
    data_loader = get_data_loader(args['dataset'], stage_type, args['batch_size'])
    
    pbar = range(len(data_loader))
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
    count = 0
    for data in data_loader:
        count += 1
        images = data.to(device)
        with torch.no_grad():
            vqmodel.eval()
            images_rec_ab, _ = vqmodel(images)
            images_rec = torch.cat((images[:, 0:1, :, :], images_rec_ab), dim=1)
            
            # 计算PSNR
            PSNR = peak_signal_noise_ratio(images.cpu().numpy(), images_rec.cpu().numpy())
            PSNR_SUM += PSNR
            
            pbar.set_description(
                (
                    f'rand:{count}, PSNR:{PSNR:.4f}, average_PSNR:{PSNR_SUM / count:.4f}'
                )
            )
            
            # 保存验证集的结果
            if count > 0 and count % 1000 == 0:
                save_image(images, f"ori_{count}", os.path.join(image_path, 'origin'))
                save_image(images_rec, f'rec_{count}', os.path.join(image_path, 'reconstruct'))
                pbar.update(1000)
    
    return PSNR_SUM / len(data_loader)


args = {
    'random_seed': 1029,  # 随机种子
    'dataset': 'ILSVRC2012',  # 要使用的数据集
    
    # 如果从现有训练模型加载，则使用root_path，否则使用save_path
    'load_model': False,  # 是否加载已有的模型
    'state_dict_path': "",  # state dict path
    'root_path': "",  # 存储中间数据的文件夹位置
    'save_path': './output/reconstruct',  # 实验中间输出存储的位置
    
    # 训练阶段的参数
    'image_size': (256, 256),  # 数据集图像的统一大小，需要与model的resolution保持一致
    'batch_size': 8,  # batch大小
    'lr': 0.0001,  # 损失率
    'start_epoch': 0,  # epoch起始位置
    'total_epoch': 100,  # 总epoch数
    'start_count': 0,  # count计数开始位置
    
    'vqmodel_config': {  # VQModel参数设置
        'ckpt_path': './output/reconstruct/2021-11-27T21-15-59/train/model/VQModel_0_120000.pth',
        'embed_dim': 256,  # codebook中的code的维度
        'n_embed': 1024,  # codebook的大小
        'ed_config': {  # encoder与decoder部分的参数设置
            'z_channels': 256,  # 量化的长度
            'resolution': 256,  # 输入图像的像素
            'in_channels': 3,  # encoder输入维度
            'out_channels': 2,  # decoder输出维度
            'map_channels': 128,  # 从初始维度映射到的中间维度
            'ch_mult': [1, 1, 2, 2, 4],  # 维度改变
            'num_res_blocks': 2,  # ResNet块的个数
            'attn_resolutions': [16],  # encoder与decoder中需要用到注意力的像素规模
            'dropout': 0.0  # dropout
        },
    },
}

val_test_vqmodel(args)
# train_vqmodel(args)
