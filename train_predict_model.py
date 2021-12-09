import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from estimate.FID import InceptionV3, frechet_distance
from decoupling.decoupling_model.predict_model import Predict_Model
from decoupling.decoupling_model.transformer import sample_with_past
import torch.nn.functional as F
from utils import save_image, set_random_seed, make_dirs, get_data_loader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
import pdb


@torch.no_grad()
def save_reconstruct_image(predict_model, images, qzshape, image_path, subpath_name, count):
    """
    存储重建的图像
    :param predict_model: predict_model
    :param images: 图像batch
    :param qzshape: qzshape
    :param image_path: 图像路径
    :param subpath_name: 子路径
    :param count: count轮次
    :return:
    """
    predict_model.eval()
    logits, _ = predict_model(images)
    indexs = predict_model.logits_to_index(logits)
    images_rec_ab = predict_model.decode_to_ab(indexs, qzshape)
    images_rec = torch.cat((images[:, 0:1, :, :], images_rec_ab), dim=1)
    save_image(images_rec, f'{subpath_name}_{count}', os.path.join(image_path, subpath_name))


@torch.no_grad()
def save_sample_image(predict_model, images, qzshape, image_path, subpath_name, count):
    """
    存储采样而来的图像
    :param qzshape: qzshape
    :param count: 存储的轮次
    :param subpath_name: 子路径名
    :param image_path: 图像存储的路径
    :param predict_model: 模型
    :param images: 图像数据
    :return:
    """
    predict_model.eval()
    l_embedding = predict_model.embedding_l(images[:, 0:1, :, :])
    batch_size, embd_dim, h, w = l_embedding.shape
    l_embedding = l_embedding.permute(0, 2, 3, 1).contiguous()
    l_embedding = l_embedding.view(batch_size, h * w, embd_dim)
    sample = sample_with_past(embeddings=l_embedding, model=predict_model.gpt_model, steps=256,
                              sample_logits=False)
    images_pre_ab = predict_model.decode_to_ab(sample, qzshape)
    images_pre = torch.cat((images[:, 0:1, :, :], images_pre_ab), dim=1)
    save_image(images_pre, f'{subpath_name}_{count}', os.path.join(image_path, subpath_name))


def train_predict_model(args=None):
    set_random_seed(args['random_seed'])
    image_path, model_path, log_path, now = make_dirs(args, "train")
    writer = SummaryWriter(log_dir=log_path)
    device = torch.device('cuda:0')
    
    # 加载数据集
    train_loader = get_data_loader(args['dataset'], "train", args['batch_size'])
    
    # 获取一个batch的固定测试图像数据
    test_image_data = None
    for data in train_loader:
        test_image_data = data.to(device)
        save_image(test_image_data, f'test_origin', os.path.join(image_path, 'test'))
        break
    
    # predict model
    predict_model = Predict_Model(**args['predict_model_config']).to(device)
    
    # initialize optimizer
    loss_rate = args['lr']
    predict_model_params = []
    for key, value in dict(predict_model.named_parameters()).items():
        if value.requires_grad:
            predict_model_params += [{'params': [value], 'lr': loss_rate}]
    predict_model_optimizer = torch.optim.Adam(predict_model_params, betas=(0.5, 0.9))
    # 从现有模型加载优化器
    if args['load_model']:
        predict_model_optimizer.load_state_dict(
            torch.load(args['state_dict_path'], map_location="cpu")["predict_model.optim"])
    
    pbar = range(args['total_epoch'])
    pbar = tqdm(pbar, initial=args['start_epoch'], dynamic_ncols=True, smoothing=0.01)
    count = args['start_count']
    for idx in pbar:
        for i, data in enumerate(train_loader):
            try:
                images = data.to(device)
                
                predict_model.train()
                predict_model.zero_grad()
                
                logits, target = predict_model(images)
                
                # 计算交叉熵损失
                ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                model_loss = ce_loss
                model_loss.backward()
                predict_model_optimizer.step()
                
                # 显示损失函数的值
                writer.add_scalar('loss', model_loss, count)
                pbar.set_description(
                    (
                        f'rand:{count}, loss:{model_loss:.4f}'
                    )
                )
                
                if (count <= 5000 and count % 250 == 0) \
                        or (5000 < count <= 10000 and count % 500 == 0) \
                        or (count > 10000 and count % 1000 == 0):
                    with torch.no_grad():
                        predict_model.eval()
                        qzshape = [args['batch_size'], 256, 16, 16]  # vqmodel decoder的输入维度
                        
                        # 存储原图
                        save_image(images, f"ori_{count}", os.path.join(image_path, "origin"))
                        # 存储重建的图像
                        save_reconstruct_image(predict_model, images, qzshape, image_path, "reconstruct", count)
                        # 存储采样的重建图像
                        save_sample_image(predict_model, images, qzshape, image_path, "sample_rec", count)
                        
                        # 存储test图像
                        save_reconstruct_image(predict_model, test_image_data, qzshape, image_path, "sample_test",
                                               count)
                        # 存储采样预测的test图像
                        save_sample_image(predict_model, test_image_data, qzshape, image_path, "sample_test", count)
                
                if count > 0 and count % 30000 == 0:
                    # 在验证集上计算PSNR
                    predict_model.eval()
                    PSNR = val_test_predict_model(args, predict_model, "val", now, count)
                    writer.add_scalar("VAL_PSNR", PSNR, count)
                    writer.add_scalar('loss', model_loss, count)
                    
                    # 存储模型断点
                    state_dict = {
                        'predict_model': predict_model.state_dict(),
                        'predict_model.optim': predict_model_optimizer.state_dict()
                    }
                    start_epoch = args['start_epoch']
                    torch.save(state_dict, os.path.join(model_path, f'PM_{idx + start_epoch}_{count}.pth'))
                count = count + 1
            except BaseException as e:
                print('Exception save model start!!!')
                state_dict = {
                    'predict_model': predict_model.state_dict(),
                    'predict_model.optim': predict_model_optimizer.state_dict()
                }
                start_epoch = args['start_epoch']
                torch.save(state_dict, os.path.join(model_path, f'PM_{idx + start_epoch}_{count}.pth'))
                print('Exception save model success!!!')
                print(e)


@torch.no_grad()
def val_test_predict_model(args, predict_model=None, stage_type="val", now=None, count=None):
    set_random_seed(args['random_seed'])
    image_path, _, _, _ = make_dirs(args, stage_type, now)
    if count is not None:
        image_path = os.path.join(image_path, str(count))
    device = torch.device('cuda:0')
    
    if predict_model is None:
        predict_model = Predict_Model(**args['predict_model_config']).to(device)
    
    # 加载数据集
    PSNR_SUM = 0
    data_loader = get_data_loader(args['dataset'], stage_type, args['batch_size'])
    
    # FID模型
    inception = InceptionV3().eval().to(device)
    mu, cov = [], []
    actvs1, actvs2, actvs3 = [], [], []
    
    pbar = tqdm(range(len(data_loader)), initial=0, dynamic_ncols=True, smoothing=0.01)
    count = 1
    percent = 0
    for data in data_loader:
        images = data.to(device)
        with torch.no_grad():
            predict_model.eval()
            qzshape = [args['batch_size'], 256, 16, 16]  # vqmodel decoder的输入维度
            
            l_embedding = predict_model.embedding_l(images[:, 0:1, :, :])
            batch_size, embd_dim, h, w = l_embedding.shape
            l_embedding = l_embedding.permute(0, 2, 3, 1).contiguous()
            l_embedding = l_embedding.view(batch_size, h * w, embd_dim)
            sample = sample_with_past(embeddings=l_embedding, model=predict_model.gpt_model, steps=256,
                                      sample_logits=False)
            images_pre_ab = predict_model.decode_to_ab(sample, qzshape)
            images_pre = torch.cat((images[:, 0:1, :, :], images_pre_ab), dim=1)
            
            # 如果是染上色的图像
            for i in range(args['batch_size']):
                image_ori = images[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2LAB)
                image_ori = torch.Tensor(image_ori).permute(2, 0, 1).contiguous()
                image_ori = torch.unsqueeze(image_ori, 0)
                actv1 = inception(image_ori.to(device))
                actvs1.append(actv1)
                
                image_pre_ab = images_pre_ab[i, :, :, :].view(1, 2, 256 * 256)
                pre_std = image_pre_ab.std(dim=2).mean()
                if pre_std > 0.03:
                    percent += 1
                    image_pre = images_pre[i, :, :, :].permute(1, 2, 0).cpu().numpy()
                    image_pre = cv2.cvtColor(image_pre, cv2.COLOR_BGR2LAB)
                    image_pre = torch.Tensor(image_pre).permute(2, 0, 1).contiguous()
                    image_pre = torch.unsqueeze(image_pre, 0)
                    actv2 = inception(image_pre.to(device))
                    actvs2.append(actv2)
                    
                    actv3 = actv1
                    actvs3.append(actv3)
                    
                    # image_pre = torch.unsqueeze(images_pre[i, :, :, :], 0)
                    # image_ori = torch.unsqueeze(images[i, :, :, :], 0)
                    # save_image(image_ori, f"ori_{count}_{i}", os.path.join(image_path, 'origin'))
                    # save_image(image_pre, f'sample_{count}_{i}', os.path.join(image_path, 'sample'))
            
            # 计算PSNR
            PSNR = peak_signal_noise_ratio(images.cpu().numpy(), images_pre.cpu().numpy())
            PSNR_SUM += PSNR
            
            pbar.set_description(
                (
                    f'rand:{count}, PSNR:{PSNR:.4f}, average_PSNR:{PSNR_SUM / count:.4f}'
                )
            )
            pbar.update(1)
            
            # 保存验证集的结果
            # if count % 1 == 0:
            #     save_reconstruct_image(predict_model, images, qzshape, image_path, "reconstruct", count)
            #     save_image(images, f"ori_{count}", os.path.join(image_path, 'origin'))
            #     save_image(images_pre, f'sample_{count}', os.path.join(image_path, 'sample'))
            #     pbar.update(1)
        count += 1
    actvs1 = torch.cat(actvs1, dim=0).cpu().detach().numpy()
    actvs2 = torch.cat(actvs2, dim=0).cpu().detach().numpy()
    actvs3 = torch.cat(actvs3, dim=0).cpu().detach().numpy()
    mu.append(np.mean(actvs1, axis=0))  # 整个验证集
    mu.append(np.mean(actvs2, axis=0))  # 染上色的图像
    mu.append(np.mean(actvs3, axis=0))  # 染上色的图像对应的验证集图像
    cov.append(np.cov(actvs1, rowvar=False))
    cov.append(np.cov(actvs2, rowvar=False))
    cov.append(np.cov(actvs3, rowvar=False))
    val_pre_fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    val_gro_fid_value = frechet_distance(mu[0], cov[0], mu[2], cov[2])
    pre_gro_fid_value = frechet_distance(mu[1], cov[1], mu[2], cov[2])
    print(f'val_pre_fid_value={val_pre_fid_value}')
    print(f'val_gro_fid_value={val_gro_fid_value}')
    print(f'pre_gro_fid_value={pre_gro_fid_value}')
    print(f'percentage={percent / len(data_loader)}')
    return PSNR_SUM / len(data_loader)


args = {
    'random_seed': 1029,  # 随机种子
    'dataset': 'ILSVRC2012',  # 要使用的数据集
    
    # 如果从现有训练模型加载，则使用root_path，否则使用save_path
    'load_model': True,  # 是否加载已有的模型
    'state_dict_path': "./output/predict/2021-11-28T22-01-08/train/model/PM_3_270000.pth",  # state dict path
    'root_path': "./output/predict/2021-11-28T22-01-08",  # 存储中间数据的文件夹位置
    'save_path': './output/predict',  # 实验中间输出存储的位置
    
    # 训练阶段的参数
    'image_size': (256, 256),  # 数据集图像的统一大小，需要与model的resolution保持一致
    'batch_size': 32,  # batch大小
    'lr': 0.0001,  # 损失率
    'start_epoch': 4,  # epoch起始位置
    'total_epoch': 100,  # 总epoch数
    'start_count': 270000,  # count计数开始位置
    
    'predict_model_config': {
        'ckpt_path': "./output/predict/2021-11-28T22-01-08/train/model/PM_3_270000.pth",
        
        # 第二阶段gpt的参数设置
        'gpt_config': {
            'vocab_size': 1024,  # codebook的大小
            'block_size': 512,  # transformer序列最大长度
            'n_layer': 12,  # transformer block的个数
            'n_head': 8,  # 注意力头的个数
            'n_embd': 1024,  # 嵌入维度
            'embd_pdrop': 0.,  # 嵌入层dropout
            'resid_pdrop': 0.,  # 残差连接dropout
            'attn_pdrop': 0.,  # attention层dropout
            'n_unmasked': 0  #
        },
        
        # 第一阶段vqmodel的参数设置
        'vqmodel_config': {
            'ckpt_path': './output/reconstruct/2021-11-27T21-15-59/train/model/VQModel_0_120000.pth',
            # check point path
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
    },
}

val_test_predict_model(args)
# train_predict_model(args)

