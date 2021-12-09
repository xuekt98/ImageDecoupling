import os
import torch

from tqdm import tqdm
from decoupling.decoupling_model.predict_model import Predict_Model
from decoupling.decoupling_model.transformer import sample_with_past
from datetime import datetime
from utils import get_data_loader, save_image, mkdir, set_random_seed

import pdb


@torch.no_grad()
def sample_image(args=None):
    set_random_seed(args['random_seed'])
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sample_path = os.path.join(args['save_path'], now)
    mkdir(sample_path)
    device = torch.device('cuda:0')
    
    # 加载数据集
    data_loader = get_data_loader(args['dataset'], "val", args['batch_size'])
    
    # decoupling model
    predict_model = Predict_Model(**args['predict_model_config']).to(device)
    predict_model.eval()
    
    pbar = range(len(data_loader))
    pbar = tqdm(range(10), initial=0, dynamic_ncols=True, smoothing=0.01)
    count = 0
    for idx in pbar:
        if count > 10:
            break
        for data in data_loader:
            if count > 10:
                break
            
            images = data.to(device)
            qzshape = [args['batch_size'], 256, 16, 16]
            
            save_image(images, f"ori_{count}", os.path.join(sample_path, "origin"))
            
            # 保存重建出来的序列
            logits, target = predict_model(images)
            rec_indexs = predict_model.logits_to_index(logits)
            rec_indexs = rec_indexs.reshape(args['batch_size'], 1, 256)
            images_rec_ab = predict_model.decode_to_ab(rec_indexs, qzshape)
            images_rec = torch.cat((images[:, 0:1, :, :], images_rec_ab), dim=1)
            save_image(images_rec, f'rec_{count}', os.path.join(sample_path, "reconstruct"))
            
            l_embedding = predict_model.embedding_l(images[:, 0:1, :, :])
            batch_size, embd_dim, h, w = l_embedding.shape
            l_embedding = l_embedding.permute(0, 2, 3, 1).contiguous()
            l_embedding = l_embedding.view(batch_size, h * w, embd_dim)
            for t in range(10):
                # 保存采样预测出来的序列
                sample_indexs = sample_with_past(embeddings=l_embedding, model=predict_model.gpt_model, steps=256,
                                                 sample_logits=True,
                                                 temperature=args['temperature'],
                                                 top_k=args['top_k'],
                                                 top_p=args['top_p'])
                # logits, target = predict_model(images)
                # _, lab_indices = predict_model.encode_lab(images)
                
                images_pre_ab = predict_model.decode_to_ab(sample_indexs, qzshape)
                images_pre = torch.cat((images[:, 0:1, :, :], images_pre_ab), dim=1)
                save_image(images_pre, f'pre_{count}_{t}', os.path.join(sample_path, f"predict/{count}"))
            count = count + 1
            pbar.update(1)


args = {
    'random_seed': 1029,  # 随机种子
    'dataset': 'ILSVRC2012',  # 要使用的数据集
    'save_path': './output/sample',  # 采样的图像存储位置
    
    'batch_size': 4,
    'top_k': 600,
    'top_p': 0.92,
    'temperature': 1.0,
    
    'predict_model_config': {
        'ckpt_path': './output/predict/2021-11-28T22-01-08/train/model/PM_3_270000.pth',
        'gpt_config': {
            'vocab_size': 1024,
            'block_size': 512,
            'n_layer': 12,
            'n_head': 8,
            'n_embd': 1024,
            'embd_pdrop': 0.,
            'resid_pdrop': 0.,
            'attn_pdrop': 0.,
            'n_unmasked': 0
        },
        'vqmodel_config': {
            'ckpt_path': './output/reconstruct/2021-11-27T21-15-59/train/model/VQModel_0_120000.pth',
            'embed_dim': 256,
            'n_embed': 1024,
            'ed_config': {
                'z_channels': 256,
                'resolution': 256,
                'in_channels': 3,
                'out_channels': 2,
                'map_channels': 128,
                'ch_mult': [1, 1, 2, 2, 4],
                'num_res_blocks': 2,
                'attn_resolutions': [16],
                'dropout': 0.0
            },
        },
    },
}

sample_image(args)

