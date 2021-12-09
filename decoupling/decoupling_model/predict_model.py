import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from decoupling.decoupling_model.vqmodel import VQModel
from decoupling.decoupling_model.transformer import GPT


class L_Encoder(nn.Module):
    """
        将L层图像转换成1024维的feature，作为Transformer的条件信息
    """
    
    def __init__(self, pretrained):
        super(L_Encoder, self).__init__()
        self.feature = []  # vgg卷积部分
        # self.lin = []  # vgg线性部分
        
        # vgg第一部分 (batch, 1, 256, 256)->(batch, 64, 128, 128)
        self.feature.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # vgg第二部分 (batch, 64, 128, 128)->(batch, 128, 64, 64)
        self.feature.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # vgg第三部分 (batch, 128, 64, 64)->(batch, 256, 32, 32)
        self.feature.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # vgg第四部分 (batch, 256, 32, 32)->(batch, 512, 16, 16)
        self.feature.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # vgg第五部分 (batch, 512, 16, 16)->(batch, 1024, 8, 8)
        self.feature.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1))
        self.feature.append(nn.ReLU())
        self.feature.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # 线性部分
        # self.lin.append(nn.Linear(in_features=512 * 8 * 8, out_features=4096))
        # self.lin.append(nn.ReLU())
        # self.lin.append(nn.Dropout(p=0.5))
        # self.lin.append(nn.Linear(in_features=4096, out_features=2048))
        # self.lin.append(nn.ReLU())
        # self.lin.append(nn.Dropout(p=0.5))
        # self.lin.append(nn.Linear(in_features=2048, out_features=1024))
        
        self.feature = nn.Sequential(*self.feature)
        # self.lin = nn.Sequential(*self.lin)
    
    def forward(self, L):
        feat = self.feature(L)
        # feat = feat.view(L.size(0), -1)
        # out = self.lin(feat)
        return feat


class Predict_Model(nn.Module):
    """
        Conditional Transformer 利用GPT进行预测AB颜色空间
    """
    
    def __init__(self, gpt_config, vqmodel_config, l_encoder_pretrained=False,
                 ckpt_path=None, ignore_keys=[]):
        super().__init__()
        self.vqmodel = VQModel(**vqmodel_config)
        self.gpt_model = GPT(**gpt_config)
        self.l_encoder = L_Encoder(pretrained=l_encoder_pretrained)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["predict_model"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def forward(self, input):
        # pdb.set_trace()
        _, lab_indices = self.encode_lab(input)
        l_embedding = self.embedding_l(input[:, 0:1, :, :])
        batch_size, embd_dim, h, w = l_embedding.shape
        l_embedding = l_embedding.permute(0, 2, 3, 1).contiguous()
        l_embedding = l_embedding.view(batch_size, h*w, embd_dim)

        target = lab_indices
        logits, _ = self.gpt_model(idx=lab_indices, embeddings=l_embedding)
        logits = logits[:, l_embedding.shape[1]-1:]
        return logits, target
    
    @torch.no_grad()
    def encode_lab(self, lab):
        """将lab进行离散化编码"""
        quant_z, _, info = self.vqmodel.encode(lab)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices
        
    def embedding_l(self, l):
        """将L分量转换成Transformer的条件信息embedding"""
        return self.l_encoder(l)
    
    """下面是采集图像样本的部分"""
    @torch.no_grad()
    def decode_to_ab(self, index, zshape):
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])
        quant_z = self.vqmodel.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc
        )
        ab = self.vqmodel.decode(quant_z)
        return ab
    
    @torch.no_grad()
    def logits_to_index(self, logits):
        probs = F.softmax(logits, dim=-1)
        _, index = torch.topk(probs, k=1, dim=-1)

        return index
    
    # def sample(self, lab, steps, temperature=1.0, sample=False, top_k=None):
    #     for k in range(steps):
    #         logits

