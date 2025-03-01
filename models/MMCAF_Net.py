

import torch
import torch.nn as nn

from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F
import numpy as np
import math
from sklearn.cluster import KMeans

#kan
from .kan.KANLayer import *

#ISBI_utils
from .model_utils.multiscale_fusion import MultiScale_Fusion

#img_encoder
from .img_encoder1 import Img_new

class MMCAF_Net(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        #image_encoder
        self.image_encoder = Img_new()
        # 加载预训练参数
        # 去掉 state_dict 中的 "module." 前缀
        new_state_dict = {}
        state_dict = torch.load("/home/vesselseg3/yujianxun/train_result/penet_multiscale_drop/ckpt/best30.pth.tar")['model_state']  # 加载权重
        for key, value in state_dict.items():
            # 去掉 "module." 前缀
            if key.startswith("module."):
                new_key = key[len("module."):]  # 去掉 "module."
            else:
                new_key = key
            new_state_dict[new_key] = value
        #raise ValueError("key是{},value是{}".format(key,value))
        state_dict = new_state_dict

        # 获取当前模型中所有的参数键
        model_state_dict = self.image_encoder.state_dict()
        # 过滤出匹配的键
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}

        self.image_encoder.load_state_dict(filtered_state_dict)  # 加载到模型中
        # 冻结参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        #kan
        self.device=torch.device('cuda:0')
        self.kan = KANLayer(in_dim=7, out_dim=24).to(self.device)   #设置输入维度和输出维度
        #self.mlp=TabEncoder().to(self.device)

        #multiscale_fusion(MSCA)
        self.multiscale_fusion=MultiScale_Fusion(24)

        #Classifier
        self.tab_cls=nn.Linear(24, 1)
        self.img_cls = nn.Sequential(
            nn.Linear(24, 1),
        )
        self.fin_cls=nn.Linear(48, 1)
        self.sigmoid=nn.Sigmoid()
        #dropout
        self.dropout=nn.Dropout(p=0.2)

    def forward(self, i, tab):
        batch_size=i.shape[0]

        #img_encoder
        f,img_pred=self.image_encoder(i)
        #raise ValueError("img_pred是{}".format(img_pred))

        #表格数据特征提取
        
        query_tab, _, _, _ = self.kan(tab)
        #query_tab=self.mlp(tab)

        #多尺度注意力融合
        fea=self.multiscale_fusion(f,query_tab)

        #分类结果
        fea=fea.view(batch_size,-1)
        pred=self.fin_cls(fea)
        return pred


    def args_dict(self):
        model_args = {
            'info': 0,
        }

        return model_args
