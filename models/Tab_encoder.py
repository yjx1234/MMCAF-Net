
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import math

#kan
from .kan.KANLayer import *

class Tab_encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #kan
        self.device=torch.device('cuda:0')
        self.kan = KANLayer(in_dim=7, out_dim=24).to(self.device)   #设置输入维度和输出维度

        #Classifier
        self.tab_cls=nn.Linear(24, 1)

        #dropout
        self.dropout=nn.Dropout(p=0.2)


    def forward(self,tab):
        #表格数据特征提取
        query_tab, _, _, _ = self.kan(tab)

        #分类结果
        tab_pred=self.tab_cls(query_tab)
        return tab_pred

    def args_dict(self):
        model_args = {
            'info': 0,
        }

        return model_args
