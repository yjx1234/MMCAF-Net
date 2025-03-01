#

import torch
import torch.nn as nn

from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F
import numpy as np
import math

#penet_encoder
from .penet import PENet

#ISBI_utils
from .model_utils.BFPU_conv import BFPUBlock_3D
from .model_utils.multiscale import CAB_3D,SAB_3D,DCFB,UCB



class Img_new(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        #penet
        # 去掉 state_dict 中的 "module." 前缀
        new_state_dict = {}
        self.encoder=PENet(50)
        state_dict=torch.load("/home/vesselseg3/yujianxun/train_result/penet/best.pth.tar")['model_state']
        for key, value in state_dict.items():
            # 去掉 "module." 前缀
            if key.startswith("module."):
                new_key = key[len("module."):]  # 去掉 "module."
            else:
                new_key = key
            new_state_dict[new_key] = value

        # 获取当前模型中所有的参数键
        model_state_dict = self.encoder.state_dict()
        # 过滤出匹配的键
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}

        self.encoder.load_state_dict(filtered_state_dict) #加载参数
        #冻结参数
        for param in self.encoder.parameters():
            param.requires_grad=False

    
        self.device=torch.device('cuda:0')
        
        #norm+dropout
        self.scale1 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(0.3)  # 多尺度特征中加入 Dropout
        )

        self.scale2 = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout(0.3)  # 多尺度特征中加入 Dropout
        )

        self.scale3 = nn.Sequential(
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)  # 多尺度特征中加入 Dropout
            )

        #Classifier
        self.fused_cls = nn.Sequential(
            nn.Linear(24, 1),
        )

        #dropout
        self.dropout=nn.Dropout(p=0.2)

        #E3D-MSCA
        self.sab = SAB_3D().to(self.device)
        self.cab3 = CAB_3D(in_channels=1024).to(self.device)   #input_channel改成自己的输入通道
        self.dcfb3 = DCFB(in_channels=1024, out_channels=1024, stride=1, kernel_sizes=[1,3,5], expansion_factor=2,
                          dw_parallel=True, add=True, activation='relu6').to(self.device)
        self.cab2 = CAB_3D(in_channels=512).to(self.device)  # input_channel改成自己的输入通道
        self.dcfb2 = DCFB(in_channels=512, out_channels=512, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2,
                          dw_parallel=True, add=True, activation='relu6').to(self.device)
        self.cab1 = CAB_3D(in_channels=256).to(self.device)  # input_channel改成自己的输入通道
        self.dcfb1 = DCFB(in_channels=256, out_channels=256, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2,
                          dw_parallel=True, add=True, activation='relu6').to(self.device)

        #EUCB
        eucb_ks = 3  # kernel size for eucb
        self.ucb2 = UCB(in_channels=1024, out_channels=1024, kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.ucb1 = UCB(in_channels=512, out_channels=512, kernel_size=eucb_ks, stride=eucb_ks // 2)

        #BFPU
        self.bfpu1=BFPUBlock_3D(in_dim1=512)
        self.bfpu2=BFPUBlock_3D(in_dim1=1024)


        #reshape
        self.maxpool=nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(1024, 24),
                nn.ReLU(),
                nn.Dropout(0.5),  # 分类头中加入 Dropout
        )
        #self.fc = nn.Linear(1024, 24)

    def forward(self, i):
        # 多尺度金字塔输入
        img_fea_norm=[]
        pred,img_fea=self.encoder(i)  #[4,256,6,48,48],[4,512,3,24,24],[4,1024,2,12,12]
        img_fea_norm.append(self.scale1(img_fea[0]))
        img_fea_norm.append(self.scale2(img_fea[1]))
        img_fea_norm.append(self.scale3(img_fea[2]))
    #stage 1   [4,1024,2,12,12]
        batch_size=i.shape[0]

        #E3D-MSCA1
        d3 = self.cab3(img_fea_norm[2]) * img_fea_norm[2]
        #raise ValueError("d3是{}".format(d3.shape))
        d3 = self.sab(d3) * d3
        #raise ValueError("d3是{}".format(d3.shape))
        d3 = self.dcfb3(d3)    #[4,1024,2,12,12]    第二个是[4,512,3,24,24]
        #raise ValueError("d3是{}".format(d3.shape))

        #UCB1
        d2=self.ucb2(d3)   #[4,512,4,24,24]
        d2=F.interpolate(d2, size=(3, 24, 24), mode='trilinear', align_corners=True) #[4,512,3,24,24]

    # stage 2   [4,512,3,24,24]

        #E3D-MSCA2
        #img_fea[1]=img_fea[1]+d2
        fea2=img_fea_norm[1]+d2
        d2 = self.cab2(fea2) * fea2
        d2 = self.sab(d2) * d2
        d2 = self.dcfb2(d2)

        #UCB1
        d1=self.ucb1(d2)   #[4,256,6,48,48]


    # stage 3   [4,256,6,48,48]

        #E3D-MSCA3
        #img_fea[0]=img_fea[0]+d1
        fea3=img_fea_norm[0]+d1
        d1 = self.cab1(fea3) * fea3
        d1 = self.sab(d1) * d1
        d1 = self.dcfb1(d1)



        # BFPU
        bf1=self.bfpu1(d2,d1)   #[4,512,3,24,24]
        bf2=self.bfpu2(d3,bf1)  #[4,1024,2,12,12]

        #reshape
        f=self.maxpool(bf2)
        f=torch.flatten(f, start_dim=1)  # 压平为 [batch_size, channels]
        f=self.fc(f)


        #分类结果
        pred = self.fused_cls(f)
        return f,pred



    def args_dict(self):
        model_args = {
            'info': 0,
        }

        return model_args
